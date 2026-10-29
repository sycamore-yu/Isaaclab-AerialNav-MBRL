"""Benchmark the unchanged Lee/RotorGroup compute against CUDA Graph replay.

This script needs PyTorch/CUDA, but does not launch Isaac Sim or modify any task.
It imports the production Zhang graph adapter, which captures ``compute`` with
fixed input storage; it does not replace, compile, or fuse controller equations.
Candidate timing includes all four input copies. Parity requires exact outputs
and identical CUDA RNG advancement, including after partial motor-state resets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import traceback
from pathlib import Path

import torch

from isaaclab_navrl.control import LeeRotorController
from isaaclab_navrl.zhang.controller import GraphedLeeRotorController


def make_inputs(num_envs, steps, device, seed):
    """Changing finite flight states, with Zhang's zero velocity-error interface."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = torch.randn(steps, num_envs, 13, generator=generator)
    state[..., 2] = state[..., 2].abs() + 0.5
    state[..., 3:7] /= state[..., 3:7].norm(dim=-1, keepdim=True)
    state[..., 7:10] *= 3.0
    state[..., 10:13] *= 2.0
    acceleration = (torch.randn(steps, num_envs, 3, generator=generator) * 10).clamp(-30, 30)
    yaw = (torch.rand(steps, num_envs, 1, generator=generator) * 2 - 1) * math.pi
    # Include hover and a zero desired acceleration vector in the Lee formula.
    # Random states above also exercise negative commands and rotor saturation.
    state[0] = 0
    state[0, :, 2] = 2.0
    state[0, :, 6] = 1.0
    acceleration[0] = 0
    yaw[0] = 0
    if steps > 1:
        acceleration[1] = torch.tensor([0.0, 0.0, -9.81])
    state, acceleration, yaw = state.to(device), acceleration.to(device), yaw.to(device)
    return [
        (state[index], state[index, :, 7:10], acceleration[index], yaw[index])
        for index in range(steps)
    ]


def compare_controllers(reference, candidate, inputs, device):
    """Check exact per-step values, reset isolation, and global RNG consumption."""
    output_names = ("commands", "thrusts", "moments", "throttle")
    max_abs_error = {name: 0.0 for name in output_names}
    exact = {name: True for name in output_names}
    rng_equal = True
    first_failure = None
    partial_resets = 0
    storage_stable = True

    def storage_addresses():
        return [tensor.data_ptr() for tensor in candidate.inputs] + [
            getattr(candidate, name).data_ptr() for name in output_names
        ]

    captured_addresses = storage_addresses()
    all_ids = torch.arange(reference.num_envs, device=device)
    reset_ids = all_ids[::3].to(torch.int32)
    reference.reset(all_ids)
    candidate.reset(all_ids)

    def compare_outputs(step, phase):
        nonlocal first_failure, storage_stable
        storage_stable &= storage_addresses() == captured_addresses
        for name in output_names:
            expected, actual = getattr(reference, name), getattr(candidate, name)
            error = (actual - expected).abs().amax().item()
            max_abs_error[name] = max(max_abs_error[name], error)
            is_equal = torch.equal(actual, expected)
            exact[name] &= is_equal
            if not is_equal and first_failure is None:
                first_failure = {"step": step, "phase": phase, "output": name, "max_abs_error": error}

    compare_outputs(-1, "initial_reset")
    for index, values in enumerate(inputs):
        # Check the public Lee optional arguments and both supported yaw shapes;
        # timing below still uses Zhang's explicit acceleration/yaw interface.
        state, velocity, acceleration, yaw = values
        interface_case = index % 11
        if interface_case in (1, 3):
            acceleration = None
        if interface_case in (2, 3):
            yaw = None
        elif interface_case == 4:
            yaw = yaw[:, 0]
        values = state, velocity, acceleration, yaw
        if index > 0 and index % 7 == 0:
            reference.reset(reset_ids)
            candidate.reset(reset_ids)
            partial_resets += 1
            compare_outputs(index, "partial_reset")
        before_rng = torch.cuda.get_rng_state(device)
        reference.compute(*values)
        expected_rng = torch.cuda.get_rng_state(device)
        torch.cuda.set_rng_state(before_rng, device)
        candidate.compute(*values)
        actual_rng = torch.cuda.get_rng_state(device)
        current_rng_equal = torch.equal(actual_rng, expected_rng)
        rng_equal &= current_rng_equal
        if not current_rng_equal and first_failure is None:
            first_failure = {"step": index, "phase": "compute", "output": "cuda_rng_state"}
        compare_outputs(index, "compute")
    return {
        "passed": all(exact.values()) and rng_equal and storage_stable,
        "steps": len(inputs), "partial_resets": partial_resets,
        "interfaces": ["explicit acceleration/yaw", "default acceleration", "default yaw", "both defaults", "yaw N or N1"],
        "reset_envs": len(reset_ids), "bitwise_equal": exact,
        "max_abs_error": max_abs_error, "cuda_rng_state_equal": rng_equal,
        "first_failure": first_failure, "captured_storage_addresses_stable": storage_stable,
        "tolerance": "exact torch.equal; no tolerance, relaxed precision, or formula changes",
    }


def timed_run(target, inputs, warmup, synchronized_calls, device):
    """Wall timing excludes setup/reset and includes graph input-copy overhead."""
    target.reset(torch.arange(target.num_envs, device=device))
    for index in range(warmup):
        target.compute(*inputs[index % len(inputs)])
    torch.cuda.synchronize(device)
    samples = []
    started = time.perf_counter()
    for values in inputs:
        if synchronized_calls:
            torch.cuda.synchronize(device)
            call_started = time.perf_counter()
        target.compute(*values)
        if synchronized_calls:
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - call_started) * 1000)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {"wall_s": elapsed, "mean_ms_per_compute": elapsed * 1000 / len(inputs)}
    if synchronized_calls:
        ordered = sorted(samples)
        result["per_call_ms"] = {
            "mean": statistics.fmean(samples), "median": statistics.median(samples),
            "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "minimum": ordered[0], "maximum": ordered[-1],
        }
    return result


def benchmark(reference, candidate, inputs, warmup, repeats, device):
    results = {}
    for label, synchronized_calls in (("batched", False), ("synchronized_per_call", True)):
        timings = {"eager": [], "cuda_graph_with_input_copies": []}
        for repeat in range(repeats):
            order = [("eager", reference), ("cuda_graph_with_input_copies", candidate)]
            if repeat % 2:
                order.reverse()
            for name, target in order:
                timings[name].append(timed_run(target, inputs, warmup, synchronized_calls, device))
        medians = {
            name: statistics.median(run["mean_ms_per_compute"] for run in runs)
            for name, runs in timings.items()
        }
        results[label] = {
            "runs": timings, "median_ms_per_compute": medians,
            "speedup": medians["eager"] / medians["cuda_graph_with_input_copies"],
            "cuda_sync": "before and after each compute" if synchronized_calls else "run boundaries only",
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parity_steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--output", type=Path, default=Path("logs/profile/controller_cuda_graph.json"))
    args = parser.parse_args()
    if min(args.num_envs, args.parity_steps, args.measured, args.repeats) < 1 or args.warmup < 0:
        parser.error("counts must be positive; warmup must be nonnegative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("the CUDA Graph candidate requires an available CUDA device")
    torch.cuda.set_device(device)
    device = torch.device("cuda", torch.cuda.current_device())

    import isaaclab_navrl.control as control_module
    import isaaclab_navrl.zhang.controller as graph_module

    result = {
        "candidate": "CUDA Graph capture of unchanged LeeRotorController.compute",
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "device": str(device), "device_name": torch.cuda.get_device_name(device),
        "num_envs": args.num_envs, "physics_dt": 0.01, "seed": args.seed,
        "controller_source_sha256": hashlib.sha256(Path(control_module.__file__).read_bytes()).hexdigest(),
        "adapter_source_sha256": hashlib.sha256(Path(graph_module.__file__).read_bytes()).hexdigest(),
        "warmup_per_timing_run": args.warmup, "measured_per_run": args.measured, "repeats": args.repeats,
        "scope": "Zhang interface: current velocity, explicit target acceleration and yaw; compute only",
        "timing_note": "candidate includes four input copies; excludes PhysX, apply, task, policy, backward, and optimizer",
        "state_note": "motor throttle persists across calls; partial reset uses the unchanged controller.reset",
        "precision": "original dtype and CUDA kernels; no torch.compile, TF32 setting change, or formula replacement",
    }
    exit_code = 0
    try:
        with torch.no_grad():
            inputs = make_inputs(args.num_envs, max(args.parity_steps, args.measured), device, args.seed)
            reference = LeeRotorController(args.num_envs, device, dt=0.01)
            captured = LeeRotorController(args.num_envs, device, dt=0.01)
            torch.cuda.synchronize(device)
            candidate = GraphedLeeRotorController(captured, inputs[0])
            result["capture_setup_wall_s"] = candidate.capture_setup_wall_s
            result["parity"] = compare_controllers(reference, candidate, inputs[: args.parity_steps], device)
            if result["parity"]["passed"]:
                result["timing"] = benchmark(
                    reference, candidate, inputs[: args.measured], args.warmup, args.repeats, device
                )
                result["status"] = "passed"
            else:
                result["status"] = "parity_failed"
                exit_code = 1
    except Exception as error:
        result["status"] = "error"
        result["exception"] = {"type": type(error).__name__, "message": str(error)}
        result["traceback"] = traceback.format_exc()
        exit_code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("CONTROLLER_BENCHMARK " + json.dumps(result, allow_nan=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
