#!/usr/bin/env python3
r"""Opt-in differential check and microbenchmark of Zhang policy_transition.

This imports the repository's pure Zhang modules without starting Isaac Lab.
It does not patch dynamics, train a policy, change training defaults, or select
an implementation for a running trainer. PyTorch with its compiler toolchain is
required; --help itself only needs the standard library.

Example (run in the same Python environment as training):

    TORCH_LOGS=graph_breaks,recompiles python scripts/benchmark_zhang_compile.py \
        --device cuda:0 --transitions 150 --repeats 5 \
        --output /tmp/zhang_compile_150.json

Both record_positions branches are checked. Every trial starts with fresh leaf
state/command tensors, retains ALL transition outputs, and calls backward once
after the complete rollout. No retain_graph, input mutation, CUDA graph step
markers, mixed precision, or reduce-overhead mode is used. Inputs, including
the parameter dictionary, stay alive and unchanged until backward finishes.

The validation objective uses fixed random +/-1 cotangents and a SUM, not an
environment mean that could conceal small gradients behind an absolute
tolerance. Finite-difference gradcheck is deliberately inappropriate here:
gradient_decay and the detached airmode define an intentional custom adjoint.

Initial FP32 gates are forward atol=1e-5/rtol=1e-4 and gradient
atol=2e-5/rtol=2e-4. They are NOT automatically relaxed for a long rollout or
the acos(cosine) near-hover case. A numeric failure exits nonzero and skips
timing unless --benchmark_on_mismatch is explicitly supplied; that option
never changes validation_passed or makes a candidate eligible.

First-use timings include forward AND backward compilation/cache loading;
they are not a measurement of compiler time alone. Hot timing alternates
eager/compiled order, rejects recompilation, and includes history stacking,
the same synthetic scalar objective, and backward. It excludes input cloning
and host result copies. A speedup is only for this pure-dynamics workload,
not the camera, PhysX, policy network, or end-to-end training. GPU competition
can change all timings, even CUDA-event durations.

Official API references (version pinned to the target runtime):
https://docs.pytorch.org/docs/2.11/generated/torch.compile.html
https://docs.pytorch.org/docs/2.11/generated/torch.compiler.set_stance.html
https://docs.pytorch.org/docs/2.11/generated/torch.compiler.cudagraph_mark_step_begin.html
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
import traceback
import types
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

NUM_ENVS = 1024
STATE_FIELDS = ("position", "velocity", "acceleration", "actuator")


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--transitions", type=int, default=150, help="Connected policy steps before one backward; N is fixed at 1024."
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Additional full-rollout warmups per implementation/branch."
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; existing files are refused.")
    parser.add_argument(
        "--zhang_dir", type=Path, default=Path(__file__).resolve().parents[1] / "src/isaaclab_navrl/zhang"
    )
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--benchmark_on_mismatch", action="store_true")
    parser.add_argument("--forward_atol", type=float, default=1e-5)
    parser.add_argument("--forward_rtol", type=float, default=1e-4)
    parser.add_argument("--gradient_atol", type=float, default=2e-5)
    parser.add_argument("--gradient_rtol", type=float, default=2e-4)
    args = parser.parse_args(argv)
    if args.transitions < 2 or args.warmup < 1 or args.repeats < 1:
        parser.error("transitions must be >=2; warmup and repeats must be >=1")
    for name in ("forward_atol", "gradient_atol", "forward_rtol", "gradient_rtol"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name.endswith("atol") and value == 0):
            parser.error(f"{name} must be finite and {'positive' if name.endswith('atol') else 'nonnegative'}")
    return args


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically preserve completed checks even if a later compile/run fails."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def load_source(zhang_dir: Path) -> tuple[Any, Any]:
    """Keep package-level Isaac Lab registration out of this pure torch test."""
    for name in ("config.py", "dynamics.py", "model.py"):
        if not (zhang_dir / name).is_file():
            raise FileNotFoundError(zhang_dir / name)
    package_name = "_zhang_compile_benchmark_source"
    package = types.ModuleType(package_name)
    package.__path__ = [str(zhang_dir)]
    package.__package__ = package_name
    package.__spec__ = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
    sys.modules[package_name] = package
    return (importlib.import_module(f"{package_name}.config"), importlib.import_module(f"{package_name}.dynamics"))


def make_fixture(
    torch: Any, dynamics: Any, device: Any, case: str, transitions: int, seed: int, *, parameter_seed: int | None = None
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def normal(shape: tuple[int, ...], scale: float = 1.0) -> Any:
        return (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(device)

    position = normal((NUM_ENVS, 3), 0.5)
    position[:, 2] += 1.5
    velocity = normal((NUM_ENVS, 3), 0.8)
    acceleration = normal((NUM_ENVS, 3), 0.3)
    actuator = normal((NUM_ENVS, 3), 0.75)
    quaternion = normal((NUM_ENVS, 4))
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    rotation = dynamics.quaternion_matrix(quaternion).contiguous()
    commands = actuator.unsqueeze(0) + normal((transitions, NUM_ENVS, 3), 0.4)
    if case == "tilted_near_collinear":
        # Four equal-size groups: exact equality and perturbations 1e-4, 1e-3, 1e-2.
        epsilon = torch.tensor((0.0, 1e-4, 1e-3, 1e-2), device=device).repeat(NUM_ENVS // 4)
        commands = actuator.unsqueeze(0) + normal((transitions, NUM_ENVS, 3)) * epsilon[None, :, None]

    # A separately seeded parameter draw makes the resample test change ONLY
    # tensor values, not state/command/rotation values, dict keys or metadata.
    generator.manual_seed(parameter_seed if parameter_seed is not None else seed + 1000)

    def uniform(shape: tuple[int, ...], low: float, high: float) -> Any:
        return (low + (high - low) * torch.rand(shape, generator=generator, dtype=torch.float32)).to(device)

    parameters = {
        "wind": normal((NUM_ENVS, 3), 0.2),
        "dg": normal((NUM_ENVS, 3), 0.1),
        "drag_2": uniform((NUM_ENVS, 2), 0.02, 0.12),
        "z_drag_coef": uniform((NUM_ENVS, 1), 0.8, 1.2),
        "pitch_ctl_delay": torch.full((NUM_ENVS, 1), 12.0, device=device),
        "control_gain_scale": uniform((NUM_ENVS, 1), 0.6, 1.4),
        "mass_scale": uniform((NUM_ENVS, 1), 0.9, 1.1),
        "inertia_scale": uniform((NUM_ENVS, 1), 0.9, 1.1),
        "thrust_scale": uniform((NUM_ENVS, 1), 0.9, 1.1),
    }
    if case == "nominal_hover":
        velocity.zero_()
        acceleration.zero_()
        actuator.zero_()
        commands.zero_()
        rotation = torch.eye(3, device=device).unsqueeze(0).repeat(NUM_ENVS, 1, 1)
        for name in ("wind", "dg", "drag_2"):
            parameters[name].zero_()
        for name in ("z_drag_coef", "control_gain_scale", "mass_scale", "inertia_scale", "thrust_scale"):
            parameters[name].fill_(1.0)
    return {
        "state": (position, velocity, acceleration, actuator),
        "commands": commands,
        "rotation": rotation,
        "parameters": parameters,
    }


def make_cotangents(torch: Any, fixture: dict[str, Any], record_positions: bool, seed: int) -> dict[str, Any]:
    shape = tuple(fixture["commands"].shape)
    shapes = {name: shape for name in STATE_FIELDS}
    if record_positions:
        shapes["recorded_positions"] = (shape[0], NUM_ENVS, 10, 3)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {
        name: (2 * torch.randint(0, 2, size, generator=generator) - 1).to(
            device=fixture["commands"].device, dtype=torch.float32
        )
        for name, size in shapes.items()
    }


def synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_trial(
    torch: Any,
    dynamics: Any,
    transition: Any,
    cfg: Any,
    fixture: dict[str, Any],
    record_positions: bool,
    cotangents: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One connected graph, one backward; detached CPU snapshots only for checks."""
    device = fixture["commands"].device
    leaves = tuple(value.detach().clone().requires_grad_(True) for value in fixture["state"])
    commands = tuple(value.detach().clone().requires_grad_(True) for value in fixture["commands"])
    state = dynamics.PointMassState(*leaves)
    history = {name: [] for name in STATE_FIELDS}
    if record_positions:
        history["recorded_positions"] = []
    synchronize(torch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_bytes = torch.cuda.memory_allocated(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    started = time.perf_counter()
    for command in commands:
        result = transition(
            state, command, fixture["rotation"], fixture["parameters"], cfg, record_positions=record_positions
        )
        if record_positions:
            state, positions = result
            history["recorded_positions"].append(positions)
        else:
            state = result
        if not isinstance(state, dynamics.PointMassState):
            raise TypeError(f"Expected PointMassState output, got {type(state).__name__}")
        for name in STATE_FIELDS:
            # Retain the original returned tensors, not immediate clones. This
            # exposes output-lifetime problems across repeated compiled calls.
            history[name].append(getattr(state, name))
    outputs = {name: torch.stack(values) for name, values in history.items()}
    if cotangents is None:
        loss = sum(value.square().mean() for value in outputs.values())
    else:
        loss = sum((value * cotangents[name]).sum() for name, value in outputs.items())
    loss.backward()
    if device.type == "cuda":
        end_event.record()
    synchronize(torch, device)
    elapsed = time.perf_counter() - started
    trial = {
        "wall_s": elapsed,
        "cuda_event_s": None,
        "allocated_baseline_bytes": None,
        "peak_allocated_bytes": None,
        "peak_increment_bytes": None,
    }
    if device.type == "cuda":
        trial.update(
            cuda_event_s=start_event.elapsed_time(end_event) / 1000.0,
            allocated_baseline_bytes=baseline_bytes,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_increment_bytes=torch.cuda.max_memory_allocated(device) - baseline_bytes,
        )
    if any(value.grad is None for value in leaves + commands):
        raise RuntimeError("Missing gradient for an initial-state field or policy command")
    # All validation/synchronizing .item()/.cpu() operations are outside timing.
    trial["loss_finite"] = bool(torch.isfinite(loss).item())
    state_gradients = torch.stack([value.grad for value in leaves])
    command_gradients = torch.stack([value.grad for value in commands])
    trial["gradients_finite"] = bool(
        (torch.isfinite(state_gradients).all() & torch.isfinite(command_gradients).all()).item()
    )
    if cotangents is not None:
        trial["outputs"] = {name: value.detach().cpu() for name, value in outputs.items()}
        trial["gradients"] = {name: value.grad.detach().cpu() for name, value in zip(STATE_FIELDS, leaves, strict=True)}
        trial["gradients"]["commands"] = command_gradients.detach().cpu()
        trial["recorded_endpoint_exact"] = (
            bool(torch.equal(outputs["recorded_positions"][:, :, -1], outputs["position"]))
            if record_positions
            else None
        )
    return trial


def tensor_comparison(torch: Any, expected: Any, actual: Any, tolerance: Tolerance) -> dict[str, Any]:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return {
            "passed": False,
            "error": "shape/dtype mismatch",
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
            "expected_dtype": str(expected.dtype),
            "actual_dtype": str(actual.dtype),
        }
    ref, got = expected.double(), actual.double()
    ref_finite, got_finite = torch.isfinite(ref), torch.isfinite(got)
    finite = bool((ref_finite & got_finite).all().item())
    result = {
        "shape": list(ref.shape),
        "passed": False,
        "finite": finite,
        "reference_nonfinite": int((~ref_finite).sum().item()),
        "candidate_nonfinite": int((~got_finite).sum().item()),
    }
    if not finite:
        return result
    difference = (got - ref).abs()
    permitted = tolerance.atol + tolerance.rtol * ref.abs()
    mismatches = int((difference > permitted).sum().item())
    reference_norm = float(ref.norm().item())
    difference_norm = float(difference.norm().item())
    flat_index = int(difference.argmax().item())
    coordinates = []
    for dimension in reversed(ref.shape):
        coordinates.append(flat_index % dimension)
        flat_index //= dimension
    coordinates.reverse()
    result.update(
        passed=mismatches == 0,
        mismatch_count=mismatches,
        elements=ref.numel(),
        max_abs=float(difference.max().item()),
        mean_abs=float(difference.mean().item()),
        max_scaled_error=float((difference / permitted).max().item()),
        reference_l2=reference_norm,
        difference_l2=difference_norm,
        relative_l2=difference_norm / reference_norm if reference_norm else (0.0 if difference_norm == 0 else None),
    )
    result.update(
        max_abs_index=coordinates,
        expected_at_max_abs=float(ref[tuple(coordinates)].item()),
        actual_at_max_abs=float(got[tuple(coordinates)].item()),
    )
    return result


def summarize_timings(trials: list[dict[str, Any]], transitions: int) -> dict[str, Any]:
    wall = [trial["wall_s"] for trial in trials]
    cuda = [trial["cuda_event_s"] for trial in trials if trial["cuda_event_s"] is not None]
    median = statistics.median(wall)
    return {
        "samples": len(wall),
        "median_wall_s": median,
        "min_wall_s": min(wall),
        "max_wall_s": max(wall),
        "wall_stdev_s": statistics.stdev(wall) if len(wall) > 1 else None,
        "median_cuda_event_s": statistics.median(cuda) if cuda else None,
        "environment_policy_transitions_per_s": NUM_ENVS * transitions / median,
        "environment_physics_substeps_per_s": NUM_ENVS * transitions * 10 / median,
        "all_losses_finite": all(trial["loss_finite"] for trial in trials),
        "all_gradients_finite": all(trial["gradients_finite"] for trial in trials),
        "trials": trials,
    }


def execute(args: argparse.Namespace, report: dict[str, Any]) -> int:
    import torch

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment")
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("Only cuda or cpu devices are supported")
    if not hasattr(torch.compiler, "set_stance"):
        raise RuntimeError("This benchmark requires torch.compiler.set_stance (target runtime: PyTorch 2.11)")
    config_module, dynamics = load_source(args.zhang_dir.resolve())
    cfg = config_module.ZhangTrainCfg(physics_dt=0.01, policy_dt=0.1, grad_decay=0.4)
    cfg.validate()
    if round(cfg.policy_dt / cfg.physics_dt) != 10:
        raise RuntimeError("The benchmark requires exactly ten physics substeps per policy transition")
    report["runtime"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_logs": os.environ.get("TORCH_LOGS"),
        "num_threads": torch.get_num_threads(),
    }
    report["source"] = {
        "directory": str(args.zhang_dir.resolve()),
        "sha256": {
            name: hashlib.sha256((args.zhang_dir / name).read_bytes()).hexdigest()
            for name in ("config.py", "dynamics.py", "model.py")
        },
    }
    report["config"] = asdict(cfg)
    report["compile"] = {
        "call": "torch.compile(policy_transition, fullgraph=True, dynamic=False)",
        "backend": "inductor",
        "mode": "default",
        "hot_stance": "fail_on_recompile",
        "cache_may_exist_from_an_earlier_process": True,
    }
    eager = dynamics.policy_transition
    compiled = torch.compile(dynamics.policy_transition, fullgraph=True, dynamic=False)
    forward_tolerance = Tolerance(args.forward_atol, args.forward_rtol)
    gradient_tolerance = Tolerance(args.gradient_atol, args.gradient_rtol)
    report["tolerances"] = {"forward": asdict(forward_tolerance), "gradient": asdict(gradient_tolerance)}
    write_report(args.output, report)

    # The resample pair has identical non-parameter inputs. All cases use the
    # same dictionary schema, sizes, dtype and requires_grad pattern.
    checks = (
        ("randomized_flight", 1, None),
        ("parameter_resample", 1, args.seed + 2000),
        ("nominal_hover", 1, None),
        ("tilted_near_collinear", 1, None),
        ("connected_rollout", args.transitions, None),
    )
    callable_by_branch: dict[bool, bool] = {}
    for record_positions in (False, True):
        callable_by_branch[record_positions] = True
        for case, transitions, parameter_seed in checks:
            entry: dict[str, Any] = {
                "case": case,
                "record_positions": record_positions,
                "transitions": transitions,
                "passed": False,
            }
            print(f"[check] record_positions={record_positions} {case} H={transitions}", flush=True)
            try:
                fixture = make_fixture(
                    torch, dynamics, device, case, transitions, args.seed, parameter_seed=parameter_seed
                )
                cotangents = make_cotangents(torch, fixture, record_positions, args.seed + 3000)
                reference = run_trial(torch, dynamics, eager, cfg, fixture, record_positions, cotangents)
                candidate = run_trial(torch, dynamics, compiled, cfg, fixture, record_positions, cotangents)
                entry["eager_first_use_wall_s"] = reference["wall_s"]
                entry["compiled_first_use_wall_s"] = candidate["wall_s"]
                entry["forward"] = {
                    name: tensor_comparison(torch, expected, candidate["outputs"][name], forward_tolerance)
                    for name, expected in reference["outputs"].items()
                }
                entry["gradient"] = {
                    name: tensor_comparison(torch, expected, candidate["gradients"][name], gradient_tolerance)
                    for name, expected in reference["gradients"].items()
                }
                entry["recorded_endpoint_exact"] = {
                    "eager": reference["recorded_endpoint_exact"],
                    "compiled": candidate["recorded_endpoint_exact"],
                }
                entry["passed"] = (
                    all(
                        metric["passed"] for group in (entry["forward"], entry["gradient"]) for metric in group.values()
                    )
                    and reference["loss_finite"]
                    and candidate["loss_finite"]
                    and reference["gradients_finite"]
                    and candidate["gradients_finite"]
                    and (
                        not record_positions
                        or (reference["recorded_endpoint_exact"] and candidate["recorded_endpoint_exact"])
                    )
                )
                failed_fields = [
                    f"{group}.{name}"
                    for group in ("forward", "gradient")
                    for name, metric in entry[group].items()
                    if not metric["passed"]
                ]
                print(
                    f"  {'PASS' if entry['passed'] else 'FAIL'}; fields={failed_fields}; "
                    f"compiled first-use {candidate['wall_s']:.3f}s",
                    flush=True,
                )
                del reference, candidate, cotangents, fixture
            except Exception as error:
                entry["error"] = f"{type(error).__name__}: {error}"
                entry["traceback"] = traceback.format_exc()
                callable_by_branch[record_positions] = False
                print(f"  ERROR: {entry['error']}", flush=True)
            report["checks"].append(entry)
            write_report(args.output, report)
            if not callable_by_branch[record_positions]:
                break

    report["validation_passed"] = len(report["checks"]) == 2 * len(checks) and all(
        entry["passed"] for entry in report["checks"]
    )
    may_benchmark = (
        all(callable_by_branch.values())
        and not args.check_only
        and (report["validation_passed"] or args.benchmark_on_mismatch)
    )
    if may_benchmark:
        fixture = make_fixture(torch, dynamics, device, "connected_rollout", args.transitions, args.seed)
        for record_positions in (False, True):
            print(f"[warmup] record_positions={record_positions}, H={args.transitions}", flush=True)
            for _ in range(args.warmup):
                run_trial(torch, dynamics, eager, cfg, fixture, record_positions)
                run_trial(torch, dynamics, compiled, cfg, fixture, record_positions)
            samples: dict[str, list[dict[str, Any]]] = {"eager": [], "compiled": []}
            order_log = []
            for repeat in range(args.repeats):
                order = ("eager", "compiled") if repeat % 2 == 0 else ("compiled", "eager")
                order_log.append(list(order))
                for name in order:
                    print(
                        f"[time] record_positions={record_positions} repeat={repeat + 1}/{args.repeats} {name}",
                        flush=True,
                    )
                    # Warmup must have captured leaf and recurrent non-leaf
                    # states. A guard miss now invalidates hot timing instead
                    # of hiding compilation or eager fallback in the sample.
                    with torch.compiler.set_stance("fail_on_recompile"):
                        samples[name].append(
                            run_trial(
                                torch, dynamics, eager if name == "eager" else compiled, cfg, fixture, record_positions
                            )
                        )
            eager_summary = summarize_timings(samples["eager"], args.transitions)
            compiled_summary = summarize_timings(samples["compiled"], args.transitions)
            benchmark = {
                "record_positions": record_positions,
                "transitions": args.transitions,
                "order": order_log,
                "eager": eager_summary,
                "compiled": compiled_summary,
                "median_wall_speedup": eager_summary["median_wall_s"] / compiled_summary["median_wall_s"],
                "median_paired_wall_speedup": statistics.median(
                    [
                        ref["wall_s"] / got["wall_s"]
                        for ref, got in zip(samples["eager"], samples["compiled"], strict=True)
                    ]
                ),
            }
            report["benchmarks"].append(benchmark)
            write_report(args.output, report)
            print(
                f"  median wall speedup {benchmark['median_wall_speedup']:.3f}x "
                f"(numeric gate={'PASS' if report['validation_passed'] else 'FAIL'})",
                flush=True,
            )
    else:
        report["benchmark_skipped_reason"] = (
            "check_only"
            if args.check_only
            else "An eager/compiled call raised an error"
            if not all(callable_by_branch.values())
            else "Numeric validation failed; --benchmark_on_mismatch was not supplied"
        )
    report["candidate_eligible_for_further_training_profile"] = (
        report["validation_passed"]
        and bool(report["benchmarks"])
        and all(
            item[name]["all_losses_finite"] and item[name]["all_gradients_finite"]
            for item in report["benchmarks"]
            for name in ("eager", "compiled")
        )
    )
    report["status"] = "completed" if report["validation_passed"] else "validation_failed"
    if report["benchmarks"] and not all(
        item[name]["all_losses_finite"] and item[name]["all_gradients_finite"]
        for item in report["benchmarks"]
        for name in ("eager", "compiled")
    ):
        report["status"] = "benchmark_failed"
    return 0 if report["status"] == "completed" else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the output before any expensive work; never overwrite evidence.
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write("{}\n")
    except FileExistsError:
        print(f"Refusing existing output: {args.output}", file=sys.stderr)
        return 1
    report: dict[str, Any] = {
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "num_envs": NUM_ENVS,
        "physics_hz": 100,
        "policy_hz": 10,
        "physics_substeps": 10,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "checks": [],
        "benchmarks": [],
        "validation_passed": False,
        "candidate_eligible_for_further_training_profile": False,
        "notes": [
            "This is a synthetic pure-dynamics differential test, not training acceptance.",
            "First-use includes forward/backward compilation or cache loading; it is not isolated compiler time.",
            "Hot workload retains every transition/substep output and performs one backward for the entire chain.",
            "Initial state fields and each command require gradients to test all four state adjoints explicitly.",
            "No PhysX alignment, episode resets, policy CNN, sensors, optimizer or real training loss is timed.",
            "Concurrent GPU jobs affect wall and CUDA-event timings; neither is an isolated throughput guarantee.",
            "A faster microbenchmark does not establish an end-to-end training improvement.",
            "The source acos near one may amplify rounding; failures are not automatically tolerated.",
        ],
    }
    write_report(args.output, report)
    try:
        exit_code = execute(args, report)
    except (Exception, KeyboardInterrupt) as error:
        report.update(
            status="error",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            candidate_eligible_for_further_training_profile=False,
        )
        print(report["error"], file=sys.stderr, flush=True)
        exit_code = 1
    report["finished_at"] = datetime.now(UTC).isoformat()
    write_report(args.output, report)
    print(f"JSON: {args.output}; status={report['status']}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
