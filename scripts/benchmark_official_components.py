"""Compare official-component changes with an immutable source snapshot.

Component timings include the whole observation-delivery call. Training mode
runs the existing synchronized benchmark child in ABBA order on the same GPU.
Neither mode stops other processes or changes task/training parameters.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def components(args, root, baseline):
    import torch
    from isaaclab.sensors.ray_caster import patterns
    from isaaclab.utils.math import unproject_depth

    from isaaclab_navrl.tasks.navigation import randomization

    reference = load_module(
        "_official_components_baseline_randomization",
        baseline / "src/isaaclab_navrl/tasks/navigation/randomization.py",
    )
    device = args.device
    cfg = randomization.SimToRealCfg(enabled=True)
    checks = {}
    previous = None
    # Exercise buffer wraparound, mixed delays, partial resets and repeated reads.
    objects = [module.PolicyPerturbation(cfg, 8, device, 417, policy_dt=0.05) for module in (reference, randomization)]
    for value in objects:
        value.reset(torch.arange(8, device=device))
    saved = []
    for step in range(16):
        if step in (4, 9):
            for value in objects:
                value.reset(torch.tensor([0, 3, 7], device=device))
        sample = torch.arange(56, device=device, dtype=torch.float32).reshape(8, 7) + step
        outputs = [value.observe(step, lambda sample=sample: sample) for value in objects]
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(objects[0].observation_age_s, objects[1].observation_age_s, rtol=0, atol=0)
        assert torch.equal(objects[0].generator.get_state(), objects[1].generator.get_state())
        assert objects[1].observe(step, lambda sample=sample: sample + 100).data_ptr() == outputs[1].data_ptr()
        saved.append((outputs[1], outputs[1].clone()))
        sample.add_(10000)
        if previous is not None:
            torch.testing.assert_close(previous[0], previous[1], rtol=0, atol=0)
        previous = (objects[1].latest_observation, objects[1].latest_observation.clone())
    for actual, expected in saved:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    checks["delay"] = {
        "status": "passed",
        "steps": 16,
        "environments": 8,
        "max_abs_error": 0,
        "rng_equal": True,
        "retained_outputs_stable": True,
    }
    del objects, saved, outputs, previous
    gc.collect()

    # The current camera uses half-pixel centers; the official unproject helper
    # uses integer coordinates. This is an executable negative replacement test.
    height, width = 120, 160
    intrinsics = torch.tensor([[138.5636727978, 0, 80.125], [0, 138.5636727978, 60.125], [0, 0, 1]], device=device)
    depth = torch.full((1, height, width), 10.0, device=device)
    official = unproject_depth(depth, intrinsics)
    y, x = torch.meshgrid(
        torch.arange(height, device=device) + 0.5, torch.arange(width, device=device) + 0.5, indexing="ij"
    )
    rays = torch.linalg.solve(intrinsics, torch.stack((x, y, torch.ones_like(x))).reshape(3, -1))
    expected = (rays * 10).T.reshape(height, width, 3).transpose(0, 1).reshape(1, -1, 3)
    checks["unproject_depth_direct_replacement"] = {
        "status": "rejected",
        "max_point_error_m": float((official - expected).abs().max()),
        "reason": "official integer pixels differ from the current half-pixel centers",
    }
    assert checks["unproject_depth_direct_replacement"]["max_point_error_m"] > 0.03

    camera_cfg = patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
        intrinsics.cpu().reshape(-1).tolist(), width=width, height=height
    )
    calibration = intrinsics[None].repeat(64, 1, 1)
    starts, directions = patterns.pinhole_camera_pattern(camera_cfg, calibration, device)
    original = (starts.contiguous(), directions.contiguous())
    shared = tuple(value[:1].clone().expand_as(value) for value in original)
    for actual, expected in zip(shared, original, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    import warp as wp

    wp.init()
    views = tuple(wp.from_torch(value, dtype=wp.vec3f) for value in shared)
    checks["shared_native_rays"] = {
        "status": "passed",
        "batch_size": 64,
        "rays_per_camera": height * width,
        "original_storage_bytes": sum(value.untyped_storage().nbytes() for value in original),
        "shared_storage_bytes": sum(value.untyped_storage().nbytes() for value in shared),
        "warp_strides_bytes": [list(value.strides) for value in views],
        "max_abs_error": 0,
    }
    del starts, directions, original, shared, views, depth, official, expected, rays, calibration
    gc.collect()

    timings = []
    sample = torch.linspace(0, 1, args.num_envs * 19405, device=device).reshape(args.num_envs, 19405)
    tick = 0
    for label, module in (
        ("baseline", reference),
        ("candidate", randomization),
        ("candidate", randomization),
        ("baseline", reference),
    ):
        value = module.PolicyPerturbation(cfg, args.num_envs, device, 417, policy_dt=0.05)
        value.reset(torch.arange(args.num_envs, device=device))
        for _ in range(8):
            value.observe(tick, lambda: sample)
            tick += 1
        torch.cuda.synchronize(device)
        start_allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        rounds = []
        for _ in range(5):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for _ in range(40):
                value.observe(tick, lambda: sample)
                tick += 1
            torch.cuda.synchronize(device)
            rounds.append((time.perf_counter() - started) * 1000 / 40)
        timings.append(
            {
                "source": label,
                "round_ms_per_observe": rounds,
                "median_ms": statistics.median(rounds),
                "incremental_peak_bytes": torch.cuda.max_memory_allocated(device) - start_allocated,
            }
        )
        del value
        gc.collect()
    return {
        "mode": "components",
        "device": device,
        "num_envs": args.num_envs,
        "observation_values": 19405,
        "checks": checks,
        "timings": timings,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "timing": "synchronized wall time, 8 warmup calls then five 40-call blocks per ABBA leg",
    }


def training(args, root, baseline):
    helper = load_module("_official_components_benchmark", root / "scripts/benchmark_backends.py")
    helper.NUM_ENVS = args.num_envs
    runs = []
    output = args.output.parent
    sequence = (
        (("baseline", baseline), ("candidate", root), ("candidate", root), ("baseline", baseline))
        if args.sequence == "abba"
        else (("candidate", root),)
    )
    for index, (label, source) in enumerate(sequence):
        run_dir = output / f"train_{index}_{label}"
        run_dir.mkdir(parents=True, exist_ok=False)
        train_dir = run_dir / "training"
        argv = helper.training_arguments(train_dir, "newton_mjwarp", "warp", args.updates, args.device)
        # Use the current fS1/40 workload for both implementations.
        argv[argv.index("--num_dynamic_obstacles") + 1] = "40"
        command = [sys.executable, str(source / "scripts/benchmark_backends.py"), "--_train_child", *argv]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(source / "src")
        env["PYTHONUNBUFFERED"] = "1"
        env["NAVRL_STAGE_STARTED_MONOTONIC"] = str(time.monotonic())
        env["WARP_CACHE_PATH"] = str(output / "warp_cache")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(output / "inductor_cache")
        run = {
            "label": label,
            "source": str(source),
            "command": command,
            "gpu_before": helper.gpu_snapshot(),
            "memory_samples_mib": [],
        }
        print(
            f"Starting {index + 1}/{len(sequence)}: {label}, {args.num_envs} environments, {args.updates} updates",
            flush=True,
        )
        with (run_dir / "console.log").open("w") as log:
            process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            run["pid"] = process.pid
            while process.poll() is None:
                usage = helper.capture(
                    ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"]
                )
                for line in usage.get("stdout", "").splitlines():
                    fields = line.split(",")
                    if len(fields) == 2 and fields[0].strip() == str(process.pid):
                        run["memory_samples_mib"].append(float(fields[1]))
                time.sleep(1)
        run["returncode"] = process.returncode
        run["peak_process_memory_mib"] = max(run["memory_samples_mib"], default=None)
        run["gpu_after"] = helper.gpu_snapshot()
        run["training_dir"] = str(train_dir)
        for filename in ("training_result.json", "config.json"):
            path = train_dir / filename
            if path.exists():
                run[filename] = json.loads(path.read_text())
        timing_file = run_dir / "timing.json"
        if timing_file.exists():
            run["timing"] = json.loads(timing_file.read_text())
        updates = train_dir / "updates.jsonl"
        if updates.exists():
            run["updates"] = [json.loads(line) for line in updates.read_text().splitlines() if line.strip()]
        runs.append(run)
        write_json(args.output, {"mode": "training", "runs": runs, "complete": False})
        print(f"Finished {label}: exit={process.returncode}", flush=True)
        if process.returncode:
            raise RuntimeError(f"Training failed; inspect {run_dir / 'console.log'}")
    return {
        "mode": "training",
        "runs": runs,
        "complete": True,
        "num_envs": args.num_envs,
        "horizon": 150,
        "warmup_updates": args.warmup,
        "baseline": str(baseline),
        "candidate": str(root),
        "protocol": f"{args.sequence.upper()}, seed 0, fS1/40, 20 Hz, Newton+Warp, other GPU workloads remain running",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("components", "training"), required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sequence", choices=("abba", "candidate"), default="abba")
    args = parser.parse_args()
    if args.num_envs < 1 or args.updates <= args.warmup or args.warmup < 0:
        parser.error("Require positive environments and updates > warmup >= 0")
    root = Path(__file__).resolve().parents[1]
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = (components if args.mode == "components" else training)(args, root, args.baseline.resolve())
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "mode": args.mode, "complete": True}), flush=True)


if __name__ == "__main__":
    main()
