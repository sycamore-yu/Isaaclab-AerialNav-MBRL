"""Compare three backend/frontend combinations on the fixed Zhang baseline.

Each subprocess executes the real train_zhang.py entry point. Only this benchmark
wraps the trainer's clock with CUDA synchronization; production training is unchanged.
Other GPU jobs are observed, never stopped, and defaults are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

COMBINATIONS = (
    ("physx_torch", "isaacsim_physx", "torch"),
    ("newton_torch", "newton_mjwarp", "torch"),
    ("newton_warp", "newton_mjwarp", "warp"),
)
NUM_ENVS, HORIZON = 256, 150
TRAIN_PROTOCOL = {
    "variant": "dcol",
    "algorithm": "bptt",
    "recurrent_model": "gru",
    "timesteps": HORIZON,
    "seed": 0,
    "velocity_loss_mode": "full_window",
    "velocity_window_s": 2.0,
    "height_loss_mode": "navrl_range",
    "lr": 0.001,
    "lr_schedule": "cosine",
    "cosine_period_updates": 400,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "recovery_enabled": True,
    "gradient_diagnostic_updates": [1, 50, 250, 1000],
    "gradient_projection": "none",
    "observation_normalization": False,
    "normalization_variance_floor": 1e-4,
    "collision_metric": "dcol",
    "approach_mode": "finite_difference",
    "lateral_avoidance": 0.0,
    "risk_speed_floor": 1.0,
    "max_grad_norm": 5.0,
    "grad_decay": 0.4,
    "coef_v": 1.0,
    "coef_v_pred": 2.0,
    "coef_obj_avoidance": 3.0,
    "coef_collide": 7.5,
    "coef_d_acc": 0.01,
    "coef_d_jerk": 0.001,
    "coef_height_boundary": 1.0,
    "boundary_margin": 0.5,
    "boundary_scale": 0.5,
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def capture(command, *, cwd=None):
    """Keep read-only system observations, including failures, in the report."""
    try:
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=15, check=False)
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def gpu_snapshot():
    return {
        "at": datetime.now(UTC).isoformat(),
        "summary": capture(["nvidia-smi"]),
        "gpus": capture(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,temperature.gpu,clocks.current.sm",
                "--format=csv",
            ]
        ),
        "processes": capture(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory",
                "--format=csv",
            ]
        ),
    }


def source_hashes(root):
    paths = [path for folder in ("src", "scripts") for path in (root / folder).rglob("*.py")]
    paths += [root / "pyproject.toml", root / "uv.lock", root / "configs/experiments/zhang_finetune_s1.json"]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
        if path.is_file()
    }


def versions(root):
    packages = {}
    for name in (
        "torch",
        "warp-lang",
        "newton",
        "mujoco",
        "mujoco-warp",
        "isaaclab",
        "isaaclab-rl",
        "isaaclab-experimental",
        "isaacsim",
        "rsl-rl-lib",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "git_head": capture(["git", "rev-parse", "HEAD"], cwd=root),
        "git_status": capture(["git", "status", "--short"], cwd=root),
        "source_sha256": source_hashes(root),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def option_value(value):
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, list):
        return ",".join(map(str, value))
    return str(value)


def training_arguments(log_dir, physics, frontend, updates, device):
    arguments = [
        "--log_dir",
        str(log_dir),
        "--num_envs",
        str(NUM_ENVS),
        "--max_iterations",
        str(updates),
        "--stage",
        "finetune_s1",
        "--fresh_stage",
        "--terrain_seed",
        "0",
        "--report_episodes",
        "1024",
        "--timing",
        "unified_20hz",
        "--camera_batch_size",
        "64",
        "--num_dynamic_obstacles",
        "60",
        "--height_target_min",
        "0.5",
        "--height_target_max",
        "2.5",
        "--target_speed_min",
        "2.25",
        "--target_speed_max",
        "9.75",
        "--device",
        device,
        f"physics={physics}",
        "--frontend",
        frontend,
    ]
    for name, value in TRAIN_PROTOCOL.items():
        arguments += ["--" + name, option_value(value)]
    return arguments


def train_child(arguments):
    """Execute the existing trainer, injecting synchronization only into its clock."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--log_dir", type=Path, required=True)
    parsed, _ = parser.parse_known_args(arguments)
    root = Path(__file__).resolve().parents[1]
    entrypoint = root / "scripts/train_zhang.py"
    spec = importlib.util.spec_from_file_location("_benchmark_train_zhang", entrypoint)
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)

    process_started = float(os.environ["NAVRL_STAGE_STARTED_MONOTONIC"])
    original_factory = training.create_navigation_env

    def create_timed_env(*factory_args, **factory_kwargs):
        # Preserve train_zhang's normal import/bootstrap order. In particular,
        # SummaryWriter must not be imported before tasks and the environment cfg.
        env = original_factory(*factory_args, **factory_kwargs)
        try:
            import torch

            import isaaclab_navrl.zhang.runner as runner

            if Path(runner.__file__).resolve() != root / "src/isaaclab_navrl/zhang/runner.py":
                raise RuntimeError("Benchmark imported the trainer from a different source tree")
            original_run = runner.ZhangTrainer.run

            def timed_run(self):
                device = torch.device(self.env.device)
                if device.type != "cuda":
                    raise ValueError("The backend comparison requires a CUDA device")
                timing = {
                    "synchronization": "torch.cuda.synchronize(device) at runner perf_counter boundaries only",
                    "recurrent_state_size": self.model.recurrent_state_size,
                    "model_class": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
                    "model_parameters": sum(parameter.numel() for parameter in self.model.parameters()),
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "gpu": torch.cuda.get_device_name(device),
                    "device": str(device),
                    "synchronize_calls": 0,
                }
                original_clock = runner.time

                class SynchronizedClock:
                    def perf_counter(self):
                        torch.cuda.synchronize(device)
                        timing["synchronize_calls"] += 1
                        value = time.perf_counter()
                        # The trainer's first counter is immediately before its update loop,
                        # after scene/model initialization, reset and logging setup.
                        if "setup_s" not in timing:
                            timing["setup_s"] = time.monotonic() - process_started
                        return value

                    def __getattr__(self, name):
                        return getattr(original_clock, name)

                runner.time = SynchronizedClock()
                try:
                    if self.model.recurrent_state_size != 192:
                        raise ValueError("The baseline requires the existing 192-unit GRU")
                    return original_run(self)
                finally:
                    runner.time = original_clock
                    try:
                        torch.cuda.synchronize(device)
                    except RuntimeError as exc:
                        timing["final_sync_error"] = str(exc)
                    timing["training_return_process_s"] = time.monotonic() - process_started
                    write_json(parsed.log_dir.parent / "timing.json", timing)

            runner.ZhangTrainer.run = timed_run
        except BaseException:
            env.close()
            raise
        return env

    training.create_navigation_env = create_timed_env
    sys.argv = [str(entrypoint), *arguments]
    training.main()


def stats(values):
    return {"count": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def finite_json(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    return True


def summarize_run(run, updates, warmup):
    directory = Path(run["directory"])
    log_dir = directory / "training"
    errors = []

    def read(name, *, lines=False, parent=False):
        path = (directory if parent else log_dir) / name
        try:
            content = path.read_text()
            value = (
                [json.loads(line) for line in content.splitlines() if line.strip()] if lines else json.loads(content)
            )
            if not finite_json(value):
                errors.append(f"Nonfinite value in {path}")
            return value
        except (OSError, ValueError) as exc:
            errors.append(f"Cannot read {path}: {exc}")
            return [] if lines else {}

    config = read("config.json")
    result = read("training_result.json")
    records = read("updates.jsonl", lines=True)
    timing = read("timing.json", parent=True)

    def expect(actual, expected, name):
        if actual != expected:
            errors.append(f"{name}: expected {expected!r}, received {actual!r}")

    expect(run["returncode"], 0, "subprocess returncode")
    expected_train = {
        **TRAIN_PROTOCOL,
        "physics": run["physics"],
        "frontend": run["frontend"],
        "max_iterations": updates,
        "stop_after_acceptance": False,
        "physics_dt": 0.05,
        "policy_dt": 0.05,
        "height_target_range": [0.5, 2.5],
    }
    for key, expected in expected_train.items():
        expect(config.get("train", {}).get(key), expected, f"train.{key}")
    expected_environment = {
        "num_envs": NUM_ENVS,
        "seed": 0,
        "terrain_seed": 0,
        "stage": "finetune_s1",
        "timing": "unified_20hz",
        "static_obstacles": 300,
        "dynamic_obstacles": 60,
        "randomization": True,
        "report_episodes": 1024,
    }
    for key, expected in expected_environment.items():
        expect(config.get("environment", {}).get(key), expected, f"environment.{key}")
    expect(config.get("environment", {}).get("sim_to_real", {}).get("enabled"), True, "randomization enabled")
    expect(config.get("fresh"), True, "fresh initialization")
    expect(config.get("direct_stage_start"), True, "direct-stage start")
    expect(config.get("parent_checkpoint"), None, "parent checkpoint")
    expect(result.get("status"), "completed", "training status")
    expect(result.get("completed_updates"), updates, "completed updates")
    expect(result.get("stage_updates"), updates, "completed stage updates")
    expect(result.get("environment_steps"), updates * NUM_ENVS * HORIZON, "environment steps")
    expect(result.get("stage"), "finetune_s1", "final stage")
    expect(result.get("nonfinite_events"), 0, "nonfinite environment events")
    expect(result.get("nonfinite_grad_events"), 0, "nonfinite gradient events")
    expect(timing.get("recurrent_state_size"), 192, "GRU width")
    if timing.get("final_sync_error") or not timing.get("synchronize_calls", 0):
        errors.append("CUDA synchronization did not complete successfully")
    if not isinstance(timing.get("setup_s"), (int, float)) or timing["setup_s"] <= 0:
        errors.append("The synchronized start of the training loop was not recorded")
    for payload, name in ((config, "config"), (result, "result")):
        expect(payload.get("physics"), run["physics"], f"{name}.physics")
        expect(payload.get("frontend"), run["frontend"], f"{name}.frontend")
        if not payload.get("environment_class"):
            errors.append(f"{name} lacks the actual environment class")
    expect(config.get("environment_class"), result.get("environment_class"), "actual environment class")
    checkpoint = result.get("checkpoint")
    if not checkpoint or not Path(checkpoint).is_file():
        errors.append("Completed checkpoint is missing")
    expect(len(records), updates, "update log length")
    for index, record in enumerate(records, 1):
        expect(record.get("update"), index, "update sequence")
        expect(record.get("rollout_steps"), HORIZON, f"update {index} horizon")
        expect(record.get("environment_steps"), index * NUM_ENVS * HORIZON, f"update {index} environment steps")
        if not isinstance(record.get("update_s"), (int, float)) or record["update_s"] <= 0:
            errors.append(f"Update {index} has an invalid duration")
        if not isinstance(record.get("process_elapsed_s"), (int, float)) or record["process_elapsed_s"] <= 0:
            errors.append(f"Update {index} has an invalid process timestamp")
    summary = {
        **run,
        "valid": not errors,
        "errors": errors,
        "timing": timing,
        "config_path": str(log_dir / "config.json"),
        "updates_path": str(log_dir / "updates.jsonl"),
        "result_path": str(log_dir / "training_result.json"),
        "actual_environment_class": result.get("environment_class"),
        "raw_result": result,
        "warmup_updates": warmup,
    }
    if not errors:
        measured = records[warmup:]
        durations = [record["update_s"] for record in measured]
        intervals = [
            records[index]["process_elapsed_s"] - records[index - 1]["process_elapsed_s"]
            for index in range(max(warmup, 1), len(records))
        ]
        summary.update(
            measured_updates=[record["update"] for record in measured],
            measured_update_s=durations,
            stable_update_s=stats(durations),
            stable_environment_steps_per_s=NUM_ENVS * HORIZON / statistics.median(durations),
            first_update_process_s=records[0]["process_elapsed_s"],
            stable_end_to_end_update_s=stats(intervals) if intervals else None,
        )
    write_json(directory / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/benchmarks") / datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S_backends"),
    )
    parser.add_argument("--updates", type=int, default=5, help="Total updates per fresh training subprocess")
    parser.add_argument("--warmup", type=int, default=2, help="Leading updates excluded from stable-update statistics")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.updates <= args.warmup or args.warmup < 0 or args.rounds < 1:
        parser.error("Require updates > warmup >= 0 and rounds >= 1")
    if not args.device.startswith("cuda"):
        parser.error("This comparison requires --device cuda[:index]")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    initial_versions = versions(root)
    plan = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_root": str(root),
        "device": args.device,
        "updates": args.updates,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "train_protocol": TRAIN_PROTOCOL,
        "versions": initial_versions,
        "measurement": "Full Zhang update: rollout, backward, optimizer, diagnostics; CUDA-synchronized runner clock",
        "end_to_end_intervals": "Consecutive update-log timestamps also include intervening reporting/checkpoint work",
        "other_gpu_jobs": "Unchanged; inspect per-run nvidia_before.json and nvidia_after.json for contention",
        "cache_policy": "Normal shared compilation caches; warmup and rotated ordering, no cache deletion",
        "runs": [],
    }
    write_json(output / "benchmark.json", plan)
    write_json(
        output / "source_diff.json",
        capture(["git", "diff", "--", "src", "scripts", "configs", "pyproject.toml", "uv.lock"], cwd=root),
    )
    for round_index in range(args.rounds):
        order = COMBINATIONS[round_index % 3 :] + COMBINATIONS[: round_index % 3]
        for position, (label, physics, frontend) in enumerate(order, 1):
            directory = output / f"round_{round_index + 1:02d}" / f"{position}_{label}"
            directory.mkdir(parents=True)
            training_args = training_arguments(directory / "training", physics, frontend, args.updates, args.device)
            command = [sys.executable, str(Path(__file__).resolve()), "--_train_child", *training_args]
            run = {
                "round": round_index + 1,
                "position": position,
                "combination": label,
                "physics": physics,
                "frontend": frontend,
                "directory": str(directory),
                "command": command,
                "training_entrypoint": str(root / "scripts/train_zhang.py"),
                "training_arguments": training_args,
            }
            write_json(directory / "command.json", run)
            write_json(directory / "nvidia_before.json", gpu_snapshot())
            print(f"Starting round {round_index + 1}/{args.rounds}: {label}; logs: {directory}", flush=True)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
            environment["PYTHONUNBUFFERED"] = "1"
            started = time.monotonic()
            environment["NAVRL_STAGE_STARTED_MONOTONIC"] = str(started)
            with (directory / "console.log").open("w") as log:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            run.update(returncode=completed.returncode, process_wall_s=time.monotonic() - started)
            write_json(directory / "nvidia_after.json", gpu_snapshot())
            summary = summarize_run(run, args.updates, args.warmup)
            plan["runs"].append(summary)
            write_json(output / "benchmark.json", plan)
            print(
                json.dumps(
                    {
                        "combination": label,
                        "round": round_index + 1,
                        "valid": summary["valid"],
                        "stable_update_s": summary.get("stable_update_s"),
                        "errors": summary["errors"],
                    }
                ),
                flush=True,
            )
    plan["source_unchanged"] = initial_versions["source_sha256"] == source_hashes(root)
    plan["aggregate"] = {}
    for label, _, _ in COMBINATIONS:
        successful = [run for run in plan["runs"] if run["combination"] == label and run["valid"]]
        if successful:
            durations = [duration for run in successful for duration in run["measured_update_s"]]
            plan["aggregate"][label] = {
                "successful_rounds": len(successful),
                "expected_rounds": args.rounds,
                "stable_update_s": stats(durations),
                "round_median_update_s": stats([run["stable_update_s"]["median"] for run in successful]),
                "setup_s": stats([run["timing"]["setup_s"] for run in successful]),
                "process_wall_s": stats([run["process_wall_s"] for run in successful]),
                "stable_environment_steps_per_s": NUM_ENVS * HORIZON / statistics.median(durations),
            }
    plan["valid"] = plan["source_unchanged"] and all(run["valid"] for run in plan["runs"])
    if plan["valid"]:
        plan["observed_order_fastest_first"] = sorted(
            plan["aggregate"], key=lambda label: plan["aggregate"][label]["round_median_update_s"]["median"]
        )
    plan["finished_at"] = datetime.now(UTC).isoformat()
    write_json(output / "benchmark.json", plan)
    print(
        json.dumps(
            {
                "report": str(output / "benchmark.json"),
                "valid": plan["valid"],
                "aggregate": plan["aggregate"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if plan["valid"] else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_train_child":
        train_child(sys.argv[2:])
    else:
        raise SystemExit(main())
