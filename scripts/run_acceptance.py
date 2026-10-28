"""Run a fresh training job followed by independent held-out evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def validate_train_args(method, arguments):
    """Permit only full method-hyperparameter names in protocol overrides."""
    allowed = {
        "--timesteps", "--velocity_window_s", "--lr", "--max_grad_norm", "--grad_decay",
        "--coef_obj_avoidance", "--coef_collide", "--coef_d_acc", "--coef_d_jerk",
    } if method == "zhang" else set()
    rejected = [token for token in arguments if token.startswith("--") and token.split("=", 1)[0] not in allowed]
    if rejected:
        raise ValueError(f"--train_args only accepts supported method hyperparameters; rejected: {rejected}")


def run(command, log_path, environment):
    with log_path.open("x") as stream:
        print("RUN " + " ".join(str(value) for value in command), flush=True)
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, check=False)
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; see {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--method", choices=("navrl", "zhang"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=2200)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=10000)
    parser.add_argument("--dynamic_obstacles", type=int, nargs="+", default=[0, 60, 80, 100, 120])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train_args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.train_seed == args.eval_seed:
        parser.error("training and evaluation seeds must differ")
    if args.output.exists():
        parser.error(f"fresh runs require a new output directory: {args.output}")
    try:
        validate_train_args(args.method, args.train_args)
    except ValueError as error:
        parser.error(str(error))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    start = time.time()
    protocol = {
        "method": args.method,
        "status": "running",
        "started_unix_s": start,
        "fresh": True,
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "evaluation_episodes_per_difficulty": args.episodes,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "static_obstacles": 350,
        "training_dynamic_obstacles": 100,
        "evaluation_dynamic_obstacles": args.dynamic_obstacles,
        "train_args": args.train_args,
        "meets_requested_budget": (
            args.num_envs == 1024 and args.max_iterations == 2200 and args.episodes >= 100
            and {0, 60, 80, 100, 120}.issubset(args.dynamic_obstacles)
        ),
        "acceptance_passed": False,
    }
    status_path = args.output / "acceptance.json"
    status_path.write_text(json.dumps(protocol, indent=2) + "\n")
    try:
        train_dir = args.output / "train"
        run(
            [sys.executable, str(root / "scripts" / f"train_{args.method}.py"),
             "--log_dir", str(train_dir), "--num_envs", str(args.num_envs),
             "--max_iterations", str(args.max_iterations), "--seed", str(args.train_seed),
             "--terrain_seed", str(args.train_seed), "--static_obstacles", "350", "--dynamic_obstacles", "100",
             "--device", args.device, *args.train_args],
            args.output / "train.log", environment,
        )
        training = json.loads((train_dir / "training_result.json").read_text())
        protocol["training_result"] = training
        if not (training["status"] == "completed" and training["fresh"]
                and training["num_envs"] == args.num_envs and training["completed_updates"] == args.max_iterations):
            raise RuntimeError("The training manifest does not satisfy this run's fresh-training budget.")
        checkpoint = Path(training["checkpoint"])
        status_path.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
        summaries = []
        for dynamic in args.dynamic_obstacles:
            output = args.output / f"evaluation_350_{dynamic}.json"
            run(
                [sys.executable, str(root / "scripts/evaluate.py"), "--method", args.method,
                 "--checkpoint", str(checkpoint), "--output", str(output), "--episodes", str(args.episodes),
                 "--seed", str(args.eval_seed), "--terrain_seed", str(args.eval_seed),
                 "--static_obstacles", "350", "--dynamic_obstacles", str(dynamic), "--device", args.device],
                args.output / f"evaluation_350_{dynamic}.log", environment,
            )
            result = json.loads(output.read_text())
            summaries.append({"static_obstacles": 350, "dynamic_obstacles": dynamic, **result["summary"]})
            protocol["evaluations"] = summaries
            status_path.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
        with (args.output / "summary.csv").open("w", newline="") as stream:
            rows = [{key: value for key, value in row.items() if key != "counts"} for row in summaries]
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        main_result = next((row for row in summaries if row["dynamic_obstacles"] == 100), None)
        target_reached = main_result is not None and main_result["success_rate"] >= 0.8
        protocol.update(
            status="completed", checkpoint=str(checkpoint), elapsed_s=time.time() - start,
            main_success_target=0.8,
            main_success_target_reached=target_reached,
            acceptance_passed=(
                protocol["meets_requested_budget"] and target_reached
                and all(row["episodes"] >= 100 and row["invalid_rate"] == 0 for row in summaries)
                and training.get("nonfinite_events") == 0
                and training.get("nonfinite_grad_events", 0) == 0
            ),
        )
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        protocol.update(status="failed", error=str(error), elapsed_s=time.time() - start)
        raise
    finally:
        status_path.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    print(json.dumps(protocol, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
