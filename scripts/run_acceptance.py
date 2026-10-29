"""Start fresh curriculum training; acceptance comes from training episodes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def validate_train_args(method, arguments):
    allowed = {
        "--timesteps", "--velocity_window_s", "--lr", "--max_grad_norm", "--grad_decay",
        "--coef_obj_avoidance", "--coef_collide", "--coef_d_acc", "--coef_d_jerk", "--cosine_period_updates",
    } if method == "zhang" else set()
    rejected = [token for token in arguments if token.startswith("--") and token.split("=", 1)[0] not in allowed]
    if rejected:
        raise ValueError(f"Unsupported method hyperparameters: {rejected}")


def training_command(method, output, seed, device, train_args):
    validate_train_args(method, train_args)
    script = Path(__file__).resolve().parent / f"train_{method}.py"
    return [sys.executable, str(script), "--log_dir", str(output), "--num_envs", "1024",
            "--seed", str(seed), "--terrain_seed", str(seed), "--static_obstacles", "350",
            "--device", device, "--stop_after_acceptance", *train_args]


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--method", choices=("navrl", "zhang", "both"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train_args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Fresh training requires a new output directory: {args.output}")
    methods = ("navrl", "zhang") if args.method == "both" else (args.method,)
    commands = []
    for method in methods:
        output = args.output / method if args.method == "both" else args.output
        train_args = args.train_args if method == "zhang" or args.method != "both" else []
        try:
            commands.append((training_command(method, output.resolve(), args.train_seed, args.device, train_args), output))
        except ValueError as error:
            parser.error(str(error))
    environment = os.environ.copy()
    environment.update(OMNI_KIT_ACCEPT_EULA="YES", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    for command, output in commands:
        print("RUN " + " ".join(command), flush=True)
        result = subprocess.run(command, env=environment, check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
        training_result = json.loads((output / "training_result.json").read_text())
        if training_result["status"] != "completed" or not training_result["acceptance_passed"]:
            raise RuntimeError(f"Training ended before acceptance: {output}")


if __name__ == "__main__":
    main()
