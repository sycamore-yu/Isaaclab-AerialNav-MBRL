"""Train each acceptance stage in a fresh process with the same run's checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STAGES = ("pretrain_s0", "pretrain_s1", "pretrain_s2", "finetune_s1", "finetune_s2")
STAGE_CONFIG = {
    "pretrain_s0": {"static_obstacles": 300, "dynamic_obstacles": 0, "randomization": False},
    "pretrain_s1": {"static_obstacles": 300, "dynamic_obstacles": 60, "randomization": False},
    "pretrain_s2": {"static_obstacles": 400, "dynamic_obstacles": 120, "randomization": False},
    "finetune_s1": {"static_obstacles": 300, "dynamic_obstacles": 60, "randomization": True},
    "finetune_s2": {"static_obstacles": 400, "dynamic_obstacles": 100, "randomization": True},
}


def parse_binary_flag(raw: str) -> bool:
    """Use the same explicit 0/1 grammar as train_zhang.py."""
    if raw not in ("0", "1"):
        raise ValueError("expected 0 or 1")
    return raw == "1"


def parse_gradient_diagnostic_updates(raw: str) -> tuple[int, ...]:
    """Validate the same opt-in update-list grammar as train_zhang.py."""
    if raw == "":
        return ()
    parts = raw.split(",")
    if any(not part.isascii() or not part.isdecimal() for part in parts):
        raise ValueError("gradient_diagnostic_updates requires comma-separated positive integers without empty entries")
    updates = tuple(int(part) for part in parts)
    if any(update < 1 for update in updates) or len(set(updates)) != len(updates):
        raise ValueError("gradient_diagnostic_updates must contain unique positive integers")
    return updates


TrainArgValue = str | int | float | bool | tuple[int, ...]
TRAIN_ARG_CHOICES = {
    "timing": ("baseline", "unified_20hz", "unified_50hz"),
    "variant": ("step", "continuous", "dcol"),
    "algorithm": ("bptt", "shac", "ahac"),
    "lr_schedule": ("cosine", "plateau", "constant"),
    "critic_value_transform": ("linear", "softplus"),
    "velocity_loss_mode": ("variant_default", "per_step", "truncated_window", "full_window"),
    "height_loss_mode": ("target_exp", "target_huber", "boundary_huber", "navrl_range"),
    "optimizer": ("adamw", "adam"),
    "gradient_projection": ("none", "height_priority"),
}
TRAIN_ARG_TYPES = {
    "timing": str,
    "variant": str,
    "algorithm": str,
    "lr_schedule": str,
    "observation_normalization": parse_binary_flag,
    "normalization_variance_floor": float,
    "recovery_enabled": parse_binary_flag,
    "timesteps": int,
    "velocity_window_s": float,
    "velocity_loss_mode": str,
    "camera_batch_size": int,
    "lr": float,
    "optimizer": str,
    "weight_decay": float,
    "plateau_patience": int,
    "height_loss_mode": str,
    "height_target_min": float,
    "height_target_max": float,
    "gradient_projection": str,
    "max_grad_norm": float,
    "grad_decay": float,
    "coef_obj_avoidance": float,
    "coef_collide": float,
    "coef_d_acc": float,
    "coef_d_jerk": float,
    "coef_height_boundary": float,
    "boundary_margin": float,
    "boundary_scale": float,
    "cosine_period_updates": int,
    "critic_lr": float,
    "critic_iterations": int,
    "critic_max_iterations": int,
    "critic_value_transform": str,
    "discount": float,
    "td_lambda": float,
    "ahac_min_horizon": int,
    "ahac_max_horizon": int,
    "ahac_initial_horizon": int,
    "ahac_risk_clearance": float,
    "ahac_risk_scale": float,
    "ahac_risk_budget": float,
    "ahac_dual_lr": float,
    "gradient_diagnostic_updates": parse_gradient_diagnostic_updates,
}


def parse_train_args(tokens: list[str]) -> dict[str, TrainArgValue]:
    """Accept only Zhang hyperparameters; process and curriculum flags stay fixed."""
    values = {}
    index = 1 if tokens and tokens[0] == "--" else 0
    while index < len(tokens):
        option, separator, raw_value = tokens[index].partition("=")
        key = option.removeprefix("--")
        if not option.startswith("--") or key not in TRAIN_ARG_TYPES:
            raise ValueError(f"Unsupported --train_args option: {option}")
        if key in values:
            raise ValueError(f"Duplicate --train_args option: {option}")
        if not separator:
            index += 1
            if index == len(tokens):
                raise ValueError(f"Missing value for {option}")
            raw_value = tokens[index]
        try:
            value = TRAIN_ARG_TYPES[key](raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid value for {option}: {raw_value}") from exc
        if key in TRAIN_ARG_CHOICES:
            if value not in TRAIN_ARG_CHOICES[key]:
                raise ValueError(f"Unknown Zhang {key}: {value}")
        elif key == "gradient_diagnostic_updates" or isinstance(value, bool):
            pass  # The converters already validate tuple and explicit 0/1 grammars.
        elif not math.isfinite(value):
            raise ValueError(f"{option} must be finite")
        elif key == "discount":
            if not 0 <= value < 1:
                raise ValueError("discount must be in [0, 1) for continuing cost learning")
        elif key == "td_lambda":
            if not 0 <= value <= 1:
                raise ValueError("td_lambda must be in [0, 1]")
        elif key == "grad_decay":
            if not 0 < value <= 1:
                raise ValueError("grad_decay must be in (0, 1]")
        elif key.startswith("coef_") or key in ("boundary_margin", "ahac_risk_clearance", "weight_decay", "plateau_patience"):
            if value < 0:
                raise ValueError(f"{option} must be nonnegative")
        elif value <= 0:
            raise ValueError(f"{option} must be positive")
        values[key] = value
        index += 1
    if values.get("algorithm", "bptt") == "bptt" and values.get("critic_value_transform", "linear") != "linear":
        raise ValueError("--critic_value_transform softplus requires --algorithm shac or ahac")
    if values.get("algorithm", "bptt") != "bptt":
        if values.get("variant", "dcol") != "dcol":
            raise ValueError("SHAC/AHAC comparisons require --variant dcol")
        if values.get("gradient_diagnostic_updates", ()):
            raise ValueError("Seven-term gradient diagnostics require --algorithm bptt")
        if values.get("gradient_projection", "none") != "none":
            raise ValueError("Height-priority projection requires --algorithm bptt")
    if values.get("timesteps", 150) < 2:
        raise ValueError("timesteps must be at least 2")
    if not 2 <= values.get("ahac_min_horizon", 16) <= values.get("ahac_initial_horizon", 32) <= values.get("ahac_max_horizon", 300):
        raise ValueError("AHAC requires 2 <= min <= initial <= max horizon")
    if not 0.2 < values.get("height_target_min", 0.5) < values.get("height_target_max", 2.5) < 4.0:
        raise ValueError("height target range must be increasing and inside (0.2, 4.0) metres")
    return values


def build_command(
    *,
    method: str,
    stage: str,
    log_dir: Path,
    scripts_dir: Path,
    num_envs: int,
    seed: int,
    device: str,
    finetune_lr: float,
    train_args: dict[str, TrainArgValue],
    parent_checkpoint: Path | None,
    executable: str = sys.executable,
    report_episodes: int | None = None,
) -> list[str]:
    command = [
        executable,
        str(scripts_dir / f"train_{method}.py"),
        "--stage", stage,
        "--log_dir", str(log_dir),
        "--num_envs", str(num_envs),
        "--seed", str(seed),
        "--terrain_seed", str(seed),
        "--device", device,
        "--stop_after_acceptance",
    ]
    if parent_checkpoint is not None:
        command += ["--checkpoint", str(parent_checkpoint)]
    if report_episodes is not None:
        command += ["--report_episodes", str(report_episodes)]
    if method == "zhang":
        for key, value in train_args.items():
            if key != "lr" or not stage.startswith("finetune_"):
                if isinstance(value, bool):
                    formatted = str(int(value))
                else:
                    formatted = ",".join(str(update) for update in value) if isinstance(value, tuple) else str(value)
                command += [f"--{key}", formatted]
    if stage.startswith("finetune_"):
        command += ["--lr", str(finetune_lr)]
    return command


def validate_stage_result(
    result: dict[str, Any],
    *,
    method: str,
    stage: str,
    num_envs: int,
    parent_checkpoint: Path | None,
    variant: str = "dcol",
    algorithm: str = "bptt",
    report_episodes: int | None = None,
    velocity_loss_mode: str | None = None,
    screening: bool = False,
) -> list[str]:
    """Validate one real training report against the fixed acceptance protocol."""
    errors = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    if velocity_loss_mode is None:
        velocity_loss_mode = "per_step" if algorithm == "shac" else "full_window"
    expected = STAGE_CONFIG[stage]
    expect(result.get("method") == method, "method does not match the launched trainer")
    if method == "zhang":
        expect(result.get("variant") == variant, "Zhang variant does not match the launched experiment")
        expect(result.get("algorithm") == algorithm, "Zhang algorithm does not match the launched experiment")
        resolved_mode = velocity_loss_mode
        if velocity_loss_mode == "variant_default":
            resolved_mode = "per_step" if variant == "step" else "truncated_window"
        elif velocity_loss_mode == "full_window":
            resolved_mode = "window"
        expect(result.get("velocity_loss_mode") == velocity_loss_mode,
               "Zhang velocity_loss_mode does not match the launched experiment")
        expect(result.get("resolved_velocity_loss_mode") == resolved_mode,
               "Zhang resolved_velocity_loss_mode does not match the launched experiment")
    expect(result.get("status") == "completed", "training status is not completed")
    expect(result.get("stage") == stage, "training stage does not match")
    report_episodes = num_envs if report_episodes is None else report_episodes
    expect(result.get("num_envs") == num_envs and num_envs >= (1 if screening else 1024),
           "num_envs must match; formal acceptance requires at least 1024")
    expect(report_episodes >= (1 if screening else 1024), "formal reports require at least 1024 episodes")
    expect(result.get("stage_passed") is True, "trainer has not passed the stage")
    expect(result.get("max_updates") is None, "training has a maximum update limit")
    expected_parent = str(parent_checkpoint) if parent_checkpoint is not None else None
    expect((stage == STAGES[0]) is (parent_checkpoint is None),
           "formal training must start fresh only at pretrain_s0 and inherit every later stage")
    expect(result.get("parent_checkpoint") == expected_parent, "parent checkpoint is not the previous stage of this run")
    expect(result.get("fresh") is (parent_checkpoint is None), "fresh-run flag disagrees with the checkpoint lineage")

    curriculum = result.get("curriculum")
    if not isinstance(curriculum, dict):
        return errors + ["curriculum result is missing"]
    expect(curriculum.get("stage") == stage, "curriculum stage does not match")
    expect(curriculum.get("stage_passed") is True, "curriculum has not passed the stage")
    expect(curriculum.get("stage_config") == expected, "curriculum obstacle counts or randomization do not match")
    expect(curriculum.get("active_dynamic_obstacles") == expected["dynamic_obstacles"], "active dynamic obstacle count does not match")
    expect(curriculum.get("promotion_success_rate") == 0.8, "curriculum promotion threshold must be 0.8")
    expect(curriculum.get("episodes_per_report_min") == report_episodes, "episode report threshold does not match")

    report = curriculum.get("latest_training_report")
    if not isinstance(report, dict):
        return errors + ["current-stage training report is missing"]
    expect(report.get("stage") == stage, "training report belongs to another stage")
    for key, value in expected.items():
        expect(report.get(key) == value, f"training report {key} does not match")
    episodes = report.get("episodes")
    expect(isinstance(episodes, int) and episodes >= report_episodes, "training report has too few completed episodes")
    success_rate = report.get("success_rate")
    expect(
        isinstance(success_rate, (int, float)) and math.isfinite(success_rate) and 0.8 <= success_rate <= 1.0,
        "current-stage first-arrival success rate is below 0.8 or invalid",
    )
    if stage == STAGES[-1]:
        expect(result.get("acceptance_passed") is (num_envs >= 1024),
               "final-stage trainer acceptance does not match the validated environment scale")
        expect(curriculum.get("acceptance_passed") is True, "final-stage curriculum acceptance has not passed")
    return errors


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_pipeline(args: argparse.Namespace, train_args: dict[str, TrainArgValue]) -> int:
    screening = getattr(args, "screening", False)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    methods = ("zhang", "navrl") if args.method == "both" else (args.method,)
    summary: dict[str, Any] = {
        "status": "running",
        "acceptance_passed": False,
        "purpose": "training_comparison" if screening else "formal_acceptance",
        "screening": screening,
        "stage_thresholds_passed": False,
        "method": args.method,
        "num_envs": args.num_envs,
        "report_episodes": getattr(args, "report_episodes", None) or args.num_envs,
        "train_seed": args.train_seed,
        "finetune_lr": args.finetune_lr,
        "train_args": train_args,
        "stage_order": list(STAGES),
        "started_at": datetime.now(UTC).isoformat(),
        "methods": {},
        "error": None,
    }
    summary_path = output / "training_result.json"
    write_json(summary_path, summary)
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1")
    scripts_dir = Path(__file__).resolve().parent
    current_method: dict[str, Any] | None = None
    current_stage: dict[str, Any] | None = None
    try:
        for method in methods:
            current_method = {"status": "running", "acceptance_passed": False, "stages": []}
            summary["methods"][method] = current_method
            parent_checkpoint = None
            for stage in STAGES:
                current_stage = None
                log_dir = output / method / stage
                log_dir.mkdir(parents=True)
                stdout_log = log_dir.parent / f"{stage}.stdout.log"
                command = build_command(
                    method=method, stage=stage, log_dir=log_dir, scripts_dir=scripts_dir,
                    num_envs=args.num_envs, seed=args.train_seed, device=args.device,
                    finetune_lr=args.finetune_lr, train_args=train_args, parent_checkpoint=parent_checkpoint,
                    report_episodes=getattr(args, "report_episodes", None),
                )
                current_stage = {
                    "stage": stage,
                    "stage_config": STAGE_CONFIG[stage],
                    "status": "running",
                    "log_dir": str(log_dir),
                    "stdout_log": str(stdout_log),
                    "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint is not None else None,
                    "command": command,
                    "exit_code": None,
                    "validation_errors": [],
                    "result": None,
                }
                current_method["stages"].append(current_stage)
                write_json(summary_path, summary)
                print(f"Starting {method}/{stage}: {log_dir}", flush=True)
                with stdout_log.open("w", encoding="utf-8") as stream:
                    process = subprocess.run(
                        command, cwd=scripts_dir.parent, env=environment,
                        stdout=stream, stderr=subprocess.STDOUT, check=False,
                    )
                current_stage["exit_code"] = process.returncode
                result_path = log_dir / "training_result.json"
                if result_path.is_file():
                    try:
                        current_stage["result"] = json.loads(result_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid training result {result_path}: {exc}") from exc
                result = current_stage["result"]
                if process.returncode == 0 and isinstance(result, dict) and result.get("status") == "stopped":
                    # A requested stop saves a complete-update checkpoint but
                    # does not certify or advance the current curriculum stage.
                    checkpoint = log_dir / "checkpoint_stopped.pt"
                    reported_checkpoint = result.get("checkpoint")
                    if result.get("error") is not None:
                        raise RuntimeError(f"Stopped trainer reported an error: {result['error']}")
                    if (not isinstance(reported_checkpoint, str)
                            or Path(reported_checkpoint).resolve() != checkpoint.resolve()
                            or not checkpoint.is_file()):
                        raise RuntimeError(f"Stopped trainer checkpoint is missing or mismatched: {checkpoint}")
                    current_stage.update(status="stopped", stage_passed=False, acceptance_passed=False,
                                         checkpoint=str(checkpoint), error=None)
                    current_method.update(status="stopped", acceptance_passed=False,
                                          checkpoint=str(checkpoint), error=None)
                    summary.update(status="stopped", acceptance_passed=False, checkpoint=str(checkpoint),
                                   error=None, finished_at=datetime.now(UTC).isoformat())
                    write_json(summary_path, summary)
                    print(f"Stopped {method}/{stage}: {checkpoint}", flush=True)
                    return 0
                errors = []
                if process.returncode != 0:
                    errors.append(f"trainer exited with code {process.returncode}; see {stdout_log}")
                if not isinstance(current_stage["result"], dict):
                    errors.append("trainer did not write a training_result.json object")
                else:
                    errors += validate_stage_result(
                        current_stage["result"], method=method, stage=stage,
                        num_envs=args.num_envs, parent_checkpoint=parent_checkpoint,
                        variant=train_args.get("variant", "dcol"),
                        algorithm=train_args.get("algorithm", "bptt"),
                        report_episodes=getattr(args, "report_episodes", None),
                        velocity_loss_mode=train_args.get("velocity_loss_mode"),
                        screening=screening,
                    )
                checkpoint = log_dir / "checkpoint_stage.pt"
                if not checkpoint.is_file():
                    errors.append(f"stage checkpoint is missing: {checkpoint}")
                current_stage["validation_errors"] = errors
                if errors:
                    raise RuntimeError(f"{method}/{stage}: " + "; ".join(errors))
                current_stage["status"] = "completed"
                current_stage["checkpoint"] = str(checkpoint)
                parent_checkpoint = checkpoint
                write_json(summary_path, summary)
                print(f"Passed {method}/{stage}", flush=True)
            current_method["status"] = "completed"
            current_method["stage_thresholds_passed"] = True
            current_method["acceptance_passed"] = not screening
        summary["status"] = "completed"
        summary["stage_thresholds_passed"] = True
        summary["acceptance_passed"] = not screening
        exit_code = 0
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc) or "Training interrupted"
        if current_method is not None:
            current_method["status"] = "failed"
        if current_stage is not None:
            current_stage["status"] = "failed"
        print(summary["error"], file=sys.stderr, flush=True)
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    summary["finished_at"] = datetime.now(UTC).isoformat()
    write_json(summary_path, summary)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--method", choices=("navrl", "zhang", "both"), default="zhang")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--report_episodes", type=int, default=None)
    parser.add_argument("--screening", action="store_true",
                        help="Allow reduced-env fresh comparisons; never label them as formal acceptance")
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--finetune_lr", type=float, default=5e-5)
    parser.add_argument("--train_args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num_envs must be positive")
    if args.num_envs < 1024 and not args.screening:
        parser.error("formal acceptance requires --num_envs >= 1024")
    if args.report_episodes is not None and args.report_episodes < (1 if args.screening else 1024):
        parser.error("report_episodes must be positive; formal reports require at least 1024")
    if not math.isfinite(args.finetune_lr) or args.finetune_lr <= 0:
        parser.error("--finetune_lr must be finite and positive")
    try:
        train_args = parse_train_args(args.train_args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.method == "navrl" and train_args:
        parser.error("--train_args contains Zhang hyperparameters; use --method zhang or both")
    if Path(args.output).expanduser().exists():
        parser.error("--output must be a new directory; existing runs cannot be resumed")
    return run_pipeline(args, train_args)


if __name__ == "__main__":
    raise SystemExit(main())
