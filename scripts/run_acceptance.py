"""Train each acceptance stage in a fresh process with the same run's checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from isaaclab_navrl.physics import DEFAULT_FRONTEND, DEFAULT_PHYSICS, parse_physics_args, validate_runtime
from isaaclab_navrl.source import freeze_source, verify_source
from isaaclab_navrl.training import STAGE_CONFIG, STAGES, write_json
from isaaclab_navrl.zhang.cli import TrainArgValue, parse_train_args


def selected_stages(start_stage="pretrain_s0", *, screening=False, method="zhang", fixed_stage=False):
    """Keep skipped stages explicit and prevent direct starts from certifying a full curriculum."""
    if start_stage not in STAGES:
        raise ValueError(f"Unknown start stage: {start_stage}")
    if start_stage != STAGES[0] and (not screening or method != "zhang"):
        raise ValueError("Direct stage starts require --screening --method zhang")
    if fixed_stage:
        if not screening or method != "zhang":
            raise ValueError("Fixed-stage comparisons require --screening --method zhang")
        return (start_stage,)
    return STAGES[STAGES.index(start_stage) :]


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
    fresh_stage: bool = False,
    fixed_stage: bool = False,
    physics: str = DEFAULT_PHYSICS,
    frontend: str = DEFAULT_FRONTEND,
) -> list[str]:
    validate_runtime(physics, frontend)
    command = [
        executable,
        str(scripts_dir / f"train_{method}.py"),
        "--stage",
        stage,
        "--log_dir",
        str(log_dir),
        "--num_envs",
        str(num_envs),
        "--seed",
        str(seed),
        "--terrain_seed",
        str(seed),
        "--device",
        device,
        f"physics={physics}",
        "--frontend",
        frontend,
    ]
    if not fixed_stage:
        command += ["--stop_after_acceptance"]
    if parent_checkpoint is not None:
        command += ["--checkpoint", str(parent_checkpoint)]
    elif fresh_stage:
        command += ["--fresh_stage"]
    if report_episodes is not None:
        command += ["--report_episodes", str(report_episodes)]
    if method == "zhang":
        for key, value in train_args.items():
            if key != "lr" or not stage.startswith("finetune_") or fresh_stage:
                if isinstance(value, bool):
                    formatted = str(int(value))
                else:
                    formatted = ",".join(str(update) for update in value) if isinstance(value, tuple) else str(value)
                command += [f"--{key}", formatted]
    if stage.startswith("finetune_") and not fresh_stage:
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
    recurrent_model: str = "gru",
    report_episodes: int | None = None,
    velocity_loss_mode: str | None = None,
    screening: bool = False,
    fresh_start_stage: str = "pretrain_s0",
    num_dynamic_obstacles: int | None = None,
) -> list[str]:
    """Validate one real training report against the fixed acceptance protocol."""
    errors = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    if velocity_loss_mode is None:
        velocity_loss_mode = "per_step" if algorithm in ("shac", "ahac") else "full_window"
    expected = STAGE_CONFIG[stage].copy()
    if num_dynamic_obstacles is not None:
        expected["dynamic_obstacles"] = num_dynamic_obstacles
    expect(result.get("method") == method, "method does not match the launched trainer")
    if method == "zhang":
        expect(result.get("variant") == variant, "Zhang variant does not match the launched experiment")
        expect(result.get("algorithm") == algorithm, "Zhang algorithm does not match the launched experiment")
        expect(
            result.get("recurrent_model", "gru") == recurrent_model,
            "Zhang recurrent_model does not match the launched experiment",
        )
        resolved_mode = velocity_loss_mode
        if velocity_loss_mode == "variant_default":
            resolved_mode = "per_step" if variant == "step" else "truncated_window"
        elif velocity_loss_mode == "full_window":
            resolved_mode = "window"
        expect(
            result.get("velocity_loss_mode") == velocity_loss_mode,
            "Zhang velocity_loss_mode does not match the launched experiment",
        )
        expect(
            result.get("resolved_velocity_loss_mode") == resolved_mode,
            "Zhang resolved_velocity_loss_mode does not match the launched experiment",
        )
    expect(result.get("status") == "completed", "training status is not completed")
    expect(result.get("stage") == stage, "training stage does not match")
    report_episodes = num_envs if report_episodes is None else report_episodes
    expect(
        result.get("num_envs") == num_envs and num_envs >= (1 if screening else 1024),
        "num_envs must match; formal acceptance requires at least 1024",
    )
    expect(report_episodes >= (1 if screening else 1024), "formal reports require at least 1024 episodes")
    expect(result.get("stage_passed") is True, "trainer has not passed the stage")
    expect(result.get("max_updates") is None, "training has a maximum update limit")
    expected_parent = str(parent_checkpoint) if parent_checkpoint is not None else None
    expect(fresh_start_stage in STAGES, "unknown fresh start stage")
    expect(
        fresh_start_stage == STAGES[0] or (screening and method == "zhang"),
        "direct stage starts are Zhang screening comparisons only",
    )
    expect(
        (stage == fresh_start_stage) is (parent_checkpoint is None),
        "training must start fresh only at its declared start stage and inherit every later stage",
    )
    expect(
        result.get("parent_checkpoint") == expected_parent, "parent checkpoint is not the previous stage of this run"
    )
    expect(result.get("fresh") is (parent_checkpoint is None), "fresh-run flag disagrees with the checkpoint lineage")

    curriculum = result.get("curriculum")
    if not isinstance(curriculum, dict):
        return errors + ["curriculum result is missing"]
    expect(curriculum.get("stage") == stage, "curriculum stage does not match")
    expect(curriculum.get("stage_passed") is True, "curriculum has not passed the stage")
    expect(curriculum.get("stage_config") == expected, "curriculum obstacle counts or randomization do not match")
    expect(
        curriculum.get("active_dynamic_obstacles") == expected["dynamic_obstacles"],
        "active dynamic obstacle count does not match",
    )
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
        expect(
            result.get("acceptance_passed") is (num_envs >= 1024),
            "final-stage trainer acceptance does not match the validated environment scale",
        )
        expect(curriculum.get("acceptance_passed") is True, "final-stage curriculum acceptance has not passed")
    return errors


def run_pipeline(args: argparse.Namespace, train_args: dict[str, TrainArgValue]) -> int:
    screening = getattr(args, "screening", False)
    start_stage = getattr(args, "start_stage", STAGES[0])
    fixed_stage = getattr(args, "fixed_stage", False)
    stages = selected_stages(start_stage, screening=screening, method=args.method, fixed_stage=fixed_stage)
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
        "physics": args.physics_preset,
        "frontend": args.frontend,
        "num_envs": args.num_envs,
        "report_episodes": getattr(args, "report_episodes", None) or args.num_envs,
        "train_seed": args.train_seed,
        "finetune_lr": args.finetune_lr,
        "train_args": train_args,
        "start_stage": start_stage,
        "fixed_stage": fixed_stage,
        "stage_order": list(stages),
        "skipped_stages": list(STAGES[: STAGES.index(start_stage)]),
        "complete_curriculum_passed": False,
        "started_at": datetime.now(UTC).isoformat(),
        "methods": {},
        "error": None,
    }
    summary_path = output / "training_result.json"
    write_json(summary_path, summary)
    environment = os.environ.copy()
    environment.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMNI_KIT_ACCEPT_EULA="YES",
        PYTHONUNBUFFERED="1",
    )
    current_method: dict[str, Any] | None = None
    current_stage: dict[str, Any] | None = None
    try:
        source_identity = freeze_source(Path(__file__).resolve().parents[1], output / "source")
        expected_commit = environment.get("NAVRL_EXPECTED_COMMIT")
        if expected_commit is not None and source_identity["commit"] != expected_commit:
            raise RuntimeError("Pipeline snapshot differs from the dispatcher's pinned commit")
        summary["source_identity"] = source_identity
        scripts_dir = Path(source_identity["root"]) / "scripts"
        environment.update(
            PYTHONPATH=str(scripts_dir.parent / "src"),
            PYTHONDONTWRITEBYTECODE="1",
            NAVRL_SOURCE_ROOT=source_identity["root"],
            NAVRL_SOURCE_COMMIT=source_identity["commit"],
        )
        write_json(summary_path, summary)
        for method in methods:
            environment["NAVRL_TENSORBOARD_DIR"] = str(output / method / "tensorboard")
            current_method = {"status": "running", "acceptance_passed": False, "stages": []}
            summary["methods"][method] = current_method
            parent_checkpoint = None
            for stage in stages:
                verify_source(scripts_dir.parent, source_identity["commit"])
                current_stage = None
                log_dir = output / method / stage
                log_dir.mkdir(parents=True)
                stdout_log = log_dir.parent / f"{stage}.stdout.log"
                command = build_command(
                    method=method,
                    stage=stage,
                    log_dir=log_dir,
                    scripts_dir=scripts_dir,
                    num_envs=args.num_envs,
                    seed=args.train_seed,
                    device=args.device,
                    finetune_lr=args.finetune_lr,
                    train_args=train_args,
                    parent_checkpoint=parent_checkpoint,
                    report_episodes=getattr(args, "report_episodes", None),
                    fresh_stage=stage == start_stage and start_stage != STAGES[0],
                    fixed_stage=fixed_stage,
                    physics=args.physics_preset,
                    frontend=args.frontend,
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
                environment["NAVRL_STAGE_STARTED_MONOTONIC"] = str(time.monotonic())
                with stdout_log.open("w", encoding="utf-8") as stream:
                    process = subprocess.run(
                        command,
                        cwd=scripts_dir.parent,
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
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
                    if (
                        not isinstance(reported_checkpoint, str)
                        or Path(reported_checkpoint).resolve() != checkpoint.resolve()
                        or not checkpoint.is_file()
                    ):
                        raise RuntimeError(f"Stopped trainer checkpoint is missing or mismatched: {checkpoint}")
                    current_stage.update(
                        status="stopped",
                        stage_passed=False,
                        acceptance_passed=False,
                        checkpoint=str(checkpoint),
                        error=None,
                    )
                    current_method.update(
                        status="stopped", acceptance_passed=False, checkpoint=str(checkpoint), error=None
                    )
                    summary.update(
                        status="stopped",
                        acceptance_passed=False,
                        checkpoint=str(checkpoint),
                        error=None,
                        finished_at=datetime.now(UTC).isoformat(),
                    )
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
                        current_stage["result"],
                        method=method,
                        stage=stage,
                        num_envs=args.num_envs,
                        parent_checkpoint=parent_checkpoint,
                        variant=train_args.get("variant", "dcol"),
                        algorithm=train_args.get("algorithm", "bptt"),
                        recurrent_model=train_args.get("recurrent_model", "gru"),
                        report_episodes=getattr(args, "report_episodes", None),
                        velocity_loss_mode=train_args.get("velocity_loss_mode"),
                        screening=screening,
                        fresh_start_stage=start_stage,
                        num_dynamic_obstacles=train_args.get("num_dynamic_obstacles"),
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
        summary["complete_curriculum_passed"] = start_stage == STAGES[0] and not fixed_stage
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
    parser.add_argument(
        "--screening",
        action="store_true",
        help="Allow reduced-env fresh comparisons; never label them as formal acceptance",
    )
    parser.add_argument(
        "--start_stage",
        choices=STAGES,
        default=STAGES[0],
        help="Zhang screening only: start fresh here, then follow the remaining stage order",
    )
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument(
        "--fixed_stage", action="store_true", help="Screening only: stay in the start scene after reaching 80%"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--finetune_lr", type=float, default=5e-5)
    parser.add_argument("--train_args", nargs=argparse.REMAINDER, default=[])
    args = parse_physics_args(parser)
    try:
        selected_stages(args.start_stage, screening=args.screening, method=args.method, fixed_stage=args.fixed_stage)
    except ValueError as exc:
        parser.error(str(exc))
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
    if any(key in train_args for key in ("num_dynamic_obstacles", "target_speed_min", "target_speed_max")):
        if not (args.screening and args.fixed_stage and args.method == "zhang"):
            parser.error("Scene overrides require fixed-stage Zhang screening")
    if args.method == "navrl" and train_args:
        parser.error("--train_args contains Zhang hyperparameters; use --method zhang or both")
    if Path(args.output).expanduser().exists():
        parser.error("--output must be a new directory; existing runs cannot be resumed")
    return run_pipeline(args, train_args)


if __name__ == "__main__":
    raise SystemExit(main())
