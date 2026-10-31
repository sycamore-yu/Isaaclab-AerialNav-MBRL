"""Shared training lifecycle and reporting; no extra evaluation rollout."""

import json
import os
import signal
from contextlib import contextmanager
from pathlib import Path

STAGES = ("pretrain_s0", "pretrain_s1", "pretrain_s2", "finetune_s1", "finetune_s2")
DEFAULT_STAGE = "pretrain_s0"
STAGE_CONFIG = {
    "pretrain_s0": {"static_obstacles": 300, "dynamic_obstacles": 0, "randomization": False},
    "pretrain_s1": {"static_obstacles": 300, "dynamic_obstacles": 40, "randomization": False},
    "pretrain_s2": {"static_obstacles": 400, "dynamic_obstacles": 120, "randomization": False},
    "finetune_s1": {"static_obstacles": 300, "dynamic_obstacles": 40, "randomization": True},
    "finetune_s2": {"static_obstacles": 400, "dynamic_obstacles": 100, "randomization": True},
}


def tensorboard_directory(log_dir):
    """A pipeline shares one event directory across curriculum stages."""
    return Path(os.environ.get("AERIALNAV_TENSORBOARD_DIR", str(Path(log_dir) / "tensorboard")))


def write_training_scalars(writer, metrics, update):
    """Keep dashboard essentials; full diagnostics remain in updates.jsonl."""
    groups = {
        "Loss": (
            "total",
            "actor_objective",
            "velocity",
            "velocity_prediction",
            "obstacle_avoidance",
            "collision",
            "height_boundary",
            "acceleration",
            "jerk",
            "failure_terminal",
        ),
        "Optimization": ("learning_rate", "grad_norm"),
        "Timing": ("process_elapsed_s", "update_s", "environment_steps", "rollout_steps"),
        "Motion": (
            "mean_altitude_m",
            "mean_horizontal_speed_mps",
            "mean_abs_vertical_speed_mps",
            "surrogate_velocity_error_mps",
            "relative_closing_mps",
            "target_speed_scale",
        ),
        "Failures": ("terminal_lower_bound_count", "terminal_upper_bound_count"),
        "Observation": ("depth_age_s", "depth_known_fraction"),
    }
    for group, names in groups.items():
        for name in names:
            if name in metrics:
                writer.add_scalar(f"{group}/{name}", metrics[name], update)
    for name in (
        "ActorCritic/critic_loss",
        "ActorCritic/bootstrap",
        "ActorCritic/discounted_cost",
        "ActorCritic/critic_next_value_negative_fraction",
        "Memory/torch_peak_allocated_mb",
    ):
        if name in metrics:
            writer.add_scalar(name, metrics[name], update)
    if "Optim/update_ratio" in metrics:
        writer.add_scalar("Optimization/update_ratio", metrics["Optim/update_ratio"], update)


@contextmanager
def graceful_training_stop(trainer):
    """Request stopping at an update boundary, then restore process handlers."""
    previous = {}

    def request_stop(signum, _frame):
        trainer.request_stop(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, request_stop)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def record_curriculum(curriculum, update, log_dir, writer, save_checkpoint):
    if curriculum is None:
        return None
    report = curriculum.pop_report(update)
    if report is None:
        return None
    log_dir = Path(log_dir)
    difficulty = report.get("stage", report["dynamic_obstacles"])
    if report["new_best"]:
        checkpoint = log_dir / f"checkpoint_best_{difficulty}.pt"
        save_checkpoint(checkpoint)
        curriculum.best_by_stage[difficulty]["checkpoint"] = str(checkpoint.resolve())
        report["checkpoint"] = str(checkpoint.resolve())
    with (log_dir / "training_statistics.jsonl").open("a") as stream:
        stream.write(json.dumps(report, allow_nan=False) + "\n")
    for key, value in report.items():
        if key.endswith(("_rate", "_mean")) or key in ("episodes", "success_time_mean_s"):
            if value is not None:
                writer.add_scalar(f"Train/{key}", value, update)
    writer.add_scalar("Curriculum/dynamic_obstacles", report["dynamic_obstacles"], update)
    writer.add_scalar("Train/stage_passed", int(curriculum.stage_passed), update)
    writer.add_scalar("Train/acceptance_passed", int(curriculum.acceptance_passed), update)
    if difficulty in STAGES:
        writer.add_scalar("Curriculum/stage", STAGES.index(difficulty), update)
    if "static_obstacles" in report:
        writer.add_scalar("Curriculum/static_obstacles", report["static_obstacles"], update)
        writer.add_scalar("Curriculum/randomization", int(report["randomization"]), update)
    writer.flush()
    write_json(log_dir / "curriculum.json", curriculum.state_dict())
    print("TRAINING_EPISODES " + json.dumps(report, allow_nan=False), flush=True)
    return report
