"""Shared training-only reporting; no extra evaluation environment or rollout."""

import json
from pathlib import Path


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
    difficulty = report["dynamic_obstacles"]
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
                writer.add_scalar(f"Train/{difficulty}/{key}", value, update)
    writer.add_scalar("Curriculum/dynamic_obstacles", difficulty, update)
    writer.add_scalar("Train/100/acceptance_passed", int(curriculum.acceptance_passed), update)
    writer.flush()
    write_json(log_dir / "curriculum.json", curriculum.state_dict())
    print("TRAINING_EPISODES " + json.dumps(report, allow_nan=False), flush=True)
    return report
