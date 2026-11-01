"""Compute threshold times from complete-episode reports and per-update wall-clock logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(statistics: Path, updates: Path) -> dict:
    reports = [json.loads(line) for line in statistics.read_text().splitlines() if line.strip()]
    metrics = {
        row["update"]: row for row in (json.loads(line) for line in updates.read_text().splitlines() if line.strip())
    }
    if not reports:
        raise ValueError("No training episode reports available")
    if not metrics:
        raise ValueError("No training update metrics available")

    def point(report: dict) -> dict:
        update = metrics[report["update"]]
        elapsed = float(update["process_elapsed_s"])
        steps = int(update["environment_steps"])
        return {
            "update": int(report["update"]),
            "success_count": int(report["counts"]["success"]),
            "completed_episodes": int(report["episodes"]),
            "success_rate": float(report["success_rate"]),
            "process_elapsed_s": elapsed,
            "environment_steps": steps,
            "average_environment_steps_per_s": steps / elapsed,
            "static_obstacles": int(report["static_obstacles"]),
            "dynamic_obstacles": int(report["dynamic_obstacles"]),
        }

    targets = (0.5, 0.7, 0.8)
    threshold_points = {}
    for threshold in targets:
        match = next((row for row in reports if row["success_rate"] >= threshold), None)
        threshold_points[f"{threshold:.0%}"] = point(match) if match is not None else None
    return {
        "description": "Single-seed training reports, not an independent evaluation",
        "wall_clock_definition": "process_elapsed_s: cumulative training process time including initialization",
        "time_to_threshold": threshold_points,
        "best": point(max(reports, key=lambda r: r["success_rate"])),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--updates", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.statistics, args.updates)
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            parser.error(f"Refusing to overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end="")
