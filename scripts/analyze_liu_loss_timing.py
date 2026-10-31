"""Summarize every requested timing/reset cell, including pending and failed runs."""

import argparse
import itertools
import json
from pathlib import Path

from isaaclab_navrl.training import write_json


def json_rows(path):
    return [json.loads(line) for line in path.read_text().split("\n")[:-1] if line] if path.exists() else []


def summarize(root):
    plan = json.loads((root / "plan.json").read_text())
    state = json.loads((root / "study.json").read_text())
    rows = []
    cases = plan.get("runs")
    if cases is None:
        cases = [
            {
                "name": f"train_{family}_{timing}_height{'reset' if height else 'continue'}",
                "loss_family": family,
                "loss_timing": timing,
                "height_bounds_reset": height,
            }
            for family, timing, height in itertools.product(
                plan.get("families", ["zhang"]), plan["timings"], plan["height_bounds_reset"]
            )
        ]
    for case in cases:
        name = case["name"]
        family = case.get("loss_family", "zhang")
        timing = case.get("loss_timing", plan.get("loss_timing"))
        height = case.get("height_bounds_reset", plan.get("height_bounds_reset"))
        directory = root / name
        updates = json_rows(directory / "updates.jsonl")
        reports = json_rows(directory / "training_statistics.jsonl")
        counts = {
            key: sum(report["counts"].get(key, 0) for report in reports)
            for key in ("success", "collision", "bounds", "invalid", "timeout")
        }
        reported = sum(counts.values())
        completed = sum(update["completed_episodes"] for update in updates)
        if reported != sum(report["episodes"] for report in reports) or reported > completed:
            raise ValueError(f"Inconsistent completed-episode denominator: {name}")
        path = root / (name + "_evaluation.json")
        evaluation = json.loads(path.read_text()) if path.exists() else None
        if evaluation:
            if (evaluation["seed"], evaluation["terrain_seed"], evaluation["stage"]) != (
                plan["evaluation_seed"],
                plan["evaluation_terrain_seed"],
                plan["stage"],
            ):
                raise ValueError("Evaluation changed the common task")
        measured = updates[2:]
        seconds = sum(update["update_s"] for update in measured)
        rows.append(
            {
                "name": name,
                "collision_metric": case.get("collision_metric"),
                "position_integration": case.get("position_integration"),
                "loss_family": family,
                "loss_timing": timing,
                "height_bounds_reset": height,
                "status": state["runs"].get(name, {}).get("status", "pending"),
                "updates": len(updates),
                "completed_training_episodes": completed,
                "reported_training_episodes": reported,
                "unreported_training_episodes": completed - reported,
                "training_counts": counts,
                "training_reported_success_rate": counts["success"] / reported if reported else None,
                "height_violation_step_fraction": sum(
                    update["height_below_0p2_fraction"] + update["height_above_4_fraction"] for update in updates
                )
                / len(updates)
                if updates
                else None,
                "measured_full_update_transitions_per_s": len(measured) * plan["num_envs"] * plan["timesteps"] / seconds
                if seconds
                else None,
                "torch_peak_allocated_mib": max(
                    (update["Memory/torch_peak_allocated_mb"] for update in updates), default=None
                ),
                "evaluation": evaluation["summary"] if evaluation else None,
            }
        )
    return {
        "status": state["status"],
        "source_commit": state["source_commit"],
        "requested_cells": len(rows),
        "completed_cells": sum(row["status"] == "completed" for row in rows),
        "rows": rows,
        "limits": f"Single seed, {plan['updates']}-update {plan['stage']} budget, H{plan['timesteps']}; shared GPU; not convergence acceptance. Training reset rules differ; compare held-out evaluations under the common original reset rule. Unreported completed episodes remain visible.",
        "performance_boundary": "update_s after two warmup updates; excludes inter-update checkpoint/report IO; not an isolated speed comparison",
    }


def write_report(root):
    result = summarize(root)
    write_json(root / "comparison.json", result)
    lines = [
        "# Loss timing and height reset controls",
        "",
        result["limits"],
        "",
        f"Status: {result['status']}; completed {result['completed_cells']}/{result['requested_cells']} cells.",
        "",
        "| Control | Collision metric | Position integration | Timing | Height reset | Updates | Reported train success | Common evaluation | Status |",
        "|---|---|---|---|---|---:|---|---|---|",
    ]
    for row in result["rows"]:
        train = (
            f"{row['training_counts']['success']}/{row['reported_training_episodes']}"
            if row["reported_training_episodes"]
            else "pending"
        )
        evaluation = json.dumps(row["evaluation"], ensure_ascii=False) if row["evaluation"] else "pending"
        lines.append(
            f"| {row['name']} | {row['collision_metric']} | {row['position_integration']} | {row['loss_timing']} | {row['height_bounds_reset']} | {row['updates']} | {train} | {evaluation} | {row['status']} |"
        )
    lines += [
        "",
        "Full failure counts, unreported episodes, height violations, memory and measured update throughput are in comparison.json.",
        "",
        result["performance_boundary"],
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    write_report(args.root)


if __name__ == "__main__":
    main()
