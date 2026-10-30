"""Read-only first-order projection analysis of saved Zhang GradientDiagnostic logs.

Uses stored seven-term Gram matrices; never alters actors, gradients, optimizers,
running training processes, or checkpoints. The projected update is a hypothetical
clipped SGD direction, NOT a simulated AdamW update or a new environment rollout.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def project_from_gram(record: dict, group: str) -> dict:
    """Project the other-task gradient to avoid opposing the height gradient.

    All operations occur in the parameter subspace described by 'group'.
    'guard' minimally prevents an uphill height direction; 'priority' additionally
    preserves the weighted height term's full first-order descent component.
    """
    names = record["term_order"]
    h = names.index("height_boundary")
    c = names.index("collision")
    w = np.asarray(record["coefficients"], dtype=np.float64)
    g = np.asarray(record["groups"][group]["raw_gram"], dtype=np.float64)
    if g.shape != (len(w), len(w)) or not np.isfinite(g).all():
        raise ValueError("Invalid component-gradient Gram matrix")
    g = 0.5 * (g + g.T)
    height_sq = float(g[h, h])
    if height_sq < -1e-9:
        raise ValueError("Height gradient has a negative squared norm")
    height_sq = max(0.0, height_sq)
    total_height_dot = float(g[h] @ w)
    other_height_dot = total_height_dot - w[h] * height_sq
    gradient_norm = math.sqrt(max(0.0, float(w @ g @ w)))
    observed = record["groups"][group]["linearized_delta_loss"]
    max_grad_norm = float(record["max_grad_norm"])

    def compute(extra: float) -> dict:
        changed = w.copy()
        changed[h] += extra
        squared = float(changed @ g @ changed)
        norm = math.sqrt(max(0.0, squared))
        clip = min(1.0, max_grad_norm / (norm + 1e-6))
        height_dot = float(g[h] @ changed)
        collision_dot = float(g[c] @ changed)
        cosine = float(w @ g @ changed) / max(gradient_norm * norm, 1e-30)
        return {
            "additional_height_gradient_coefficient": extra,
            "effective_height_coefficient": float(changed[h]),
            "unclipped_gradient_norm": norm,
            "clip_scale": clip,
            "angle_to_original_degrees": math.degrees(math.acos(max(-1.0, min(1.0, cosine)))),
            "height_gradient_dot_update": height_dot,
            "height_loss_change_per_unit_sgd_lr": -height_dot * clip,
            "collision_loss_change_per_unit_sgd_lr": -collision_dot * clip,
        }

    if height_sq < 1e-20:
        minimum_extra = priority_extra = 0.0
    else:
        minimum_extra = max(0.0, -total_height_dot / height_sq)
        priority_extra = max(0.0, -other_height_dot / height_sq)
    original = compute(0.0)
    guard = compute(minimum_extra)
    priority = compute(priority_extra)
    if height_sq >= 1e-20:
        tolerance = 1e-7 * max(1.0, abs(total_height_dot))
        if guard["height_gradient_dot_update"] < -tolerance:
            raise AssertionError("Guard projection failed")
        if priority["height_gradient_dot_update"] < w[h] * height_sq - tolerance:
            raise AssertionError("Height-priority projection failed")
    return {
        "group": group,
        "height_gradient_norm": math.sqrt(height_sq),
        "collision_height_cosine": record["groups"][group]["cosine"][c][h],
        "other_gradient_dot_height": other_height_dot,
        "actual_adamw_height_loss_first_order_change": float(observed[h]),
        "actual_adamw_collision_loss_first_order_change": float(observed[c]),
        "original": original,
        "non_worsen_height": guard,
        "preserve_height_descent": priority,
    }


def report(root: Path, runs: list[str]) -> dict:
    records = []
    for run in runs:
        for path in sorted((root / run / "zhang").glob("*/gradient_diagnostics.jsonl")):
            for raw in path.read_text().splitlines():
                if not raw.strip():
                    continue
                record = json.loads(raw)
                entry = {
                    "run": run,
                    "stage": path.parent.name,
                    "update": record["update"],
                    "environment_steps": record.get("environment_steps"),
                    "upper_boundary_active_fraction": record.get("upper_boundary_active_fraction"),
                    "lower_boundary_active_fraction": record.get("lower_boundary_active_fraction"),
                    "mean_altitude_m": record.get("rollout", {}).get("mean_altitude_m"),
                    "groups": {group: project_from_gram(record, group) for group in ("all", "trunk", "z_head")},
                }
                records.append(entry)
    records.sort(key=lambda x: (x["run"], x["stage"], x["update"]))
    return {
        "analysis": "saved_gram_first_order_counterfactual",
        "model_parameters_modified": False,
        "optimizer_steps_executed": False,
        "limitations": [
            "Only reweights recorded gradients on their original rollout; no policy retraining.",
            "Projected updates are hypothetical globally clipped SGD directions, not AdamW steps.",
            "Actual AdamW displacement columns describe the unmodified recorded training update.",
            "A height-preserving parameter direction is not a guaranteed safe flight trajectory.",
            "Parameter-space projection cannot create absent lateral obstacle-avoidance gradients.",
        ],
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs/curriculum"))
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "zhang_followup_continuous_signed_seed0_20261029",
            "zhang_step_dcol_bptt_seed0_20261029",
            "zhang_continuous_height8_retry1_seed0_20261029",
            "zhang_step_h5_lr0p0005_s0_20261029",
        ],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/results/zhang_gradient_projection_counterfactual_20261029.json")
    )
    args = parser.parse_args()
    result = report(args.root, args.runs)
    if not result["records"]:
        raise RuntimeError("No saved gradient diagnostics found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("Analyzed", len(result["records"]), "saved diagnostic updates; output:", args.output)
    for row in result["records"]:
        v = row["groups"]["all"]
        print(
            row["run"],
            "update",
            row["update"],
            "height/collision cosine",
            round(v["collision_height_cosine"], 4) if v["collision_height_cosine"] is not None else None,
            "SGD height cost Δ orig/guard/priority",
            *(
                round(v[k]["height_loss_change_per_unit_sgd_lr"], 4)
                for k in ("original", "non_worsen_height", "preserve_height_descent")
            ),
            "priority extra coefficient",
            round(v["preserve_height_descent"]["additional_height_gradient_coefficient"], 4),
            "actual AdamW Δ height",
            round(v["actual_adamw_height_loss_first_order_change"], 6),
        )


if __name__ == "__main__":
    main()
