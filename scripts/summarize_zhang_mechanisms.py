"""Summarize manifest-selected real logs; never infer samples from configured H.

Usage: python scripts/summarize_zhang_mechanisms.py --root RUNS --manifest cases.json --output summary.json
The manifest contains runs: [{name, path, variant, algorithm, normalization, lr_schedule, seed}].
Paths are relative to --root (absolute paths inside it are also accepted).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

STAGES = ("pretrain_s0", "pretrain_s1", "pretrain_s2", "finetune_s1", "finetune_s2")
RATES = ("success_rate", "collision_rate", "bounds_rate", "invalid_rate", "timeout_rate")
IDENTITY = ("variant", "algorithm", "normalization", "lr_schedule", "seed")
NUMERICAL_METRICS = (
    "surrogate_velocity_error_mps", "surrogate_vertical_velocity_bias_mps", "executed_action_max_error",
    "Optim/update_ratio", "Optim/layers/fc/update_ratio", "Optim/head/az/update_ratio", "Optim/head/vz/update_ratio",
)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def read_source(path, jsonl=False):
    """Read one snapshot, retaining valid records before an incomplete append."""
    info = {"path": str(path), "status": "complete", "issues": []}
    empty = [] if jsonl else None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        info.update(status="missing" if isinstance(error, FileNotFoundError) else "partial")
        info["issues"].append({"kind": "read_error", "detail": str(error)})
        return empty, info
    if not jsonl:
        try:
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            return value, info
        except ValueError as error:
            info.update(status="partial")
            info["issues"].append({"kind": "invalid_json", "detail": str(error)})
            return None, info
    records, lines = [], text.splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            records.append({**value, "_source": {"path": str(path), "line": number}})
        except ValueError as error:
            info["status"] = "partial"
            info["issues"].append({"kind": "invalid_jsonl", "line": number, "detail": str(error)})
    if text and not text.endswith(("\n", "\r")):
        info["status"] = "partial"
        info["issues"].append({"kind": "unterminated_tail", "line": len(lines)})
    info["records"] = len(records)
    return records, info


def nonfinite_fields(record):
    return [key for key, value in record.items() if isinstance(value, float) and not math.isfinite(value)]


def enrich_updates(records, result, num_envs, source_info, issues):
    """Attach stage-local counters using logged totals and actual rollout lengths."""
    if not records:
        return []
    first = records[0]
    leading_error = any(issue.get("line", math.inf) < first["_source"]["line"]
                        for issue in source_info["issues"])
    if integer(result.get("completed_updates")) and integer(result.get("stage_updates")):
        expected_first = result["completed_updates"] - result["stage_updates"] + 1
        if first.get("update") != expected_first:
            leading_error = True
            issues.append({"kind": "missing_stage_start", "source": first["_source"],
                           "expected_first_update": expected_first, "first_record_update": first.get("update")})
    first_increment = (first["rollout_steps"] * num_envs
                       if integer(first.get("rollout_steps"), 1) and integer(num_envs, 1) else None)
    baseline = None
    if result.get("fresh") is True:
        baseline = 0
    elif (not leading_error and integer(first.get("environment_steps")) and first_increment is not None
          and first["environment_steps"] >= first_increment):
        baseline = first["environment_steps"] - first_increment
    elapsed_baseline = (first["elapsed_s"] - first["update_s"]
                        if not leading_error and finite(first.get("elapsed_s"))
                        and finite(first.get("update_s")) and first["update_s"] >= 0 else None)
    prefix, previous_update, previous_total = 0 if not leading_error else None, None, None
    enriched = []
    seen = set()
    for record in records:
        update = record.get("update")
        if not integer(update, 1) or update in seen or (previous_update is not None and update <= previous_update):
            issues.append({"kind": "invalid_or_duplicate_update", "source": record["_source"], "update": update})
            continue
        seen.add(update)
        if previous_update is not None and update != previous_update + 1:
            prefix = None
            issues.append({"kind": "update_gap", "source": record["_source"], "previous_update": previous_update})
        previous_update = update
        count = record.get("num_envs", num_envs)
        increment = (record["rollout_steps"] * count
                     if integer(record.get("rollout_steps"), 1) and integer(count, 1) else None)
        prefix = prefix + increment if prefix is not None and increment is not None else None
        total = record.get("environment_steps")
        stage_steps = total - baseline if integer(total) and baseline is not None and total >= baseline else prefix
        if integer(total) and previous_total is not None and total <= previous_total:
            issues.append({"kind": "nonincreasing_environment_steps", "source": record["_source"]})
            stage_steps = None
        if stage_steps is not None and prefix is not None and stage_steps != prefix:
            issues.append({"kind": "environment_step_mismatch", "source": record["_source"],
                           "logged_stage_steps": stage_steps, "summed_actual_rollout_steps": prefix})
            stage_steps = None
        if integer(total):
            previous_total = total
        elapsed = record.get("elapsed_s")
        stage_elapsed = (elapsed - elapsed_baseline if finite(elapsed) and elapsed_baseline is not None
                         and elapsed >= elapsed_baseline else None)
        enriched.append({**record, "_stage_steps": stage_steps, "_stage_elapsed_s": stage_elapsed})
    if enriched and enriched[-1]["_stage_steps"] is None:
        issues.append({"kind": "stage_counter_unavailable", "source": enriched[-1]["_source"]})
    return enriched


def enrich_reports(records, updates, stage, issues):
    by_update = {record["update"]: record for record in updates}
    reports, seen = [], set()
    for record in records:
        update = record.get("update")
        if (not integer(record.get("episodes"), 1) or not finite(record.get("success_rate"))
                or not 0 <= record["success_rate"] <= 1 or record.get("stage", stage) != stage):
            issues.append({"kind": "invalid_natural_episode_report", "source": record["_source"]})
            continue
        if integer(update) and update in seen:
            issues.append({"kind": "duplicate_episode_report", "source": record["_source"], "update": update})
            continue
        if integer(update):
            seen.add(update)
        match = by_update.get(update)
        if match is None:
            issues.append({"kind": "report_without_exact_update", "source": record["_source"], "update": update})
        reports.append({**{key: value for key, value in record.items() if key != "_source"},
                        "source": record["_source"], "update_source": match["_source"] if match else None,
                        "environment_steps": match.get("environment_steps") if match else None,
                        "stage_environment_steps": match["_stage_steps"] if match else None,
                        "elapsed_s": match.get("elapsed_s") if match else None,
                        "stage_elapsed_s": match["_stage_elapsed_s"] if match else None})
    return reports


def recent_reports(reports):
    recent = reports[-5:]
    metrics, coverage = {}, {}
    for key in RATES:
        available = [record for record in recent if finite(record.get(key)) and 0 <= record[key] <= 1]
        episodes = sum(record["episodes"] for record in available)
        # Do not present a partially observed metric as a whole-window rate.
        metrics[key] = (sum(record[key] * record["episodes"] for record in available) / episodes
                        if episodes and len(available) == len(recent) else None)
        coverage[key] = {"reports": len(available), "episodes": episodes}
    return {"report_count": len(recent), "episodes": sum(row["episodes"] for row in recent) if recent else None,
            "aggregation": "episodes_weighted_last_five_natural_reports", "rates": metrics,
            "metric_coverage": coverage, "sources": [row["source"] for row in recent]}


def metric_summary(updates, key):
    available = [record for record in updates if finite(record.get(key))]
    if not available:
        return {"samples": 0, "latest": None, "max": None, "max_abs": None, "max_source": None}
    largest = max(available, key=lambda record: record[key])
    return {"samples": len(available), "latest": available[-1][key], "max": largest[key],
            "max_abs": max(abs(record[key]) for record in available),
            "max_source": {**largest["_source"], "update": largest["update"]}}


def summarize_window(updates, reports, result=None):
    latest = updates[-1] if updates else {}
    durations = [record["update_s"] for record in updates if finite(record.get("update_s")) and record["update_s"] >= 0]
    horizons = [record["rollout_steps"] for record in updates if integer(record.get("rollout_steps"), 1)]
    numerical = {key: metric_summary(updates, key) for key in NUMERICAL_METRICS}
    z_ratios = [numerical[f"Optim/head/{axis}/update_ratio"]["max"] for axis in ("az", "vz")]
    nonfinite = [{"update": record["update"], "fields": nonfinite_fields(record), "source": record["_source"]}
                 for record in updates if nonfinite_fields(record)]
    return {
        "update_records": len(updates), "latest_update": latest.get("update"),
        "stage_environment_steps": latest.get("_stage_steps"), "environment_steps": latest.get("environment_steps"),
        "stage_elapsed_s": latest.get("_stage_elapsed_s"), "elapsed_s": latest.get("elapsed_s"),
        "walltime_scope": "logged first update start through selected final update; simulator startup excluded",
        "logged_update_seconds": sum(durations) if durations else None,
        "median_update_s": statistics.median(durations) if durations else None,
        "rollout_steps_min": min(horizons) if horizons else None, "rollout_steps_max": max(horizons) if horizons else None,
        "natural_report_count": len(reports), "latest_report": reports[-1] if reports else None,
        "best_report": max(reports, key=lambda record: record["success_rate"]) if reports else None,
        "first_natural_80_percent": next((row for row in reports if row["success_rate"] >= 0.8), None),
        "recent_five": recent_reports(reports), "numerical_metrics": numerical,
        "fc_z_head_max_ratio": max(value for value in z_ratios if value is not None) if any(
            value is not None for value in z_ratios) else None,
        "nonfinite_log_record_count": len(nonfinite) if updates else None, "nonfinite_log_records": nonfinite,
        "result_nonfinite_grad_events": result.get("nonfinite_grad_events") if result is not None else None,
        "result_nonfinite_events": result.get("nonfinite_events") if result is not None else None,
        "result_counters_scope": "current stage result, not a historical prefix" if result is not None else None,
    }


def summarize_stage(run_path, stage, declared):
    method = declared.get("method", "zhang")
    directory = run_path / method / stage
    config, config_source = read_source(directory / ("run_config.json" if method == "navrl" else "config.json"))
    result, result_source = read_source(directory / "training_result.json")
    updates, update_source = read_source(directory / "updates.jsonl", jsonl=True)
    reports, report_source = read_source(directory / "training_statistics.jsonl", jsonl=True)
    config, result = config or {}, result or {}
    train, environment = config.get("train", {}), config.get("environment", {})
    observed = {
        "variant": train.get("variant", config.get("variant", result.get("variant"))),
        "algorithm": train.get("algorithm", config.get("algorithm", result.get("algorithm"))),
        "normalization": train.get("observation_normalization"), "lr_schedule": train.get("lr_schedule"),
        "seed": train.get("seed", environment.get("seed")),
    }
    if method == "navrl":
        environment = config
        observed = {
            "variant": "navrl", "algorithm": "ppo", "normalization": False,
            "lr_schedule": config.get("training", {}).get("algorithm", {}).get("schedule"),
            "seed": config.get("seed"),
        }
    mismatches = {key: {"manifest": declared[key], "observed": observed[key]} for key in IDENTITY
                  if observed[key] is not None and observed[key] != declared[key]}
    issues = []
    updates = enrich_updates(updates, result, environment.get("num_envs", result.get("num_envs")), update_source, issues)
    reports = enrich_reports(reports, updates, stage, issues)
    sources = {"config": config_source, "result": result_source, "updates": update_source, "reports": report_source}
    statuses = [source["status"] for source in sources.values()]
    status = ("missing" if all(value == "missing" for value in statuses) else
              "partial" if any(value != "complete" for value in statuses) or issues or mismatches else "complete")
    notes = []
    if stage == "pretrain_s0" and any(record.get("rollout_steps") == 16 for record in updates):
        notes.append("H=16 is shorter than the default 20-step velocity window. Cross-rollout history and valid-window reduction affect early-window coverage; do not treat updates as equal samples.")
    summary = {
        "stage": stage, "path": str(directory), "data_status": status, "training_status": result.get("status"),
        "stage_passed": result.get("stage_passed"), "acceptance_passed": result.get("acceptance_passed"),
        "error": result.get("error"), "sources": sources, "read_validation_issues": issues,
        "observed_identity": observed, "identity_mismatches": mismatches,
        "identity_missing_fields": [key for key in IDENTITY if observed[key] is None],
        "environment": environment, "notes": notes, "summary": summarize_window(updates, reports, result),
    }
    return summary, updates, reports


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("runs"), list) or not manifest["runs"]:
        raise ValueError("manifest must contain a nonempty runs list")
    seen = set()
    for entry in manifest["runs"]:
        if not isinstance(entry, dict) or any(key not in entry for key in ("name", "path", *IDENTITY)):
            raise ValueError("each manifest run needs name, path, variant, algorithm, normalization, lr_schedule, seed")
        if any(not isinstance(entry[key], str) or not entry[key] for key in ("name", "path", "variant", "algorithm", "lr_schedule")):
            raise ValueError("manifest names, paths and algorithm labels must be nonempty strings")
        if type(entry["normalization"]) is not bool or type(entry["seed"]) is not int:
            raise ValueError("manifest normalization must be boolean and seed must be an integer")
        if entry["name"] in seen:
            raise ValueError("manifest experiment names must be unique")
        seen.add(entry["name"])
    return manifest


def summarize(root, manifest_path):
    root, manifest_path = root.resolve(), manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    runs, excluded, raw, selected_paths = [], [], {}, set()
    for entry in manifest["runs"]:
        path = (root / entry["path"]).resolve()
        reason = None
        if not path.is_relative_to(root):
            reason = "outside_root"
        elif any("smoke" in part.lower() or "externaloldrun" in part.lower()
                 for part in (*path.relative_to(root).parts, entry["name"])):
            reason = "smoke_or_externaloldrun"
        elif path in selected_paths:
            raise ValueError("manifest run paths must be unique")
        if reason:
            excluded.append({"name": entry["name"], "path": str(path), "reason": reason})
            continue
        selected_paths.add(path)
        result, source = read_source(path / "training_result.json")
        stages = []
        for stage in STAGES:
            stage_summary, updates, reports = summarize_stage(path, stage, entry)
            stages.append(stage_summary)
            raw[(entry["name"], stage)] = (updates, reports)
        runs.append({"name": entry["name"], "path": str(path), "manifest_identity": {key: entry[key] for key in IDENTITY},
                     "result_source": source, "status": (result or {}).get("status"),
                     "acceptance_passed": (result or {}).get("acceptance_passed"), "stages": stages})
    discovered = set()
    for result_path in root.rglob("training_result.json"):
        candidate = (result_path.parent.parent.parent if result_path.parent.parent.name == "zhang"
                     else result_path.parent)
        if (candidate / "zhang").is_dir():
            discovered.add(candidate.resolve())
    already_excluded = {item["path"] for item in excluded}
    excluded.extend({"path": str(path), "reason": "not_in_manifest"} for path in sorted(discovered - selected_paths)
                    if str(path) not in already_excluded)
    comparisons = []
    for seed in sorted({run["manifest_identity"]["seed"] for run in runs}):
        cohort = [run for run in runs if run["manifest_identity"]["seed"] == seed]
        for index, stage in enumerate(STAGES):
            missing = [run["name"] for run in cohort if run["stages"][index]["identity_mismatches"]
                       or run["stages"][index]["summary"]["stage_environment_steps"] is None]
            budget = (min(run["stages"][index]["summary"]["stage_environment_steps"] for run in cohort)
                      if len(cohort) >= 2 and not missing else None)
            participants = []
            for run in cohort:
                updates, reports = raw[(run["name"], stage)]
                selected_updates = [row for row in updates if budget is not None and integer(row["_stage_steps"])
                                    and row["_stage_steps"] <= budget]
                selected_reports = [row for row in reports if budget is not None and integer(row["stage_environment_steps"])
                                    and row["stage_environment_steps"] <= budget]
                participants.append({"name": run["name"], "data_status": run["stages"][index]["data_status"],
                                     "summary": summarize_window(selected_updates, selected_reports),
                                     "unmatched_report_count": sum(row["stage_environment_steps"] is None for row in reports)})
            comparisons.append({"seed": seed, "stage": stage, "budget_stage_environment_steps": budget,
                                "status": "available" if budget is not None else "insufficient_or_missing_data",
                                "missing_or_mismatched_runs": missing, "participants": participants,
                                "sampling": "whole logged updates/reports at or below the common stage budget; no interpolation"})
    return clean({
        "format_version": 1, "generated_at": datetime.now(UTC).isoformat(), "root": str(root),
        "manifest": str(manifest_path), "runs": runs, "excluded_runs": excluded, "comparisons": comparisons,
        "limitations": [
            "Only manifest-selected runs are evidence. Missing or partial files are visible and missing rates remain null.",
            "Natural reports come from training_statistics.jsonl; no extra evaluation or synthetic episodes are created.",
            "Step budgets use logged cumulative environment_steps and actual per-update rollout_steps, never configured H times update count.",
            "Exact update matches are required for report timing and prefix comparisons. Whole updates can leave different unused budget remainders.",
            "Recent-five rates are weighted by completed episodes. A single best report does not establish stability and this tool never changes defaults.",
            "Concurrent GPU runs make elapsed time a schedule-dependent wall-clock observation, not an exclusive throughput benchmark.",
            "Same-seed cases are paired comparisons, not independent seed replications. Reset semantics and surrogate dynamics differ by variant.",
            "S0 at H=16 is shorter than the default 20-step velocity window; interpret valid-window coverage and cross-rollout history explicitly.",
        ],
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error("--root must be an existing directory")
    try:
        result = summarize(args.root, args.manifest)
        output = args.output.expanduser().resolve()
        sources = {args.manifest.resolve()}
        for run in result["runs"]:
            sources.add(Path(run["result_source"]["path"]))
            for stage in run["stages"]:
                sources.update(Path(source["path"]) for source in stage["sources"].values())
        if output in sources or output.name in {"training_result.json", "config.json", "updates.jsonl", "training_statistics.jsonl"}:
            raise ValueError("--output must not overwrite a source file")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(output), "runs": len(result["runs"]), "comparisons": len(result["comparisons"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
