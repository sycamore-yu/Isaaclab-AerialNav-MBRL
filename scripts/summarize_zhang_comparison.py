"""Summarize existing Zhang comparison runs without changing training artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from isaaclab_navrl.training import STAGES

REPORT_METRICS = (
    "success_rate", "collision_rate", "bounds_rate", "invalid_rate", "timeout_rate",
    "success_time_mean_s", "episode_return_mean", "episode_length_mean", "reward_per_step_mean",
)
CONTROL_FIELDS = (
    "num_envs", "timesteps", "depth_width", "depth_height", "policy_shape", "seed", "terrain_seed",
    "physics_dt", "policy_dt", "static_obstacles", "dynamic_obstacles", "randomization",
    "velocity_window_s", "lr", "grad_decay", "max_grad_norm", "cosine_period_updates",
    "min_lr_ratio", "coef_v", "coef_v_pred", "coef_obj_avoidance", "coef_collide", "coef_d_acc", "coef_d_jerk",
    "coef_height_boundary", "boundary_margin", "boundary_scale", "height_bounds",
    "depth_pixel_noise_std", "sim_to_real",
)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def clean_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    return value


def read_object(path: Path, issues: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        issues.append({"file": str(path), "kind": "missing" if isinstance(error, FileNotFoundError) else "read_error",
                       "error": str(error)})
        return None


def read_lines(path: Path, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        issues.append({"file": str(path), "kind": "missing" if isinstance(error, FileNotFoundError) else "read_error",
                       "error": str(error)})
        return []
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            records.append(value)
        except ValueError as error:
            issues.append({"file": str(path), "kind": "invalid_jsonl", "line": number,
                           "last_line": number == len(lines), "error": str(error)})
    return records


def comparison_fields(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    train = config.get("train") or {}
    environment = config.get("environment") or {}
    camera = environment.get("depth_camera") or {}
    fields = {name: train.get(name) for name in CONTROL_FIELDS}
    for name in ("num_envs", "seed", "terrain_seed", "physics_dt", "policy_dt",
                 "static_obstacles", "dynamic_obstacles", "randomization"):
        fields[name] = environment.get(name)
    fields.update(variant=train.get("variant", config.get("variant")),
                  depth_width=camera.get("width"), depth_height=camera.get("height"),
                  policy_shape=camera.get("policy_shape"),
                  depth_pixel_noise_std=camera.get("pixel_noise_std"),
                  sim_to_real=environment.get("sim_to_real"),
                  collision_response=environment.get("collision_response"),
                  collision_terminates=environment.get("collision_terminates"))
    return fields


def recent_statistics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    recent = reports[-5:]
    metrics = {}
    for name in REPORT_METRICS:
        samples = []
        for report in recent:
            value, weight = report.get(name), report.get("episodes")
            if name == "success_time_mean_s":
                weight = report.get("counts", {}).get("success")
            elif name == "reward_per_step_mean":
                length = report.get("episode_length_mean")
                weight = weight * length if finite(weight) and finite(length) else None
            if finite(value) and finite(weight) and weight > 0:
                count = report.get("counts", {}).get(name.removesuffix("_rate"))
                if name.endswith("_rate") and finite(count):
                    value = count / weight
                samples.append((value, weight))
        denominator = sum(weight for _, weight in samples)
        mean = sum(value * weight for value, weight in samples) / denominator if denominator else None
        variance = sum(weight * (value - mean)**2 for value, weight in samples) / denominator if denominator else None
        metrics[name] = {"n": len(samples), "denominator": denominator, "mean": mean,
                         "std": math.sqrt(variance) if len(samples) >= 2 else None}
    return {"report_count": len(recent), "aggregation": "weighted_natural_episodes",
            "std_kind": "weighted_between_report_population", "metrics": metrics}


def nonfinite_fields(record: dict[str, Any]) -> list[str]:
    return [name for name, value in record.items() if isinstance(value, float) and not math.isfinite(value)]


def summarize_stage(root: Path, stage: str) -> dict[str, Any]:
    directory = root / "zhang" / stage
    issues: list[dict[str, Any]] = []
    config = read_object(directory / "config.json", issues)
    result = read_object(directory / "training_result.json", issues)
    updates = read_lines(directory / "updates.jsonl", issues)
    reports = read_lines(directory / "training_statistics.jsonl", issues)
    fields = comparison_fields(config)
    result_values = result or {}
    scored = [report for report in reports if finite(report.get("success_rate"))]
    best = max(scored, key=lambda report: report["success_rate"]) if scored else None
    first_pass = next((report for report in scored if report["success_rate"] >= 0.8 and report.get("episodes", 0) >= 1024), None)
    first_80 = None
    if first_pass is not None:
        update = first_pass.get("update")
        matching = next((record for record in updates if record.get("update") == update), None)
        elapsed = matching.get("elapsed_s") if matching is not None else None
        first_80 = {"update": update, "success_rate": first_pass["success_rate"],
                    "episodes": first_pass.get("episodes"),
                    "elapsed_s": elapsed if finite(elapsed) else None,
                    "elapsed_match": "exact_update" if finite(elapsed) else "missing",
                    "process_elapsed_s": matching.get("process_elapsed_s") if matching else None,
                    "stage_updates": None, "stage_environment_steps": None, "environment_steps": None}
        horizon, num_envs = fields["timesteps"], fields["num_envs"]
        completed, stage_updates = result_values.get("completed_updates"), result_values.get("stage_updates")
        if all(isinstance(value, int) and not isinstance(value, bool) for value in (update, horizon, num_envs, completed, stage_updates)):
            stage_count = update - (completed - stage_updates)
            if horizon > 0 and num_envs > 0 and 0 < stage_count <= stage_updates:
                steps_per_update = horizon * num_envs
                first_80["stage_updates"] = stage_count
                first_80["stage_environment_steps"] = stage_count * steps_per_update
                total_steps = result_values.get("environment_steps")
                if isinstance(total_steps, int) and total_steps >= stage_updates * steps_per_update:
                    first_80["environment_steps"] = total_steps - (stage_updates - stage_count) * steps_per_update
    held = next((reports[i:i+3] for i in range(len(reports)-2)
                 if all(finite(r.get("success_rate")) and r["success_rate"] >= 0.8
                        and r.get("episodes", 0) >= 1024 for r in reports[i:i+3])), None)
    retention = None
    if held:
        endpoint = next((r for r in updates if r.get("update") == held[-1]["update"]), {})
        retention = {"first_update": held[0]["update"], "confirmed_update": held[-1]["update"],
                     "episodes": sum(r["episodes"] for r in held), "elapsed_s": endpoint.get("elapsed_s"),
                     "process_elapsed_s": endpoint.get("process_elapsed_s"),
                     "scope": "three_nonoverlapping_training_reports_not_independent_evaluation"}
    durations = [record["update_s"] for record in updates if finite(record.get("update_s"))]
    action_errors = [record["executed_action_max_error"] for record in updates
                     if finite(record.get("executed_action_max_error"))]
    nonfinite_updates = [{"update": record.get("update"), "fields": nonfinite_fields(record)}
                         for record in updates if nonfinite_fields(record)]
    summary = {
        "stage": stage, "directory": str(directory), "config": config, "comparison_fields": fields,
        **{name: result_values.get(name) for name in (
            "status", "stage_passed", "acceptance_passed", "stage_updates", "completed_updates",
            "environment_steps", "elapsed_s", "error",
        )},
        "report_count": len(reports), "update_record_count": len(updates),
        "latest_report": reports[-1] if reports else None, "best_report": best,
        "recent_reports": recent_statistics(reports), "first_80_percent": first_80, "stable_80_percent": retention,
        "performance": {"median_update_s": statistics.median(durations) if durations else None,
                        "update_s_samples": len(durations)},
        "numerical_checks": {
            "nonfinite_grad_events": result_values.get("nonfinite_grad_events"),
            "nonfinite_events": result_values.get("nonfinite_events"),
            "nonfinite_update_count": len(nonfinite_updates) if updates else None,
            "nonfinite_updates": nonfinite_updates,
            "max_executed_action_error": max(action_errors) if action_errors else None,
            "action_error_samples": len(action_errors),
            "action_error_updates_above_1e_6": sum(value > 1e-6 for value in action_errors) if action_errors else None,
        },
        "read_issues": issues,
    }
    return clean_json(summary)


def summarize_runs(roots: list[Path]) -> dict[str, Any]:
    runs = []
    for source in roots:
        root = source.expanduser().resolve()
        issues: list[dict[str, Any]] = []
        result = read_object(root / "training_result.json", issues)
        result_values = result or {}
        runs.append({"run": str(root), "name": root.name, "status": result_values.get("status"),
                     "acceptance_passed": result_values.get("acceptance_passed"), "error": result_values.get("error"),
                     "read_issues": issues, "stages": [summarize_stage(root, stage) for stage in STAGES]})
    comparisons = []
    for index, stage in enumerate(STAGES):
        differences, missing = {}, {}
        for name in (*CONTROL_FIELDS, "variant", "collision_response", "collision_terminates"):
            values = {run["run"]: run["stages"][index]["comparison_fields"].get(name) for run in runs}
            known = [value for value in values.values() if value is not None]
            if len({json.dumps(value, sort_keys=True) for value in known}) > 1:
                differences[name] = values
            if len(known) < len(runs):
                missing[name] = [run for run, value in values.items() if value is None]
        controls_match = len(runs) >= 2 and not any(name in differences or name in missing for name in CONTROL_FIELDS)
        comparisons.append({"stage": stage, "control_fields_match": controls_match,
                            "differing_fields": differences, "missing_fields": missing})
    return {
        "generated_at": datetime.now(UTC).isoformat(), "runs": runs, "comparison": comparisons,
        "interpretation": [
            "Missing files or measurements remain null; missing measurements are not zero or an improvement.",
            "The latest five reports describe within-run variation. Their sample standard deviation does not establish across-seed stability or statistical significance.",
            "Each input directory is summarized as one run. Matching seeds do not supply evidence from different seeds; see the recorded seed values.",
            "Elapsed times and update durations reflect this execution schedule. Concurrent work on the same GPU prevents interpreting them as isolated speed benchmarks.",
            "The first 80% elapsed time is read only from the update with the same logged update number. No interpolation is used.",
            "Variant and collision-response differences are intentional experiment settings. Interpret arrival rates together with collision rates and the recorded termination rules.",
        ],
    }


def number(value: Any, percent: bool = False) -> str:
    if value is None:
        return "missing"
    if finite(value):
        return f"{100 * value:.2f}%" if percent else f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def mean_std(metric: dict[str, Any], percent: bool = False) -> str:
    return f"{number(metric['mean'], percent)} ± {number(metric['std'], percent)} (n={metric['n']})"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Zhang 训练对比", "", f"生成时间：{summary['generated_at']}", "",
             "本报告读取现有训练记录。`missing` 表示未记录或文件不可读。", "",
             "最近最多 5 份报告使用样本标准差；n < 2 时不计算标准差。这些波动不能说明跨种子稳定性。",
             "每个输入目录按一次运行汇总。相同种子不会增加不同种子下的独立重复次数。",
             "GPU 并发时，耗时只反映本次调度，不能用作单独运行的速度基准。", ""]
    for run in summary["runs"]:
        lines.append(f"- `{run['run']}`：status={number(run['status'])}；acceptance={number(run['acceptance_passed'])}")
        if run["error"]:
            lines.append(f"  错误：{run['error']}")
    for index, comparison in enumerate(summary["comparison"]):
        stage = comparison["stage"]
        lines += ["", f"## {stage}", "",
                  "| 运行 | 方案 | 环境数 | H | 深度相机 | 策略深度输入 | seed / terrain_seed |",
                  "|---|---|---:|---:|---|---|---|"]
        for run in summary["runs"]:
            fields = run["stages"][index]["comparison_fields"]
            depth = f"{number(fields['depth_width'])}×{number(fields['depth_height'])}"
            lines.append(f"| {number(run['name'])} | {number(fields['variant'])} | {number(fields['num_envs'])} | {number(fields['timesteps'])} | {depth} | {number(fields['policy_shape'])} | {number(fields['seed'])} / {number(fields['terrain_seed'])} |")
        differing_controls = [name for name in CONTROL_FIELDS if name in comparison["differing_fields"]]
        missing_controls = [name for name in CONTROL_FIELDS if name in comparison["missing_fields"]]
        if comparison["control_fields_match"]:
            lines += ["", "已核对的对照字段一致。方案和碰撞处理规则分别列于 JSON。"]
        else:
            lines += ["", f"对照配置差异：{', '.join(differing_controls) or '未发现已知差异'}。",
                      f"待核对的缺失配置：{', '.join(missing_controls) or '无'}。"]
        lines += ["", "| 运行 | 阶段状态 | 最新到达率 | 最佳到达率 | 最新碰撞率 | 最新越界率 | 最新超时率 |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for run in summary["runs"]:
            record = run["stages"][index]
            latest, best = record["latest_report"] or {}, record["best_report"] or {}
            lines.append(f"| {number(run['name'])} | {number(record['status'])} | {number(latest.get('success_rate'), True)} | {number(best.get('success_rate'), True)} | {number(latest.get('collision_rate'), True)} | {number(latest.get('bounds_rate'), True)} | {number(latest.get('timeout_rate'), True)} |")
        lines += ["", "| 运行 | 最新成功时间 (s) | 最新 return | 最新回合长度 | 更新耗时中位数 (s) | 首次80%更新 | 对应累计耗时 (s) | 对应阶段环境步数 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for run in summary["runs"]:
            record = run["stages"][index]
            latest, passed = record["latest_report"] or {}, record["first_80_percent"] or {}
            lines.append(f"| {number(run['name'])} | {number(latest.get('success_time_mean_s'))} | {number(latest.get('episode_return_mean'))} | {number(latest.get('episode_length_mean'))} | {number(record['performance']['median_update_s'])} | {number(passed.get('update'))} | {number(passed.get('elapsed_s'))} | {number(passed.get('stage_environment_steps'))} |")
        lines += ["", "### 最近最多 5 份报告", "",
                  "| 运行 | 报告数 | 到达率均值 ± 标准差 | 碰撞率均值 ± 标准差 | 越界率均值 ± 标准差 | 超时率均值 ± 标准差 |",
                  "|---|---:|---|---|---|---|"]
        for run in summary["runs"]:
            recent = run["stages"][index]["recent_reports"]
            metrics = recent["metrics"]
            values = " | ".join(mean_std(metrics[name], True) for name in ("success_rate", "collision_rate", "bounds_rate", "timeout_rate"))
            lines.append(f"| {number(run['name'])} | {recent['report_count']} | {values} |")
        lines += ["", "| 运行 | 非有限梯度事件 | 非有限环境事件 | 日志非有限更新数 | 最大执行动作误差 | 超过1e-6的更新数 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for run in summary["runs"]:
            checks = run["stages"][index]["numerical_checks"]
            values = " | ".join(number(checks[name]) for name in (
                "nonfinite_grad_events", "nonfinite_events", "nonfinite_update_count", "max_executed_action_error",
                "action_error_updates_above_1e_6",
            ))
            lines.append(f"| {number(run['name'])} | {values} |")
    issues = [issue for run in summary["runs"] for issue in (
        run["read_issues"] + [item for stage in run["stages"] for item in stage["read_issues"]]
    )]
    if issues:
        lines += ["", "## 文件读取情况", "",
                  f"共有 {len(issues)} 项缺失或读取问题。具体文件、行号和错误保存在 JSON 的 `read_issues` 中。"]
    lines += ["", "JSON 同时保留每阶段原始配置、最新及最佳报告、全部近期指标统计、首次80%的累计环境步数和数值检查证据。", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run_acceptance output directories")
    parser.add_argument("--output", type=Path, default=Path("zhang_comparison"),
                        help="report prefix; writes PREFIX.json and PREFIX.md")
    args = parser.parse_args(argv)
    summary = summarize_runs(args.runs)
    json_path, markdown_path = Path(f"{args.output}.json"), Path(f"{args.output}.md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(clean_json(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(json_path.resolve())
    print(markdown_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
