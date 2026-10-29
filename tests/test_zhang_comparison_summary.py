"""Summary-reader unit tests using temporary fixtures, never real training evidence."""

import contextlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_zhang_comparison.py"
SPEC = importlib.util.spec_from_file_location("zhang_comparison_summary", SCRIPT)
summary_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary_tool)


class ComparisonSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="zhang_summary_unit_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "fixture_run"
        self.directory = self.root / "zhang" / "pretrain_s1"
        self.directory.mkdir(parents=True)

    def write(self, name, value):
        (self.directory / name).write_text(json.dumps(value), encoding="utf-8")

    def lines(self, name, values):
        (self.directory / name).write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")

    def config(self, variant="step", horizon=150, width=160):
        return {"variant": variant, "train": {"variant": variant, "timesteps": horizon, "velocity_window_s": 2},
                "environment": {"num_envs": 1024, "seed": 0, "terrain_seed": 0,
                                "depth_camera": {"width": width, "height": 120, "policy_shape": [12, 16]}}}

    def test_missing_measurements_stay_missing(self):
        result = summary_tool.summarize_stage(self.root, "pretrain_s1")
        self.assertIsNone(result["latest_report"])
        self.assertIsNone(result["best_report"])
        self.assertIsNone(result["performance"]["median_update_s"])
        self.assertIsNone(result["numerical_checks"]["nonfinite_grad_events"])
        self.assertIsNone(result["numerical_checks"]["max_executed_action_error"])
        self.assertIsNone(result["recent_reports"]["metrics"]["success_rate"]["mean"])

    def test_recent_statistics_use_last_five_and_each_metrics_actual_sample_count(self):
        records = [{"update": i, "success_rate": i / 10, "episode_return_mean": None if i != 7 else 3.0}
                   for i in range(1, 8)]
        recent = summary_tool.recent_statistics(records)
        success = recent["metrics"]["success_rate"]
        self.assertEqual(recent["report_count"], 5)
        self.assertEqual(success["n"], 5)
        self.assertAlmostEqual(success["mean"], 0.5)
        self.assertAlmostEqual(success["std"], math.sqrt(0.025))
        self.assertEqual(recent["metrics"]["episode_return_mean"], {"n": 1, "mean": 3.0, "std": None})

    def test_first_threshold_matches_exact_update_and_stage_step_counters(self):
        self.write("config.json", self.config(horizon=96))
        self.write("training_result.json", {"completed_updates": 110, "stage_updates": 10,
                                           "environment_steps": 123456 + 10 * 96 * 1024})
        self.lines("training_statistics.jsonl", [
            {"update": 104, "success_rate": 0.7}, {"update": 105, "success_rate": 0.8, "episodes": 1200},
            {"update": 106, "success_rate": 0.9},
        ])
        self.lines("updates.jsonl", [{"update": 104, "elapsed_s": 300}, {"update": 105, "elapsed_s": 321}])
        record = summary_tool.summarize_stage(self.root, "pretrain_s1")
        passed = record["first_80_percent"]
        self.assertEqual(passed["update"], 105)
        self.assertEqual(passed["elapsed_s"], 321)
        self.assertEqual(passed["stage_environment_steps"], 5 * 96 * 1024)
        self.assertEqual(passed["environment_steps"], 123456 + 5 * 96 * 1024)
        self.assertEqual(record["best_report"]["update"], 106)
        self.lines("updates.jsonl", [{"update": 104, "elapsed_s": 300}])
        missing = summary_tool.summarize_stage(self.root, "pretrain_s1")["first_80_percent"]
        self.assertIsNone(missing["elapsed_s"])
        self.assertEqual(missing["elapsed_match"], "missing")

    def test_nonfinite_values_action_errors_and_incomplete_lines_are_visible(self):
        self.lines("updates.jsonl", [
            {"update": 1, "update_s": 2, "grad_norm": float("nan"), "executed_action_max_error": 0},
            {"update": 2, "update_s": 4, "executed_action_max_error": 2e-6},
        ])
        self.lines("training_statistics.jsonl", [{"update": 1, "success_rate": 0.2}])
        with (self.directory / "training_statistics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write('{"update": 2, "success_rate":')
        record = summary_tool.summarize_stage(self.root, "pretrain_s1")
        checks = record["numerical_checks"]
        self.assertEqual(checks["nonfinite_update_count"], 1)
        self.assertEqual(checks["nonfinite_updates"][0]["fields"], ["grad_norm"])
        self.assertEqual(checks["action_error_updates_above_1e_6"], 1)
        self.assertEqual(record["performance"]["median_update_s"], 3)
        self.assertEqual(record["report_count"], 1)
        self.assertTrue(any(issue["kind"] == "invalid_jsonl" for issue in record["read_issues"]))

    def test_config_differences_are_reported_and_outputs_do_not_modify_input(self):
        self.write("config.json", self.config())
        second = Path(self.temporary.name) / "fixture_second"
        second_directory = second / "zhang" / "pretrain_s1"
        second_directory.mkdir(parents=True)
        other_config = self.config(variant="continuous", horizon=96, width=320)
        (second_directory / "config.json").write_text(json.dumps(other_config), encoding="utf-8")
        report = summary_tool.summarize_runs([self.root, second])
        comparison = report["comparison"][0]
        self.assertEqual(set(comparison["differing_fields"]), {"variant", "timesteps", "depth_width"})
        self.assertFalse(comparison["control_fields_match"])
        before = (self.directory / "config.json").read_bytes()
        prefix = Path(self.temporary.name) / "reports" / "unit_summary"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(summary_tool.main([str(self.root), str(second), "--output", str(prefix)]), 0)
        json.loads(Path(f"{prefix}.json").read_text(encoding="utf-8"))
        markdown = Path(f"{prefix}.md").read_text(encoding="utf-8")
        self.assertIn("missing", markdown)
        self.assertIn("跨种子稳定性", markdown)
        self.assertEqual((self.directory / "config.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
