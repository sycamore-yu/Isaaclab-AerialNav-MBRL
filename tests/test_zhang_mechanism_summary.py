"""Stdlib-only summary fixtures. These generated numbers are NOT training results."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_zhang_mechanisms.py"
SPEC = importlib.util.spec_from_file_location("zhang_mechanism_summary", SCRIPT)
summary_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary_tool)


class MechanismSummaryFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="zhang_mechanism_unit_fixture_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "runs"
        self.root.mkdir()
        self.manifest = Path(self.temporary.name) / "fixture_manifest.json"
        self.entries = []

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def lines(self, path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")

    def run_fixture(self, name, horizons=(32, 16, 20), *, algorithm="ahac", seed=0, num_envs=10,
                    base_steps=0, update_offset=0, stage="pretrain_s0", relative=None):
        relative = relative or name
        entry = {"name": name, "path": relative, "variant": "dcol", "algorithm": algorithm,
                 "normalization": False, "lr_schedule": "plateau", "seed": seed}
        self.entries.append(entry)
        root = self.root / relative
        directory = root / "zhang" / stage
        self.write(root / "training_result.json", {"status": "running", "fixture_only": True})
        self.write(directory / "config.json", {
            "method": "zhang", "train": {"variant": "dcol", "algorithm": algorithm, "timesteps": 150,
                                           "observation_normalization": False, "lr_schedule": "plateau", "seed": seed},
            "environment": {"num_envs": num_envs, "seed": seed, "stage": stage},
        })
        total, elapsed, records = base_steps, 100.0, []
        for index, horizon in enumerate(horizons, 1):
            total += horizon * num_envs
            duration = index + 1.0
            elapsed += duration
            records.append({"update": update_offset + index, "rollout_steps": horizon,
                            "environment_steps": total, "elapsed_s": elapsed, "update_s": duration,
                            "surrogate_velocity_error_mps": 0.01 * index,
                            "surrogate_vertical_velocity_bias_mps": -0.02 * index,
                            "executed_action_max_error": 0.0, "Optim/update_ratio": index / 100,
                            "Optim/head/az/update_ratio": index / 10, "Optim/head/vz/update_ratio": index / 20})
        self.lines(directory / "updates.jsonl", records)
        self.lines(directory / "training_statistics.jsonl", [])
        self.write(directory / "training_result.json", {
            "status": "running", "fresh": base_steps == 0, "stage": stage,
            "completed_updates": update_offset + len(horizons), "stage_updates": len(horizons),
            "environment_steps": total, "elapsed_s": elapsed, "num_envs": num_envs,
            "nonfinite_grad_events": 0, "nonfinite_events": 0,
        })
        return directory, records

    def report(self, update, success, episodes=10, stage="pretrain_s0"):
        return {"update": update, "stage": stage, "episodes": episodes, "success_rate": success,
                "collision_rate": (1 - success) / 2, "bounds_rate": (1 - success) / 4,
                "invalid_rate": 0.0, "timeout_rate": (1 - success) / 4}

    def summary(self):
        self.write(self.manifest, {"fixture_only": True, "runs": self.entries})
        return summary_tool.summarize(self.root, self.manifest)

    def stage(self, result, name, stage="pretrain_s0"):
        run = next(row for row in result["runs"] if row["name"] == name)
        return next(row for row in run["stages"] if row["stage"] == stage)

    def test_variable_horizon_uses_real_samples_and_exact_first_eighty_percent(self):
        directory, _ = self.run_fixture("adaptive", num_envs=1024, base_steps=100000, update_offset=100)
        self.lines(directory / "training_statistics.jsonl", [
            self.report(101, 0.6, 1024), self.report(102, 0.8, 2048), self.report(103, 0.5, 1024),
        ])
        stage = self.stage(self.summary(), "adaptive")
        summary = stage["summary"]
        self.assertEqual(summary["stage_environment_steps"], (32 + 16 + 20) * 1024)
        self.assertEqual(summary["environment_steps"], 100000 + 68 * 1024)
        self.assertEqual(summary["stage_elapsed_s"], 9)
        first = summary["first_natural_80_percent"]
        self.assertEqual(first["update"], 102)
        self.assertEqual(first["stage_environment_steps"], 48 * 1024)
        self.assertEqual(first["elapsed_s"], 105)
        self.assertEqual(first["stage_elapsed_s"], 5)
        self.assertEqual(summary["best_report"]["update"], 102)
        self.assertEqual(summary["latest_report"]["update"], 103)
        self.assertAlmostEqual(summary["recent_five"]["rates"]["success_rate"], 0.675)
        self.assertEqual(summary["recent_five"]["episodes"], 4096)
        self.assertAlmostEqual(summary["fc_z_head_max_ratio"], 0.3)
        self.assertAlmostEqual(summary["numerical_metrics"]["Optim/update_ratio"]["max"], 0.03)
        self.assertAlmostEqual(summary["numerical_metrics"]["surrogate_vertical_velocity_bias_mps"]["max_abs"], 0.06)
        self.assertIn("H=16", stage["notes"][0])
        self.assertTrue(Path(first["source"]["path"]).is_absolute())

    def test_ppo_keeps_actual_step_budget_and_original_config_identity(self):
        directory, _ = self.run_fixture("navrl_ppo", horizons=(32, 32), algorithm="ppo")
        destination = directory.parent.parent / "navrl" / directory.name
        destination.parent.mkdir()
        directory.rename(destination)
        entry = self.entries[-1]
        entry.update(method="navrl", variant="navrl", lr_schedule="fixed")
        (destination / "config.json").unlink()
        self.write(destination / "run_config.json", {
            "method": "navrl", "seed": 0, "num_envs": 10,
            "training": {"algorithm": {"schedule": "fixed"}},
        })
        result = self.stage(self.summary(), "navrl_ppo")
        self.assertEqual(result["identity_mismatches"], {})
        self.assertEqual(result["identity_missing_fields"], [])
        self.assertEqual(result["summary"]["stage_environment_steps"], 640)
        self.assertEqual(result["observed_identity"]["algorithm"], "ppo")

    def test_common_stage_budget_selects_whole_updates_without_interpolation(self):
        fixed, _ = self.run_fixture("fixed", horizons=(50, 50), algorithm="shac")
        adaptive, _ = self.run_fixture("adaptive", horizons=(32, 16, 16))
        self.lines(fixed / "training_statistics.jsonl", [self.report(1, 0.4), self.report(2, 0.95)])
        self.lines(adaptive / "training_statistics.jsonl", [self.report(1, 0.3), self.report(3, 0.7)])
        result = self.summary()
        comparison = next(row for row in result["comparisons"] if row["stage"] == "pretrain_s0")
        self.assertEqual(comparison["budget_stage_environment_steps"], 640)
        selected = {row["name"]: row["summary"] for row in comparison["participants"]}
        self.assertEqual(selected["fixed"]["stage_environment_steps"], 500)
        self.assertEqual(selected["adaptive"]["stage_environment_steps"], 640)
        self.assertEqual(selected["fixed"]["latest_report"]["success_rate"], 0.4)
        self.assertIsNone(selected["fixed"]["first_natural_80_percent"])
        self.assertEqual(self.stage(result, "fixed")["summary"]["first_natural_80_percent"]["success_rate"], 0.95)
        self.assertIsNone(selected["fixed"]["result_nonfinite_events"])

    def test_whitelist_recursive_discovery_and_explicit_exclusions(self):
        self.run_fixture("kept", relative="nested/kept")
        self.run_fixture("smoke_check")
        self.run_fixture("externaloldrun")
        self.run_fixture("old_unlisted")
        self.entries.pop()
        external = {**self.entries[0], "name": "outside", "path": str(Path(self.temporary.name) / "outside")}
        self.entries.append(external)
        result = self.summary()
        self.assertEqual([run["name"] for run in result["runs"]], ["kept"])
        reasons = {Path(row["path"]).name: row["reason"] for row in result["excluded_runs"]}
        self.assertEqual(reasons["smoke_check"], "smoke_or_externaloldrun")
        self.assertEqual(reasons["externaloldrun"], "smoke_or_externaloldrun")
        self.assertEqual(reasons["old_unlisted"], "not_in_manifest")
        self.assertEqual(reasons["outside"], "outside_root")

    def test_missing_files_and_half_written_lines_remain_visible(self):
        directory, records = self.run_fixture("partial", horizons=(16,))
        records[0]["grad_norm"] = float("nan")
        self.lines(directory / "updates.jsonl", records)
        with (directory / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write('{"update": 2, "rollout_steps":')
        self.lines(directory / "training_statistics.jsonl", [self.report(1, 0.25)])
        with (directory / "training_statistics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write('{"update": 2')
        (directory / "training_result.json").unlink()
        result = self.summary()
        stage = self.stage(result, "partial")
        self.assertEqual(stage["data_status"], "partial")
        self.assertEqual(stage["sources"]["updates"]["status"], "partial")
        self.assertEqual(stage["sources"]["result"]["status"], "missing")
        self.assertEqual(stage["summary"]["latest_report"]["success_rate"], 0.25)
        self.assertEqual(stage["summary"]["nonfinite_log_record_count"], 1)
        self.assertIsNone(stage["summary"]["result_nonfinite_events"])
        future = self.stage(result, "partial", "finetune_s2")
        self.assertEqual(future["data_status"], "missing")
        self.assertIsNone(future["summary"]["recent_five"]["rates"]["success_rate"])
        self.assertIsNone(future["summary"]["fc_z_head_max_ratio"])
        json.dumps(result, allow_nan=False)

    def test_report_without_exact_update_does_not_invent_timing(self):
        directory, records = self.run_fixture("gap", horizons=(1, 1, 1))
        self.lines(directory / "updates.jsonl", [records[0], records[2]])
        self.lines(directory / "training_statistics.jsonl", [self.report(2, 0.8)])
        stage = self.stage(self.summary(), "gap")
        first = stage["summary"]["first_natural_80_percent"]
        self.assertIsNone(first["elapsed_s"])
        self.assertIsNone(first["stage_environment_steps"])
        self.assertEqual(stage["summary"]["stage_environment_steps"], 30)
        self.assertTrue(any(issue["kind"] == "report_without_exact_update" for issue in stage["read_validation_issues"]))

    def test_last_five_are_episode_weighted_and_missing_metric_is_not_zero(self):
        reports = [{**self.report(index, index / 10, episodes=index * 10), "source": {"line": index}}
                   for index in range(1, 8)]
        reports[-1].pop("collision_rate")
        recent = summary_tool.recent_reports(reports)
        expected = sum((index / 10) * index * 10 for index in range(3, 8)) / sum(index * 10 for index in range(3, 8))
        self.assertEqual(recent["report_count"], 5)
        self.assertAlmostEqual(recent["rates"]["success_rate"], expected)
        self.assertIsNone(recent["rates"]["collision_rate"])
        self.assertEqual(recent["metric_coverage"]["collision_rate"]["reports"], 4)

    def test_different_seeds_and_metadata_mismatch_do_not_get_common_budget(self):
        first, _ = self.run_fixture("first", seed=0)
        self.run_fixture("second", seed=0)
        self.run_fixture("third", seed=1)
        config = json.loads((first / "config.json").read_text())
        config["train"]["normalization"] = True  # Unrelated field must not alter the contract.
        config["train"]["observation_normalization"] = True
        self.write(first / "config.json", config)
        result = self.summary()
        comparisons = [row for row in result["comparisons"] if row["stage"] == "pretrain_s0"]
        self.assertEqual(len(comparisons), 2)
        for comparison in comparisons:
            self.assertIsNone(comparison["budget_stage_environment_steps"])
        pair = next(row for row in comparisons if row["seed"] == 0)
        self.assertEqual(pair["missing_or_mismatched_runs"], ["first"])
        self.assertEqual({row["name"] for row in pair["participants"]}, {"first", "second"})

    def test_missing_actual_horizon_never_falls_back_to_configured_h150(self):
        directory, records = self.run_fixture("unknown", horizons=(16, 32), base_steps=9000, stage="pretrain_s1")
        for record in records:
            record.pop("rollout_steps")
        self.lines(directory / "updates.jsonl", records)
        stage = self.stage(self.summary(), "unknown", "pretrain_s1")
        self.assertIsNone(stage["summary"]["stage_environment_steps"])
        self.assertEqual(stage["summary"]["environment_steps"], 9480)
        self.assertIsNone(stage["summary"]["rollout_steps_max"])
        self.assertEqual(stage["data_status"], "partial")

    def test_missing_start_of_later_stage_is_not_treated_as_a_complete_stage(self):
        directory, records = self.run_fixture("truncated_prefix", horizons=(16, 32, 16),
                                              base_steps=9000, update_offset=100, stage="pretrain_s1")
        self.lines(directory / "updates.jsonl", records[1:])
        stage = self.stage(self.summary(), "truncated_prefix", "pretrain_s1")
        self.assertEqual(stage["summary"]["environment_steps"], 9640)
        self.assertIsNone(stage["summary"]["stage_environment_steps"])
        self.assertIsNone(stage["summary"]["stage_elapsed_s"])
        self.assertTrue(any(issue["kind"] == "missing_stage_start" for issue in stage["read_validation_issues"]))

    def test_cli_writes_strict_json_without_mutating_fixture_sources(self):
        directory, _ = self.run_fixture("cli", horizons=(16,))
        self.summary()
        source = directory / "updates.jsonl"
        before = source.read_bytes()
        output = Path(self.temporary.name) / "report" / "fixture_summary.json"
        arguments = ["--root", str(self.root), "--manifest", str(self.manifest), "--output", str(output)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(summary_tool.main(arguments), 0)
        self.assertEqual(source.read_bytes(), before)
        result = json.loads(output.read_text())
        self.assertEqual(result["runs"][0]["name"], "cli")
        arguments[-1] = str(source)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            summary_tool.main(arguments)
        self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
