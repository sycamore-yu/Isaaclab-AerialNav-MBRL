"""Unit tests for command construction and report validation, not training results."""

import argparse
import copy
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "run_acceptance", Path(__file__).resolve().parents[1] / "scripts" / "run_acceptance.py"
)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
STAGE_CONFIG, STAGES = protocol.STAGE_CONFIG, protocol.STAGES
build_command, parse_train_args = protocol.build_command, protocol.parse_train_args
run_pipeline, validate_stage_result = protocol.run_pipeline, protocol.validate_stage_result


def report_fixture(stage="pretrain_s0", parent=None, algorithm="bptt", variant="step"):
    config = STAGE_CONFIG[stage].copy()
    final = stage == "finetune_s2"
    return {
        "method": "zhang",
        "variant": variant,
        "algorithm": algorithm,
        "status": "completed",
        "stage": stage,
        "num_envs": 1024,
        "stage_passed": True,
        "max_updates": None,
        "parent_checkpoint": str(parent) if parent is not None else None,
        "fresh": parent is None,
        "acceptance_passed": final,
        "curriculum": {
            "stage": stage,
            "stage_passed": True,
            "stage_config": config,
            "active_dynamic_obstacles": config["dynamic_obstacles"],
            "promotion_success_rate": 0.8,
            "episodes_per_report_min": 1024,
            "acceptance_passed": final,
            "latest_training_report": {"stage": stage, **config, "episodes": 1024, "success_rate": 0.8},
        },
    }


class AcceptanceProtocolTests(unittest.TestCase):
    def command(self, **overrides):
        arguments = {
            "method": "zhang",
            "stage": "pretrain_s0",
            "log_dir": Path("/runs/new/zhang/pretrain_s0"),
            "scripts_dir": Path("/repo/scripts"),
            "num_envs": 1024,
            "seed": 7,
            "device": "cuda:0",
            "finetune_lr": 5e-5,
            "train_args": {"timesteps": 150, "lr": 1e-3},
            "parent_checkpoint": None,
            "executable": "/repo/.venv/bin/python",
        }
        arguments.update(overrides)
        return build_command(**arguments)

    def validate(self, value, stage="pretrain_s0", parent=None):
        return validate_stage_result(value, method="zhang", stage=stage, num_envs=1024, parent_checkpoint=parent)

    def test_requested_stages(self):
        self.assertEqual(
            [(STAGE_CONFIG[s]["static_obstacles"], STAGE_CONFIG[s]["dynamic_obstacles"], STAGE_CONFIG[s]["randomization"]) for s in STAGES],
            [(300, 0, False), (300, 60, False), (400, 120, False), (300, 60, True), (400, 100, True)],
        )

    def test_allowed_parameters(self):
        self.assertEqual(
            parse_train_args(["--timesteps", "150", "--lr=0.001", "--coef_d_jerk", "0"]),
            {"timesteps": 150, "lr": 0.001, "coef_d_jerk": 0.0},
        )

    def test_variant_and_boundary_arguments_remain_fixed_across_stages(self):
        values = parse_train_args(["--variant", "dcol", "--coef_height_boundary", "1",
                                   "--boundary_margin", "0.5", "--boundary_scale", "0.5"])
        for stage in STAGES:
            command = self.command(stage=stage, train_args=values)
            self.assertEqual(command[command.index("--variant") + 1], "dcol")
            self.assertEqual(command[command.index("--coef_height_boundary") + 1], "1.0")
        with self.assertRaises(ValueError):
            parse_train_args(["--variant", "unknown"])
        with self.assertRaises(ValueError):
            parse_train_args(["--variant", "unknown_variant"])

    def test_algorithm_stability_controls_and_explicit_boolean_values_are_preserved(self):
        values = parse_train_args([
            "--algorithm", "ahac", "--variant", "dcol", "--observation_normalization", "0", "--lr_schedule", "plateau",
            "--recovery_enabled", "1", "--critic_lr", "0.0005", "--critic_iterations", "16",
            "--critic_max_iterations", "64", "--discount", "0.99", "--td_lambda", "0.95",
            "--ahac_min_horizon", "16", "--ahac_initial_horizon", "32", "--ahac_max_horizon", "150",
            "--ahac_risk_clearance", "0.1", "--ahac_risk_scale", "0.05",
            "--ahac_risk_budget", "0.1", "--ahac_dual_lr", "0.1",
        ])
        self.assertIs(values["observation_normalization"], False)
        self.assertIs(values["recovery_enabled"], True)
        for stage in STAGES:
            command = self.command(stage=stage, train_args=values)
            for key, expected in (("algorithm", "ahac"), ("observation_normalization", "0"),
                                  ("lr_schedule", "plateau"), ("recovery_enabled", "1"),
                                  ("ahac_initial_horizon", "32"), ("ahac_risk_scale", "0.05")):
                self.assertEqual(command[command.index(f"--{key}") + 1], expected)

    def test_reject_process_overrides_and_bad_values(self):
        cases = [
            ["--checkpoint", "/old/run.pt"], ["--max_updates", "2000"], ["--stage", "finetune_s2"],
            ["--num_envs", "128"], ["--timesteps"], ["--timesteps", "0"], ["--lr", "nan"],
            ["--lr", "inf"], ["--coef_d_acc", "-1"], ["--lr", "1e-3", "--lr", "1e-4"],
            ["--algorithm", "ppo"], ["--lr_schedule", "both"], ["--discount", "1"],
            ["--discount", "-1"], ["--td_lambda", "1.1"], ["--critic_iterations", "0"],
            ["--observation_normalization", "true"], ["--recovery_enabled", "false"],
            ["--observation_normalization", "2"], ["--recovery_enabled", ""],
            ["--algorithm", "shac"], ["--algorithm", "shac", "--variant", "dcol",
                                         "--gradient_diagnostic_updates", "12"],
            ["--ahac_min_horizon", "33"], ["--ahac_max_horizon", "16"], ["--timesteps", "1"],
            ["--ahac_risk_clearance", "-0.1"], ["--ahac_dual_lr", "0"],
        ]
        for tokens in cases:
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                parse_train_args(tokens)

    def test_fresh_initial_stage_has_no_checkpoint_or_training_limit(self):
        command = self.command()
        self.assertEqual(command[:2], ["/repo/.venv/bin/python", "/repo/scripts/train_zhang.py"])
        self.assertNotIn("--checkpoint", command)
        self.assertNotIn("--max_updates", command)
        self.assertIn("--stop_after_acceptance", command)
        self.assertEqual(command[command.index("--seed") + 1], "7")
        self.assertEqual(command[command.index("--terrain_seed") + 1], "7")

    def test_finetune_uses_internal_parent_and_single_finetune_lr(self):
        parent = Path("/runs/new/zhang/pretrain_s2/checkpoint_stage.pt")
        command = self.command(stage="finetune_s1", parent_checkpoint=parent)
        self.assertEqual(command[command.index("--checkpoint") + 1], str(parent))
        self.assertEqual(command.count("--lr"), 1)
        self.assertEqual(float(command[command.index("--lr") + 1]), 5e-5)
        self.assertEqual(command[command.index("--timesteps") + 1], "150")

    def test_zhang_parameters_do_not_enter_navrl_command(self):
        command = self.command(method="navrl", stage="finetune_s1")
        self.assertNotIn("--timesteps", command)
        self.assertEqual(command.count("--lr"), 1)
        self.assertEqual(float(command[command.index("--lr") + 1]), 5e-5)

    def test_exact_threshold_passes_each_stage(self):
        for index, stage in enumerate(STAGES):
            parent = Path(f"/runs/new/zhang/{STAGES[index - 1]}/checkpoint_stage.pt") if index else None
            with self.subTest(stage=stage):
                self.assertEqual(self.validate(report_fixture(stage, parent), stage, parent), [])

    def test_rejects_wrong_stage_counts_randomization_and_statistics(self):
        fixture = report_fixture()
        mutations = [
            (("stage",), "finetune_s2"), (("stage_passed",), False), (("status",), "running"),
            (("variant",), "continuous"),
            (("algorithm",), "shac"),
            (("num_envs",), 128), (("max_updates",), 2000), (("parent_checkpoint",), "/old/run.pt"),
            (("curriculum", "stage_config", "static_obstacles"), 350),
            (("curriculum", "active_dynamic_obstacles"), 100),
            (("curriculum", "latest_training_report", "randomization"), True),
            (("curriculum", "latest_training_report", "success_rate"), 0.79),
            (("curriculum", "latest_training_report", "success_rate"), float("nan")),
            (("curriculum", "latest_training_report", "episodes"), 1023),
            (("curriculum", "latest_training_report", "stage"), "finetune_s2"),
        ]
        for keys, value in mutations:
            result = copy.deepcopy(fixture)
            target = result
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            with self.subTest(keys=keys, value=value):
                self.assertTrue(self.validate(result))

    def test_algorithm_match_and_formal_stage_lineage_are_required(self):
        value = report_fixture(algorithm="shac", variant="dcol")
        self.assertTrue(self.validate(value))
        self.assertEqual(validate_stage_result(value, method="zhang", stage="pretrain_s0", num_envs=1024,
                                               parent_checkpoint=None, algorithm="shac", variant="dcol"), [])
        self.assertTrue(self.validate(report_fixture("pretrain_s1"), stage="pretrain_s1"))
        parent = Path("/old/run.pt")
        self.assertTrue(self.validate(report_fixture(parent=parent), parent=parent))

    def test_final_stage_requires_both_acceptance_flags(self):
        parent = Path("/runs/new/zhang/finetune_s1/checkpoint_stage.pt")
        for source in ("trainer", "curriculum"):
            result = report_fixture("finetune_s2", parent)
            target = result if source == "trainer" else result["curriculum"]
            target["acceptance_passed"] = False
            with self.subTest(source=source):
                self.assertTrue(self.validate(result, "finetune_s2", parent))

    def test_stage_directory_is_empty_when_trainer_starts(self):
        # These temporary reports exercise orchestration only; no trainer is launched.
        visited = []

        def unit_process(command, **kwargs):
            stage = command[command.index("--stage") + 1]
            log_dir = Path(command[command.index("--log_dir") + 1])
            parent = Path(command[command.index("--checkpoint") + 1]) if "--checkpoint" in command else None
            self.assertEqual(list(log_dir.iterdir()), [])
            self.assertEqual(Path(kwargs["stdout"].name), log_dir.parent / f"{stage}.stdout.log")
            self.assertEqual(command[command.index("--algorithm") + 1], "shac")
            self.assertEqual(command[command.index("--observation_normalization") + 1], "0")
            self.assertEqual(command[command.index("--recovery_enabled") + 1], "1")
            (log_dir / "training_result.json").write_text(json.dumps(report_fixture(stage, parent, "shac", "dcol")), encoding="utf-8")
            (log_dir / "checkpoint_stage.pt").write_bytes(b"Unit-test checkpoint fixture")
            visited.append(stage)
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory(prefix="acceptance_protocol_unit_") as temporary:
            output = Path(temporary) / "unit_run"
            args = argparse.Namespace(
                method="zhang", output=str(output), num_envs=1024, train_seed=0,
                device="cuda:0", finetune_lr=5e-5,
            )
            with patch.object(protocol.subprocess, "run", side_effect=unit_process), patch("builtins.print"):
                self.assertEqual(run_pipeline(args, {"algorithm": "shac", "variant": "dcol", "observation_normalization": False,
                                                     "lr_schedule": "cosine", "recovery_enabled": True}), 0)
            self.assertEqual(visited, list(STAGES))
            summary = json.loads((output / "training_result.json").read_text(encoding="utf-8"))
            for record in summary["methods"]["zhang"]["stages"]:
                self.assertEqual(Path(record["stdout_log"]), output / "zhang" / f"{record['stage']}.stdout.log")


if __name__ == "__main__":
    unittest.main()
