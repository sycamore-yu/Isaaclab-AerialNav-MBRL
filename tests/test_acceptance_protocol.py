"""Unit tests for command construction and report validation, not training results."""

import argparse
import contextlib
import copy
import importlib.util
import io
import json
import subprocess
import sys
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


def report_fixture(stage="pretrain_s0", parent=None, algorithm="bptt", variant="dcol",
                   velocity_loss_mode=None, num_envs=1024):
    if velocity_loss_mode is None:
        velocity_loss_mode = "per_step" if algorithm == "shac" else "full_window"
    config = STAGE_CONFIG[stage].copy()
    final = stage == "finetune_s2"
    return {
        "method": "zhang",
        "variant": variant,
        "algorithm": algorithm,
        "velocity_loss_mode": velocity_loss_mode,
        "resolved_velocity_loss_mode": (("per_step" if variant == "step" else "truncated_window")
                                        if velocity_loss_mode == "variant_default" else
                                        "window" if velocity_loss_mode == "full_window" else velocity_loss_mode),
        "status": "completed",
        "stage": stage,
        "num_envs": num_envs,
        "stage_passed": True,
        "max_updates": None,
        "parent_checkpoint": str(parent) if parent is not None else None,
        "fresh": parent is None,
        "acceptance_passed": final and num_envs >= 1024,
        "curriculum": {
            "stage": stage,
            "stage_passed": True,
            "stage_config": config,
            "active_dynamic_obstacles": config["dynamic_obstacles"],
            "promotion_success_rate": 0.8,
            "episodes_per_report_min": num_envs,
            "acceptance_passed": final,
            "latest_training_report": {"stage": stage, **config, "episodes": num_envs, "success_rate": 0.8},
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

    def test_new_height_optimizer_and_projection_options_are_preserved_in_every_stage(self):
        for height in ("target_exp", "target_huber", "boundary_huber"):
            for patience in (0, 2):
                for optimizer in ("adam", "adamw"):
                    values = parse_train_args([
                        "--variant", "dcol", "--velocity_loss_mode", "full_window",
                        "--height_loss_mode", height, "--height_target_min", "0.6", "--height_target_max", "2.4",
                        "--optimizer", optimizer, "--weight_decay", "0", "--plateau_patience", str(patience),
                        "--gradient_projection", "height_priority",
                    ])
                    for stage in STAGES:
                        with self.subTest(height=height, patience=patience, optimizer=optimizer, stage=stage):
                            command = self.command(stage=stage, train_args=values)
                            for key, value in values.items():
                                self.assertEqual(command.count(f"--{key}"), 1)
                                self.assertEqual(command[command.index(f"--{key}") + 1], str(value))
                            navrl = self.command(method="navrl", stage=stage, train_args=values)
                            for key in values:
                                self.assertNotIn(f"--{key}", navrl)

    def test_new_joint_constraints_and_ahac_upper_default_match_train_configuration(self):
        values = parse_train_args(["--algorithm", "ahac", "--variant", "dcol", "--ahac_initial_horizon", "200"])
        self.assertEqual(values["ahac_initial_horizon"], 200)
        invalid = (
            ["--height_target_min", "2.5", "--height_target_max", "0.5"],
            ["--height_target_min", "0.2"], ["--height_target_max", "4.0"],
            ["--height_target_min", "0.5", "--height_target_max", "0.5"],
            ["--height_loss_mode", "boundary"], ["--optimizer", "sgd"],
            ["--weight_decay", "-0.1"], ["--plateau_patience", "-1"], ["--plateau_patience", "1.5"],
            ["--gradient_projection", "pcgrad"],
            ["--algorithm", "shac", "--variant", "dcol", "--gradient_projection", "height_priority"],
        )
        for tokens in invalid:
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                parse_train_args(tokens)

    def test_velocity_loss_mode_is_preserved_across_stages_for_all_algorithms(self):
        self.assertEqual(parse_train_args([]), {})
        for algorithm in ("bptt", "shac", "ahac"):
            for mode in ("variant_default", "per_step", "truncated_window", "full_window"):
                values = parse_train_args(["--variant", "dcol", "--algorithm", algorithm,
                                           "--velocity_loss_mode", mode, "--velocity_window_s", "2"])
                for stage in STAGES:
                    with self.subTest(algorithm=algorithm, mode=mode, stage=stage):
                        command = self.command(stage=stage, train_args=values)
                        self.assertEqual(command.count("--velocity_loss_mode"), 1)
                        self.assertEqual(command[command.index("--velocity_loss_mode") + 1], mode)
                        self.assertEqual(command[command.index("--velocity_window_s") + 1], "2.0")
                        self.assertNotIn("--velocity_loss_mode",
                                         self.command(method="navrl", stage=stage, train_args=values))
        for raw in ("window", "step", "truncated", "", "PER_STEP", "TRUNCATED_WINDOW"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_train_args(["--velocity_loss_mode", raw])

    def test_velocity_mode_result_identity_is_checked_for_every_stage_and_algorithm(self):
        for algorithm in ("bptt", "shac", "ahac"):
            for mode, resolved in (("variant_default", "truncated_window"), ("per_step", "per_step"),
                                   ("truncated_window", "truncated_window"), ("full_window", "window")):
                for index, stage in enumerate(STAGES):
                    parent = Path(f"/runs/new/zhang/{STAGES[index - 1]}/checkpoint_stage.pt") if index else None
                    with self.subTest(algorithm=algorithm, mode=mode, stage=stage):
                        result = report_fixture(stage, parent, algorithm, "dcol", mode)
                        self.assertEqual(result["resolved_velocity_loss_mode"], resolved)
                        options = dict(method="zhang", stage=stage, num_envs=1024,
                                       parent_checkpoint=parent, algorithm=algorithm, variant="dcol",
                                       velocity_loss_mode=mode)
                        self.assertEqual(validate_stage_result(result, **options), [])
                        result["velocity_loss_mode"] = "per_step" if mode != "per_step" else "truncated_window"
                        self.assertTrue(validate_stage_result(result, **options))

    def test_velocity_mode_result_cannot_be_missing_or_report_a_different_resolution(self):
        original = report_fixture(variant="dcol", velocity_loss_mode="truncated_window")
        options = dict(method="zhang", stage="pretrain_s0", num_envs=1024,
                       parent_checkpoint=None, variant="dcol", velocity_loss_mode="truncated_window")
        for field in ("velocity_loss_mode", "resolved_velocity_loss_mode"):
            for replacement in (None, "window", "per_step", "variant_default"):
                with self.subTest(field=field, replacement=replacement):
                    result = copy.deepcopy(original)
                    if replacement is None:
                        result.pop(field)
                    else:
                        result[field] = replacement
                    self.assertTrue(validate_stage_result(result, **options))
        # Equal resolved behavior does not permit changing the experiment's explicit mode.
        result = report_fixture(velocity_loss_mode="per_step")
        self.assertTrue(self.validate(result))

    def test_critic_transform_requires_ac_and_is_preserved_across_all_stages(self):
        self.assertEqual(parse_train_args(["--critic_value_transform", "linear"]),
                         {"critic_value_transform": "linear"})
        for algorithm in ("shac", "ahac"):
            for transform in ("linear", "softplus"):
                values = parse_train_args(["--algorithm", algorithm, "--variant", "dcol",
                                           "--critic_value_transform", transform])
                for stage in STAGES:
                    with self.subTest(algorithm=algorithm, transform=transform, stage=stage):
                        command = self.command(stage=stage, train_args=values)
                        self.assertEqual(command.count("--critic_value_transform"), 1)
                        self.assertEqual(command[command.index("--critic_value_transform") + 1], transform)
                        self.assertNotIn("--critic_value_transform",
                                         self.command(method="navrl", stage=stage, train_args=values))
        for tokens in (["--critic_value_transform", "softplus"],
                       ["--algorithm", "bptt", "--critic_value_transform", "softplus"],
                       ["--algorithm", "shac", "--variant", "dcol", "--critic_value_transform", "relu"]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                parse_train_args(tokens)

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
            ["--algorithm", "shac", "--variant", "step"], ["--algorithm", "shac", "--variant", "dcol",
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

    def test_default_and_full_window_resolution_match_each_current_variant(self):
        for variant in ("step", "continuous", "dcol"):
            for mode in ("variant_default", "full_window"):
                resolved = "window" if mode == "full_window" else "per_step" if variant == "step" else "truncated_window"
                result = report_fixture(variant=variant, velocity_loss_mode=mode)
                with self.subTest(variant=variant, mode=mode):
                    self.assertEqual(result["resolved_velocity_loss_mode"], resolved)
                    options = dict(method="zhang", stage="pretrain_s0", num_envs=1024, parent_checkpoint=None,
                                   variant=variant, velocity_loss_mode=mode)
                    self.assertEqual(validate_stage_result(result, **options), [])
                    result["resolved_velocity_loss_mode"] = "window" if resolved != "window" else "truncated_window"
                    self.assertTrue(validate_stage_result(result, **options))

    def test_screening_requires_explicit_opt_in_and_never_relaxes_scale_or_stage_checks(self):
        for index, stage in enumerate(STAGES):
            parent = Path(f"/runs/new/zhang/{STAGES[index - 1]}/checkpoint_stage.pt") if index else None
            result = report_fixture(stage, parent, variant="dcol", num_envs=128)
            options = dict(method="zhang", stage=stage, num_envs=128, parent_checkpoint=parent, variant="dcol")
            with self.subTest(stage=stage):
                self.assertTrue(validate_stage_result(result, **options))
                self.assertEqual(validate_stage_result(result, **options, screening=True), [])
                for keys, value in ((("num_envs",), 64), (("max_updates",), 2),
                                    (("curriculum", "episodes_per_report_min"), 64),
                                    (("curriculum", "latest_training_report", "episodes"), 127),
                                    (("curriculum", "latest_training_report", "success_rate"), 0.79)):
                    invalid = copy.deepcopy(result)
                    target = invalid
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = value
                    with self.subTest(keys=keys):
                        self.assertTrue(validate_stage_result(invalid, **options, screening=True))
                if stage == "finetune_s2":
                    self.assertFalse(result["acceptance_passed"])
                    self.assertTrue(result["curriculum"]["acceptance_passed"])
                    result["acceptance_passed"] = True
                    self.assertTrue(validate_stage_result(result, **options, screening=True))

    def test_screening_cli_scale_gate_runs_before_any_pipeline(self):
        for count, screening, valid in ((128, False, False), (0, True, False),
                                        (128, True, True), (1024, False, True), (1024, True, True)):
            with self.subTest(count=count, screening=screening), tempfile.TemporaryDirectory() as temporary:
                tokens = ["run_acceptance.py", "--output", str(Path(temporary) / "new"), "--num_envs", str(count)]
                if screening:
                    tokens.append("--screening")
                with patch.object(sys, "argv", tokens), patch.object(protocol, "run_pipeline", return_value=0) as pipeline:
                    if valid:
                        self.assertEqual(protocol.main(), 0)
                        self.assertEqual(pipeline.call_args.args[0].screening, screening)
                    else:
                        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                            protocol.main()
                        pipeline.assert_not_called()

    def test_all_five_stage_screening_completes_without_becoming_formal_acceptance(self):
        # Synthetic subprocess files verify orchestration, never policy quality.
        for count, screening in ((128, True), (1024, False), (1024, True)):
            visited = []

            def unit_process(command, *, expected_count=count, visits=visited, **kwargs):
                stage = command[command.index("--stage") + 1]
                method = Path(command[1]).stem.removeprefix("train_")
                log_dir = Path(command[command.index("--log_dir") + 1])
                parent = Path(command[command.index("--checkpoint") + 1]) if "--checkpoint" in command else None
                self.assertEqual(int(command[command.index("--num_envs") + 1]), expected_count)
                result = report_fixture(stage, parent, variant="dcol", num_envs=expected_count)
                result["method"] = method
                (log_dir / "training_result.json").write_text(json.dumps(result), encoding="utf-8")
                (log_dir / "checkpoint_stage.pt").write_bytes(b"Synthetic protocol fixture, not a trained model")
                visits.append((method, stage))
                return subprocess.CompletedProcess(command, 0)

            with self.subTest(count=count, screening=screening), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "new"
                args = argparse.Namespace(method="both", output=str(output), num_envs=count, screening=screening,
                                          train_seed=0, device="cuda:0", finetune_lr=5e-5)
                with patch.object(protocol.subprocess, "run", side_effect=unit_process), patch("builtins.print"):
                    self.assertEqual(run_pipeline(args, {"variant": "dcol"}), 0)
                self.assertEqual(visited, [(method, stage) for method in ("zhang", "navrl") for stage in STAGES])
                summary = json.loads((output / "training_result.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["purpose"], "training_comparison" if screening else "formal_acceptance")
                self.assertEqual(summary["screening"], screening)
                self.assertIsNone(summary["error"])
                for result in (summary, *summary["methods"].values()):
                    self.assertEqual(result["status"], "completed")
                    self.assertTrue(result["stage_thresholds_passed"])
                    self.assertEqual(result["acceptance_passed"], not screening)

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
            self.assertEqual(command[command.index("--velocity_loss_mode") + 1], "truncated_window")
            (log_dir / "training_result.json").write_text(
                json.dumps(report_fixture(stage, parent, "shac", "dcol", "truncated_window")), encoding="utf-8")
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
                                                     "lr_schedule": "cosine", "recovery_enabled": True,
                                                     "velocity_loss_mode": "truncated_window"}), 0)
            self.assertEqual(visited, list(STAGES))
            summary = json.loads((output / "training_result.json").read_text(encoding="utf-8"))
            for record in summary["methods"]["zhang"]["stages"]:
                self.assertEqual(Path(record["stdout_log"]), output / "zhang" / f"{record['stage']}.stdout.log")

    def test_child_stop_preserves_checkpoint_without_advancing_or_hiding_failures(self):
        # Fake subprocess output tests orchestration; these are not training results.
        cases = (("stopped", 0, True, None, "stopped"),
                 ("stopped", 1, True, None, "failed"),
                 ("stopped", 0, False, None, "failed"),
                 ("stopped", 0, True, "child training failure", "failed"),
                 ("completed", 0, True, None, "failed"))
        for status, returncode, write_checkpoint, child_error, expected in cases:
            with self.subTest(status=status, returncode=returncode,
                              write_checkpoint=write_checkpoint, child_error=child_error):
                visited = []

                def unit_process(command, *, status=status, child_error=child_error,
                                 write_checkpoint=write_checkpoint, visited=visited, returncode=returncode, **kwargs):
                    stage = command[command.index("--stage") + 1]
                    log_dir = Path(command[command.index("--log_dir") + 1])
                    checkpoint = log_dir / "checkpoint_stopped.pt"
                    result = report_fixture(stage)
                    result.update(status=status, checkpoint=str(checkpoint), stage_passed=False,
                                  acceptance_passed=False, error=child_error)
                    result["curriculum"]["stage_passed"] = False
                    result["curriculum"]["latest_training_report"]["success_rate"] = 0.12
                    (log_dir / "training_result.json").write_text(json.dumps(result), encoding="utf-8")
                    if write_checkpoint:
                        checkpoint.write_bytes(b"Unit-test stopped checkpoint fixture, not a trained model")
                    visited.append(stage)
                    return subprocess.CompletedProcess(command, returncode)

                with tempfile.TemporaryDirectory(prefix="acceptance_stop_unit_") as temporary:
                    output = Path(temporary) / "unit_run"
                    args = argparse.Namespace(method="both", output=str(output), num_envs=1024,
                                              train_seed=0, device="cuda:0", finetune_lr=5e-5)
                    with patch.object(protocol.subprocess, "run", side_effect=unit_process), \
                            patch("builtins.print"):
                        exit_code = run_pipeline(args, {})
                    self.assertEqual(exit_code, 0 if expected == "stopped" else 1)
                    summary = json.loads((output / "training_result.json").read_text(encoding="utf-8"))
                    method = summary["methods"]["zhang"]
                    self.assertEqual(visited, [STAGES[0]])
                    self.assertEqual(list(summary["methods"]), ["zhang"])
                    self.assertEqual(len(method["stages"]), 1)
                    stage = method["stages"][0]
                    self.assertFalse(stage["result"]["stage_passed"])
                    for record in (summary, method, stage):
                        self.assertEqual(record["status"], expected)
                    self.assertFalse(summary["acceptance_passed"])
                    self.assertFalse(method["acceptance_passed"])
                    if expected == "stopped":
                        checkpoint = output / "zhang" / STAGES[0] / "checkpoint_stopped.pt"
                        for record in (summary, method, stage):
                            self.assertIsNone(record["error"])
                            self.assertFalse(record["acceptance_passed"])
                            self.assertEqual(Path(record["checkpoint"]), checkpoint)
                        self.assertFalse(stage["stage_passed"])
                        self.assertEqual(stage["validation_errors"], [])
                        self.assertEqual(Path(stage["result"]["checkpoint"]), checkpoint)
                        self.assertTrue(checkpoint.is_file())
                        self.assertFalse((checkpoint.parent / "checkpoint_stage.pt").exists())
                    else:
                        self.assertTrue(summary["error"])


if __name__ == "__main__":
    unittest.main()
