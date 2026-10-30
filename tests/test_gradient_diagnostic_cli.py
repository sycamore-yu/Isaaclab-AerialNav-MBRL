"""CLI-only regression checks; these do not simulate or certify training."""

import argparse
import contextlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from isaaclab_navrl import training
from isaaclab_navrl.zhang.cli import parse_gradient_diagnostic_updates
from isaaclab_navrl.zhang.config import ZhangTrainCfg

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"_diagnostic_cli_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


acceptance = load_script("run_acceptance")
app_stub = types.ModuleType("isaaclab.app")
app_stub.add_launcher_args = lambda parser: parser.add_argument("--device", default="cuda:0")
app_stub.launch_simulation = None
with patch.dict(sys.modules, {"isaaclab": types.ModuleType("isaaclab"), "isaaclab.app": app_stub}):
    trainer = load_script("train_zhang")
    navrl_trainer = load_script("train_navrl")


class ParsedArgs(Exception):
    def __init__(self, namespace):
        self.namespace = namespace


def parse_training_cli(tokens, module=trainer):
    """Run the real CLI registration/parse, then stop before any training import."""
    original_parse_args = argparse.ArgumentParser.parse_args

    def capture(parser, *args, **kwargs):
        raise ParsedArgs(original_parse_args(parser, *args, **kwargs))

    with patch.object(argparse.ArgumentParser, "parse_args", capture):
        with patch.object(sys, "argv", ["train_zhang.py", "--log_dir", "/unused/unit-test", *tokens]):
            try:
                module.main()
            except ParsedArgs as result:
                return result.namespace
    raise AssertionError("Training CLI did not parse its arguments")


class GradientDiagnosticCliTests(unittest.TestCase):
    def test_new_defaults_keep_bptt_and_normalization_schedule_controls_explicit(self):
        args = parse_training_cli([])
        self.assertEqual(args.stage, "pretrain_s0")
        self.assertEqual(args.algorithm, "bptt")
        self.assertEqual(args.critic_value_transform, "linear")
        self.assertIsNone(args.velocity_loss_mode)  # Resolved after parsing from the selected algorithm.
        self.assertEqual(args.height_loss_mode, "navrl_range")
        self.assertEqual((args.height_target_min, args.height_target_max), (0.5, 2.5))
        self.assertEqual(args.plateau_patience, 2)
        self.assertEqual((args.optimizer, args.weight_decay), ("adamw", 0.01))
        self.assertEqual(args.gradient_projection, "none")
        self.assertEqual(args.lr_schedule, "cosine")
        self.assertIs(args.observation_normalization, False)
        self.assertIs(args.recovery_enabled, True)
        self.assertEqual(parse_training_cli([], navrl_trainer).stage, "pretrain_s0")
        ahac = parse_training_cli(["--algorithm", "ahac", "--variant", "dcol", "--observation_normalization", "1",
                                  "--recovery_enabled", "0", "--lr_schedule", "constant"])
        self.assertEqual(ahac.algorithm, "ahac")
        self.assertIs(ahac.observation_normalization, True)
        self.assertIs(ahac.recovery_enabled, False)
        self.assertEqual((ahac.critic_lr, ahac.critic_iterations, ahac.critic_max_iterations), (0.0005, 16, 64))
        self.assertEqual((ahac.discount, ahac.td_lambda), (0.99, 0.95))
        self.assertEqual((ahac.ahac_min_horizon, ahac.ahac_initial_horizon, ahac.ahac_max_horizon), (16, 32, 300))
        self.assertEqual((ahac.ahac_risk_clearance, ahac.ahac_risk_scale, ahac.ahac_risk_budget, ahac.ahac_dual_lr),
                         (0.1, 0.05, 0.1, 0.1))

    def test_critic_value_transform_choices_and_bptt_rejection_before_imports(self):
        for algorithm in ("shac", "ahac"):
            for transform in ("linear", "softplus"):
                with self.subTest(algorithm=algorithm, transform=transform):
                    args = parse_training_cli(["--algorithm", algorithm, "--variant", "dcol",
                                               "--critic_value_transform", transform])
                    self.assertEqual(args.critic_value_transform, transform)
        for tokens in (["--critic_value_transform", "softplus"],
                       ["--algorithm", "bptt", "--critic_value_transform", "softplus"],
                       ["--algorithm", "shac", "--variant", "dcol", "--critic_value_transform", "relu"]):
            with self.subTest(tokens=tokens):
                with patch.object(sys, "argv", ["train_zhang.py", "--log_dir", "/unused/unit-test", *tokens]):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        trainer.main()
                    self.assertEqual(error.exception.code, 2)

    def test_velocity_loss_cli_choices_are_explicit_and_reject_unknown_values(self):
        for mode in ("variant_default", "per_step", "truncated_window", "full_window"):
            with self.subTest(mode=mode):
                args = parse_training_cli(["--variant", "dcol", "--velocity_loss_mode", mode])
                self.assertEqual(args.velocity_loss_mode, mode)
                self.assertEqual(args.velocity_window_s, 2.0)
        for raw in ("window", "step", "", "PER_STEP"):
            with self.subTest(raw=raw):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    parse_training_cli(["--velocity_loss_mode", raw])
                self.assertEqual(error.exception.code, 2)

    def test_restart_mode_choices_match_config_and_accept_zero_plateau_patience(self):
        for variant in ("step", "continuous", "dcol"):
            for mode in ("variant_default", "full_window", "truncated_window", "per_step"):
                with self.subTest(variant=variant, mode=mode):
                    args = parse_training_cli(["--variant", variant, "--velocity_loss_mode", mode])
                    cfg = ZhangTrainCfg(variant=args.variant, velocity_loss_mode=args.velocity_loss_mode)
                    cfg.validate()
                    expected = ("per_step" if variant == "step" else "truncated_window") if mode == "variant_default" else (
                        "window" if mode == "full_window" else mode)
                    self.assertEqual(cfg.resolved_velocity_loss_mode, expected)
        for height in ("target_exp", "target_huber", "boundary_huber"):
            for patience in (0, 2):
                for optimizer in ("adam", "adamw"):
                    tokens = ["--height_loss_mode", height, "--height_target_min", "0.6", "--height_target_max", "2.4",
                              "--plateau_patience", str(patience), "--optimizer", optimizer, "--weight_decay", "0",
                              "--gradient_projection", "height_priority"]
                    with self.subTest(height=height, patience=patience, optimizer=optimizer):
                        args = parse_training_cli(tokens)
                        values = acceptance.parse_train_args(tokens)
                        for key, value in values.items():
                            self.assertEqual(getattr(args, key), value)
                        cfg = ZhangTrainCfg(height_loss_mode=args.height_loss_mode,
                                            height_target_range=(args.height_target_min, args.height_target_max),
                                            plateau_patience=args.plateau_patience, optimizer=args.optimizer,
                                            weight_decay=args.weight_decay, gradient_projection=args.gradient_projection)
                        cfg.validate()
        for flag, value in (("height_loss_mode", "huber"), ("optimizer", "sgd"), ("gradient_projection", "pcgrad")):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_training_cli([f"--{flag}", value])

    def test_velocity_loss_reaches_the_training_configuration_without_changing_window(self):
        class CapturedConfiguration(Exception):
            def __init__(self, values):
                self.values = values

        def capture_configuration(**values):
            raise CapturedConfiguration(values)

        capture_configuration.__dataclass_fields__ = ZhangTrainCfg.__dataclass_fields__

        environment = types.SimpleNamespace(scene=types.SimpleNamespace(), sim=types.SimpleNamespace(dt=0.01), decimation=10,
                                            sim_to_real=types.SimpleNamespace(), configure_scene=lambda: None)
        imports = {
            "gymnasium": types.SimpleNamespace(), "torch": types.SimpleNamespace(),
            "isaaclab_navrl.tasks": types.SimpleNamespace(),
            "isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg": types.SimpleNamespace(
                ZhangNavRLEnvCfg=lambda: environment),
            "isaaclab_navrl.tasks.navigation.curriculum": types.SimpleNamespace(STAGE_CONFIG=acceptance.STAGE_CONFIG),
            "isaaclab_navrl.training": types.SimpleNamespace(write_json=None),
            "isaaclab_navrl.zhang.config": types.SimpleNamespace(ZhangTrainCfg=capture_configuration),
            "isaaclab_navrl.zhang.runner": types.SimpleNamespace(ZhangTrainer=None),
        }
        original_import = __import__

        def intercept(name, *args, **kwargs):
            return imports[name] if name in imports else original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory(prefix="velocity_mode_cli_unit_") as temporary:
            for algorithm in ("bptt", "shac", "ahac"):
                for mode in ("variant_default", "per_step", "truncated_window", "full_window"):
                    tokens = ["train_zhang.py", "--log_dir", str(Path(temporary) / "stage"),
                              "--variant", "dcol", "--algorithm", algorithm,
                              "--velocity_loss_mode", mode, "--velocity_window_s", "2"]
                    with self.subTest(algorithm=algorithm, mode=mode):
                        with patch.object(sys, "argv", tokens), patch("builtins.__import__", side_effect=intercept):
                            with self.assertRaises(CapturedConfiguration) as captured:
                                trainer.main()
                        self.assertEqual(captured.exception.values["velocity_loss_mode"], mode)
                        self.assertEqual(captured.exception.values["velocity_window_s"], 2.0)
                        self.assertEqual(captured.exception.values["algorithm"], algorithm)
                        config = ZhangTrainCfg(**captured.exception.values)
                        config.validate()
                        expected = "truncated_window" if mode == "variant_default" else "window" if mode == "full_window" else mode
                        self.assertEqual(config.resolved_velocity_loss_mode, expected)

            for height in ("target_exp", "target_huber", "boundary_huber"):
                for patience in (0, 2):
                    tokens = ["train_zhang.py", "--log_dir", str(Path(temporary) / "stage"),
                              "--variant", "continuous", "--height_loss_mode", height,
                              "--height_target_min", "0.6", "--height_target_max", "2.4",
                              "--optimizer", "adam", "--weight_decay", "0", "--plateau_patience", str(patience),
                              "--gradient_projection", "height_priority"]
                    with self.subTest(height=height, patience=patience):
                        with patch.object(sys, "argv", tokens), patch("builtins.__import__", side_effect=intercept):
                            with self.assertRaises(CapturedConfiguration) as captured:
                                trainer.main()
                        values = captured.exception.values
                        self.assertEqual(values["height_loss_mode"], height)
                        self.assertEqual(values["height_target_range"], (0.6, 2.4))
                        self.assertEqual(environment.height_target_range, (0.6, 2.4))
                        self.assertEqual((values["optimizer"], values["weight_decay"]), ("adam", 0.0))
                        self.assertEqual(values["plateau_patience"], patience)
                        self.assertEqual(values["gradient_projection"], "height_priority")
                        ZhangTrainCfg(**values).validate()

    def test_stop_signals_only_request_a_boundary_stop_and_restore_original_handlers(self):
        first, second = object(), object()
        handlers = {training.signal.SIGINT: first, training.signal.SIGTERM: second}
        requests = []
        stub = types.SimpleNamespace(request_stop=requests.append)

        def install(signum, handler):
            previous = handlers[signum]
            handlers[signum] = handler
            return previous

        with patch.object(training.signal, "signal", side_effect=install):
            with self.assertRaisesRegex(RuntimeError, "training error"):
                with training.graceful_training_stop(stub):
                    handlers[training.signal.SIGINT](training.signal.SIGINT, None)
                    handlers[training.signal.SIGTERM](training.signal.SIGTERM, None)
                    self.assertEqual(requests, [training.signal.SIGINT, training.signal.SIGTERM])
                    raise RuntimeError("training error")
        self.assertIs(handlers[training.signal.SIGINT], first)
        self.assertIs(handlers[training.signal.SIGTERM], second)

    def test_binary_flags_and_removed_variant_are_rejected_before_training(self):
        for flag in ("observation_normalization", "recovery_enabled"):
            for raw in ("true", "false", "2", "-1", "1.0", "", "+1", " 1"):
                with self.subTest(flag=flag, raw=raw):
                    with self.assertRaises(ValueError):
                        acceptance.parse_train_args([f"--{flag}={raw}"])
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        parse_training_cli([f"--{flag}={raw}"])
                    self.assertEqual(error.exception.code, 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_training_cli(["--variant", "unknown_variant"])

    def test_only_s0_can_be_fresh_in_unlimited_training_for_both_trainers(self):
        cases = (["--stage", "pretrain_s0", "--checkpoint", "/unit/old.pt"],
                 ["--stage", "pretrain_s1"], ["--stage", "finetune_s2"])
        for module in (trainer, navrl_trainer):
            for tokens in cases:
                with self.subTest(module=module.__name__, tokens=tokens):
                    with patch.object(sys, "argv", ["train.py", "--log_dir", "/unused/unit-test", *tokens]):
                        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                            module.main()
                    self.assertEqual(error.exception.code, 2)

    def test_finite_later_stage_smoke_is_allowed_before_training_imports(self):
        class ReachedTrainingImports(Exception):
            pass

        original_import = __import__

        def intercept(name, *args, **kwargs):
            if name == "gymnasium":
                raise ReachedTrainingImports
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory(prefix="stage_smoke_cli_unit_") as temporary:
            for index, module in enumerate((trainer, navrl_trainer)):
                tokens = ["train.py", "--log_dir", str(Path(temporary) / str(index)),
                          "--stage", "pretrain_s1", "--max_iterations", "1"]
                with patch.object(sys, "argv", tokens), patch("builtins.__import__", side_effect=intercept):
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ReachedTrainingImports):
                        module.main()

    def test_training_default_and_explicit_empty_disable_diagnostics(self):
        self.assertEqual(parse_training_cli([]).gradient_diagnostic_updates, ())
        self.assertEqual(parse_training_cli(["--gradient_diagnostic_updates", ""]).gradient_diagnostic_updates, ())
        self.assertEqual(acceptance.parse_train_args([]), {})
        self.assertEqual(acceptance.parse_train_args(["--gradient_diagnostic_updates="]),
                         {"gradient_diagnostic_updates": ()})

    def test_positive_updates_keep_order_and_both_entry_points_agree(self):
        for raw, expected in (("20,22,24,26", (20, 22, 24, 26)), ("26,20", (26, 20)), ("1", (1,))):
            with self.subTest(raw=raw):
                self.assertEqual(parse_training_cli(["--gradient_diagnostic_updates", raw]).gradient_diagnostic_updates,
                                 expected)
                self.assertEqual(acceptance.parse_train_args(["--gradient_diagnostic_updates", raw]),
                                 {"gradient_diagnostic_updates": expected})
                self.assertEqual(acceptance.parse_train_args([f"--gradient_diagnostic_updates={raw}"]),
                                 {"gradient_diagnostic_updates": expected})

    def test_invalid_lists_are_rejected_by_both_parsers(self):
        invalid = ("0", "-1", "1,0", "1,-2", ",1", "1,", "1,,2", "1.0", "1e2",
                   "one", "nan", "1,one", "1,1", "01,1", " 1", "1, 2", "+1", "١")
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_gradient_diagnostic_updates(raw)
                with self.assertRaises(ValueError):
                    acceptance.parse_train_args([f"--gradient_diagnostic_updates={raw}"])
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    parse_training_cli([f"--gradient_diagnostic_updates={raw}"])
                self.assertEqual(error.exception.code, 2)

    def test_duplicate_option_and_original_extra_arg_restrictions_remain(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            acceptance.parse_train_args(["--gradient_diagnostic_updates=20", "--gradient_diagnostic_updates=22"])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            acceptance.parse_train_args(["--max_iterations=20"])
        with self.assertRaisesRegex(ValueError, "Missing value"):
            acceptance.parse_train_args(["--gradient_diagnostic_updates"])

    def test_camera_batch_default_and_positive_integer_validation(self):
        self.assertEqual(parse_training_cli([]).camera_batch_size, 64)
        self.assertEqual(parse_training_cli(["--camera_batch_size", "64"]).camera_batch_size, 64)
        self.assertEqual(acceptance.parse_train_args(["--camera_batch_size", "64"]), {"camera_batch_size": 64})
        for raw in ("0", "-1", "1.5", "nan", ""):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    acceptance.parse_train_args([f"--camera_batch_size={raw}"])
                # Run validation after parsing too; 0/-1 must fail before any
                # simulator or training dependency can be imported.
                with patch.object(sys, "argv", ["train_zhang.py", "--log_dir", "/unused/unit-test",
                                                 f"--camera_batch_size={raw}"]):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        trainer.main()
                self.assertEqual(error.exception.code, 2)

    def test_commands_serialize_update_tuple_for_every_stage(self):
        extra = acceptance.parse_train_args(["--gradient_diagnostic_updates", "20,22,24,26", "--lr", "0.001",
                                            "--camera_batch_size", "64"])
        parent = None
        for stage in acceptance.STAGES:
            with self.subTest(stage=stage):
                command = acceptance.build_command(
                    method="zhang", stage=stage, log_dir=Path("/unit/run/zhang") / stage,
                    scripts_dir=Path("/unit/scripts"), num_envs=1024, seed=0, device="cuda:0",
                    finetune_lr=5e-5, train_args=extra, parent_checkpoint=parent, executable="python",
                )
                value = command[command.index("--gradient_diagnostic_updates") + 1]
                self.assertEqual(value, "20,22,24,26")
                self.assertEqual(parse_gradient_diagnostic_updates(value), (20, 22, 24, 26))
                self.assertEqual(command[command.index("--camera_batch_size") + 1], "64")
                self.assertEqual(command[command.index("--lr") + 1], "5e-05" if stage.startswith("finetune_") else "0.001")
                self.assertEqual("--checkpoint" in command, parent is not None)
                if parent is not None:
                    self.assertEqual(command[command.index("--checkpoint") + 1], str(parent))
                parent = Path("/unit/run/zhang") / stage / "checkpoint_stage.pt"


if __name__ == "__main__":
    unittest.main()
