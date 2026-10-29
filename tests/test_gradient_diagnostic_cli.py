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
        self.assertEqual((ahac.ahac_min_horizon, ahac.ahac_initial_horizon, ahac.ahac_max_horizon), (16, 32, 150))
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

    def test_stop_signals_only_request_a_boundary_stop_and_restore_original_handlers(self):
        first, second = object(), object()
        handlers = {trainer.signal.SIGINT: first, trainer.signal.SIGTERM: second}
        requests = []
        stub = types.SimpleNamespace(request_stop=requests.append)

        def install(signum, handler):
            previous = handlers[signum]
            handlers[signum] = handler
            return previous

        with patch.object(trainer.signal, "signal", side_effect=install):
            with self.assertRaisesRegex(RuntimeError, "training error"):
                with trainer.graceful_training_stop(stub):
                    handlers[trainer.signal.SIGINT](trainer.signal.SIGINT, None)
                    handlers[trainer.signal.SIGTERM](trainer.signal.SIGTERM, None)
                    self.assertEqual(requests, [trainer.signal.SIGINT, trainer.signal.SIGTERM])
                    raise RuntimeError("training error")
        self.assertIs(handlers[trainer.signal.SIGINT], first)
        self.assertIs(handlers[trainer.signal.SIGTERM], second)

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
            parse_training_cli(["--variant", "truncated_window"])

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
                    trainer.parse_gradient_diagnostic_updates(raw)
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
        self.assertEqual(parse_training_cli([]).camera_batch_size, 16)
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
                self.assertEqual(trainer.parse_gradient_diagnostic_updates(value), (20, 22, 24, 26))
                self.assertEqual(command[command.index("--camera_batch_size") + 1], "64")
                self.assertEqual(command[command.index("--lr") + 1], "5e-05" if stage.startswith("finetune_") else "0.001")
                self.assertEqual("--checkpoint" in command, parent is not None)
                if parent is not None:
                    self.assertEqual(command[command.index("--checkpoint") + 1], str(parent))
                parent = Path("/unit/run/zhang") / stage / "checkpoint_stage.pt"


if __name__ == "__main__":
    unittest.main()
