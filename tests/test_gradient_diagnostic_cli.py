"""CLI-only regression checks; these do not simulate or certify training."""

import argparse
import contextlib
import importlib.util
import io
import sys
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


class ParsedArgs(Exception):
    def __init__(self, namespace):
        self.namespace = namespace


def parse_training_cli(tokens):
    """Run the real CLI registration/parse, then stop before any training import."""
    original_parse_args = argparse.ArgumentParser.parse_args

    def capture(parser, *args, **kwargs):
        raise ParsedArgs(original_parse_args(parser, *args, **kwargs))

    with patch.object(argparse.ArgumentParser, "parse_args", capture):
        with patch.object(sys, "argv", ["train_zhang.py", "--log_dir", "/unused/unit-test", *tokens]):
            try:
                trainer.main()
            except ParsedArgs as result:
                return result.namespace
    raise AssertionError("Training CLI did not parse its arguments")


class GradientDiagnosticCliTests(unittest.TestCase):
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
