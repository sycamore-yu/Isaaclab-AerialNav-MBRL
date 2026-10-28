"""Acceptance tuning cannot silently change the task through argparse aliases."""

import importlib.util
from pathlib import Path

import pytest


def protocol_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_acceptance.py"
    spec = importlib.util.spec_from_file_location("run_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_approved_zhang_hyperparameters_are_allowed():
    protocol_module().validate_train_args("zhang", [
        "--lr", "0.001", "--max_grad_norm=3", "--timesteps", "96", "--grad_decay", "0.4",
        "--coef_obj_avoidance", "3", "--coef_collide", "7.5", "--velocity_window_s", "6",
        "--coef_d_acc", "0.01", "--coef_d_jerk", "0.001",
    ])


@pytest.mark.parametrize("option", ["--static=0", "--dynamic=0", "--physics=newton", "--num_e=1"])
def test_task_and_budget_overrides_cannot_use_argparse_abbreviations(option):
    with pytest.raises(ValueError):
        protocol_module().validate_train_args("zhang", [option])
