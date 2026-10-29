"""Acceptance tuning cannot silently change the task through argparse aliases."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("method", ["navrl", "zhang"])
def test_acceptance_runs_training_only_without_a_budget(method, tmp_path):
    command = protocol_module().training_command(method, tmp_path / "fresh", 0, "cuda:0", [])
    assert command[1].endswith(f"train_{method}.py")
    assert "--max_iterations" not in command
    assert "--checkpoint" not in command
    assert "--eval_seed" not in command
    assert command[command.index("--num_envs") + 1] == "1024"
    assert command[command.index("--static_obstacles") + 1] == "350"


@pytest.mark.parametrize("navrl_passes", [True, False])
def test_two_methods_start_fresh_in_order_and_require_training_acceptance(tmp_path, monkeypatch, navrl_passes):
    protocol = protocol_module()
    output = tmp_path / "fresh"
    monkeypatch.setattr(protocol.sys, "argv", [
        "run_acceptance.py", "--method", "both", "--output", str(output),
        "--train_args", "--timesteps", "150",
    ])
    calls = []

    def train(command, env, check):
        calls.append(command)
        run_dir = Path(command[command.index("--log_dir") + 1])
        assert not run_dir.exists()
        run_dir.mkdir(parents=True)
        passed = navrl_passes if len(calls) == 1 else True
        (run_dir / "training_result.json").write_text(json.dumps({
            "status": "completed", "acceptance_passed": passed,
        }))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(protocol.subprocess, "run", train)
    if navrl_passes:
        protocol.main()
        assert [Path(command[1]).stem for command in calls] == ["train_navrl", "train_zhang"]
        assert "--timesteps" not in calls[0]
        assert calls[1][calls[1].index("--timesteps") + 1] == "150"
    else:
        with pytest.raises(RuntimeError, match="before acceptance"):
            protocol.main()
        assert len(calls) == 1
    assert all("--stop_after_acceptance" in command for command in calls)
    assert all("--checkpoint" not in command and "--max_iterations" not in command for command in calls)
