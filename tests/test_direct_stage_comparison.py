"""Direct starts preserve fresh lineage without certifying skipped curriculum stages."""

import argparse
import json
import runpy
import subprocess
from pathlib import Path

import pytest
from test_acceptance_protocol import protocol, report_fixture

ROOT = Path(__file__).resolve().parents[1]
QUEUE = runpy.run_path(str(ROOT / "scripts/run_comparison_queue.py"))


@pytest.mark.parametrize("start", ["pretrain_s1", "finetune_s2"])
def test_direct_pipeline_starts_random_then_inherits_only_its_own_checkpoints(tmp_path, monkeypatch, start):
    monkeypatch.setattr(protocol, "freeze_source", lambda source, target: {"root": str(target), "commit": "fixture"})
    monkeypatch.setattr(protocol, "verify_source", lambda *args: None)
    calls = []

    def simulate(command, **kwargs):
        stage = command[command.index("--stage") + 1]
        path = Path(command[command.index("--log_dir") + 1])
        parent = Path(command[command.index("--checkpoint") + 1]) if "--checkpoint" in command else None
        if not calls:
            assert stage == start and parent is None and "--fresh_stage" in command
            assert float(command[command.index("--lr") + 1]) == .001
        else:
            assert parent == calls[-1][1] / "checkpoint_stage.pt"
            assert "--fresh_stage" not in command
            if stage.startswith("finetune"):
                assert float(command[command.index("--lr") + 1]) == 5e-5
        assert "--max_iterations" not in command
        result = report_fixture(stage, parent, num_envs=256)
        (path / "training_result.json").write_text(json.dumps(result))
        (path / "checkpoint_stage.pt").write_bytes(b"Synthetic orchestration fixture")
        calls.append((stage, path))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(protocol.subprocess, "run", simulate)
    output = tmp_path / "fresh"
    args = argparse.Namespace(method="zhang", screening=True, output=str(output), num_envs=256,
                              train_seed=0, device="cuda:0", finetune_lr=5e-5, start_stage=start)
    assert protocol.run_pipeline(args, {"lr": .001, "height_loss_mode": "navrl_range"}) == 0
    summary = json.loads((output / "training_result.json").read_text())
    expected = list(protocol.selected_stages(start, screening=True))
    assert [stage for stage, _ in calls] == summary["stage_order"] == expected
    assert summary["skipped_stages"] == list(protocol.STAGES[:protocol.STAGES.index(start)])
    assert summary["stage_thresholds_passed"] is True
    assert summary["acceptance_passed"] is summary["complete_curriculum_passed"] is False


@pytest.mark.parametrize("method,screening", [("zhang", False), ("navrl", True), ("both", True)])
def test_direct_starts_cannot_bypass_formal_or_ppo_protocol(method, screening):
    with pytest.raises(ValueError, match="screening --method zhang"):
        protocol.selected_stages("finetune_s2", screening=screening, method=method)


def test_fresh_report_requires_declared_direct_start_and_screening():
    result = report_fixture("pretrain_s1", num_envs=256)
    options = dict(method="zhang", stage="pretrain_s1", num_envs=256, parent_checkpoint=None)
    assert protocol.validate_stage_result(result, **options, screening=True)
    assert not protocol.validate_stage_result(result, **options, screening=True, fresh_start_stage="pretrain_s1")
    assert protocol.validate_stage_result(result, **options, fresh_start_stage="pretrain_s1")
    assert protocol.validate_stage_result(result, **{**options, "stage": "pretrain_s2"},
                                          screening=True, fresh_start_stage="pretrain_s1")


def test_new_plan_changes_only_start_stage_and_keeps_navrl_default(tmp_path):
    plan = json.loads((ROOT / "configs/experiments/zhang_navrl_direct_starts.json").read_text())
    assert plan["num_envs"] == 256 and plan["report_episodes"] == 1024 and plan["max_updates"] is None
    assert [r["start_stage"] for r in plan["runs"]] == ["pretrain_s1", "finetune_s2"]
    assert plan["runs"][0]["train_args"] == plan["runs"][1]["train_args"]
    for run in plan["runs"]:
        command = QUEUE["command_for"](ROOT, tmp_path / run["name"], plan, run, "cuda:0")
        assert command[command.index("--start_stage") + 1] == run["start_stage"]
        args = protocol.parse_train_args(QUEUE["train_tokens"](run))
        assert args["height_loss_mode"] == "navrl_range" and args["coef_height_boundary"] == 1
        assert args["timing"] == "unified_20hz" and args["lr"] == .001
        assert args["algorithm"] == "bptt" and args["timesteps"] == 150
        assert args["velocity_loss_mode"] == "full_window" and args["velocity_window_s"] == 2
        assert "--checkpoint" not in command
