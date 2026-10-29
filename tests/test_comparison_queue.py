"""Queue admission and fresh command contracts without launching simulation."""

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUEUE = runpy.run_path(str(ROOT / "scripts/run_comparison_queue.py"))
PROTOCOL = runpy.run_path(str(ROOT / "scripts/run_acceptance.py"))
PLAN = json.loads((ROOT / "configs/experiments/zhang_restart_controls_20261029.json").read_text())


@pytest.mark.parametrize("run", PLAN["runs"], ids=lambda run: run["name"])
def test_all_planned_commands_preserve_fresh_unlimited_protocol(run, tmp_path):
    output = tmp_path / run["path"]
    command = QUEUE["command_for"](ROOT, output, PLAN, run, "cuda:0")
    assert not output.exists()
    assert command[command.index("--method") + 1] == run["method"]
    assert command[command.index("--num_envs") + 1] == "256"
    assert "--screening" in command
    assert not any(token in command for token in ("--checkpoint", "--resume", "--max_iterations"))
    tokens = QUEUE["train_tokens"](run)
    values = PROTOCOL["parse_train_args"](tokens)
    assert values == {key: (tuple(int(v) for v in value.split(",")) if value else ())
                      if key == "gradient_diagnostic_updates" else value
                      for key, value in run["train_args"].items()}
    if run["method"] == "navrl":
        assert "--train_args" not in command
    else:
        assert values["recovery_enabled"] is True
        assert values["observation_normalization"] is run["train_args"]["observation_normalization"]


def test_admission_requires_update_capacity_and_slot():
    allow = QUEUE["admissible"]
    assert allow(0, True, 9703, 4096)
    assert allow(1, True, 5632, 3584)
    assert not allow(1, True, 5631, 3584)
    assert not allow(1, False, 20000, 3584)
    assert not allow(2, True, 20000, 3584)
    assert not allow(0, True, 7167, 5120)
    assert allow(5, True, 6144, 4096, slots=6)
    assert not allow(6, True, 20000, 4096, slots=6)
    assert not allow(5, True, 6143, 4096, slots=6)


def test_external_training_source_is_used_in_child_command(tmp_path):
    training = tmp_path / "frozen-training"
    command = QUEUE["command_for"](training, tmp_path / "new-run", PLAN, PLAN["runs"][0], "cuda:0")
    assert command[1] == str(training / "scripts/run_acceptance.py")
    assert not any(str(ROOT) in token for token in command)


def test_ready_tracks_latest_stage_and_rejects_partial_or_initial_result(tmp_path):
    job = {"output": str(tmp_path), "method": "zhang"}
    save = QUEUE["atomic_json"]
    parent = {"status": "running", "methods": {"zhang": {"stages": [
        {"status": "completed", "stage": "pretrain_s0"},
        {"status": "running", "stage": "pretrain_s1"},
    ]}}}
    save(tmp_path / "training_result.json", parent)
    old = tmp_path / "zhang/pretrain_s0"
    old.mkdir(parents=True)
    save(old / "training_result.json", {"status": "running", "stage": "pretrain_s0", "stage_updates": 100})
    assert not QUEUE["current_stage_ready"](job)
    current = tmp_path / "zhang/pretrain_s1"
    current.mkdir()
    path = current / "training_result.json"
    for updates in (0, True, None):
        save(path, {"status": "running", "stage": "pretrain_s1", "stage_updates": updates})
        assert not QUEUE["current_stage_ready"](job)
    save(path, {"status": "running", "stage": "pretrain_s1", "stage_updates": 1})
    assert QUEUE["current_stage_ready"](job)
    path.write_text('{"stage_updates":')
    assert not QUEUE["current_stage_ready"](job)


def test_comparison_identity_excludes_queued_and_ppo():
    jobs = {run["name"]: {**run, "pid": None} for run in PLAN["runs"]}
    assert QUEUE["launched_zhang_manifest"](jobs, PLAN)["runs"] == []
    jobs["dcol_truncated"]["pid"] = 123
    jobs["navrl_ppo"]["pid"] = 124
    manifest = QUEUE["launched_zhang_manifest"](jobs, PLAN)
    assert len(manifest["runs"]) == 1
    entry = manifest["runs"][0]
    assert entry["name"] == "dcol_truncated"
    assert entry["normalization"] is False and entry["num_envs"] == 256
    assert entry["velocity_loss_mode"] == "truncated_window"
    assert entry["height_loss_mode"] == "target_exp"
    assert entry["optimizer"] == "adamw" and entry["weight_decay"] == 0.01
