"""Queue admission and fresh command contracts without launching simulation."""

import copy
import json
import os
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
    assert values == {
        key: (tuple(int(v) for v in value.split(",")) if value else ())
        if key == "gradient_diagnostic_updates"
        else value
        for key, value in run["train_args"].items()
    }
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
    assert allow(2, True, 20000, 3584)
    assert allow(20, True, 20000, 3584)
    assert not allow(20, True, 4000, 3584, reserve_mib=768)
    assert allow(6, True, 4550, 3584, reserve_mib=768)
    assert not allow(0, True, 7167, 5120)
    assert allow(5, True, 6144, 4096, slots=6)
    assert not allow(6, True, 20000, 4096, slots=6)
    assert not allow(5, True, 6143, 4096, slots=6)


def test_external_training_source_is_used_in_child_command(tmp_path):
    training = tmp_path / "frozen-training"
    command = QUEUE["command_for"](training, tmp_path / "new-run", PLAN, PLAN["runs"][0], "cuda:0")
    assert command[1] == str(training / "scripts/run_acceptance.py")
    assert not any(str(ROOT) in token for token in command[1:])


def test_ready_tracks_latest_stage_and_rejects_partial_or_initial_result(tmp_path):
    job = {"output": str(tmp_path), "method": "zhang"}
    save = QUEUE["write_json"]
    parent = {
        "status": "running",
        "methods": {
            "zhang": {
                "stages": [
                    {"status": "completed", "stage": "pretrain_s0"},
                    {"status": "running", "stage": "pretrain_s1"},
                ]
            }
        },
    }
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


def test_main_matrix_preserves_requested_controls_and_twenty_hz_priority(tmp_path):
    plan = json.loads((ROOT / "configs/experiments/zhang_main_comparison.json").read_text())
    runs = {run["name"]: run for run in plan["runs"]}
    assert set(runs) == {
        "unified_20hz",
        "dcol",
        "step",
        "truncated",
        "plateau",
        "shac",
        "ahac",
        "navrl_ppo",
        "continuous",
    }
    assert plan["launch_order"][0] == "unified_20hz"
    assert plan["report_episodes"] == 1024 and plan["max_updates"] is None
    for name, run in runs.items():
        cmd = QUEUE["command_for"](ROOT, tmp_path / name, plan, run, "cuda:0")
        assert cmd[cmd.index("--report_episodes") + 1] == "1024"
        args = PROTOCOL["parse_train_args"](QUEUE["train_tokens"](run))
        if run["method"] == "zhang":
            assert args["height_loss_mode"] == "target_exp"
            assert not args["observation_normalization"]
    baseline = runs["dcol"]["train_args"]
    for name, expected in (("step", "per_step"), ("truncated", "truncated_window")):
        assert runs[name]["train_args"] == {**baseline, "velocity_loss_mode": expected}
    assert runs["unified_20hz"]["train_args"] == {**baseline, "timing": "unified_20hz"}
    assert baseline["velocity_loss_mode"] == "full_window" and baseline["velocity_window_s"] == 2.0
    assert runs["plateau"]["train_args"] == {**baseline, "lr_schedule": "plateau"}
    assert runs["ahac"]["train_args"]["ahac_max_horizon"] == 300
    assert plan["launch_order"].index("plateau") < plan["launch_order"].index("navrl_ppo")


@pytest.mark.parametrize("dirty,branch,head", [(True, "main", "pinned"), (False, "main", "changed")])
def test_queue_rejects_dirty_or_changed_source(monkeypatch, dirty, branch, head):
    def git(args, **kwargs):
        return {"rev-parse": head, "status": " M changed.py" if dirty else "", "branch": branch}[args[1]]

    monkeypatch.setattr(QUEUE["subprocess"], "check_output", git)
    with pytest.raises(RuntimeError):
        QUEUE["check_source"](ROOT, "pinned")


def test_takeover_preserves_training_arguments_and_source_identity(monkeypatch):
    adopt = QUEUE["adopt_running_jobs"]
    previous = {
        "queue_pid": 100,
        "source_commit": "original-training",
        "runs": {
            "active": {"command": ["python", "pipeline.py"], "pid": 101, "status": "running"},
            "pending": {"command": ["python", "other.py"], "pid": None, "status": "queued"},
        },
    }
    jobs = {name: {**entry, "estimated_mib": 3584} for name, entry in previous["runs"].items()}

    def ticks(pid, command=None):
        return None if pid == 100 else 9876

    monkeypatch.setitem(adopt.__globals__, "process_start_ticks", ticks)
    assert adopt(previous, jobs) == {"active": 101}
    assert jobs["active"]["process_start_ticks"] == 9876
    assert jobs["active"]["source_commit"] == "original-training"
    assert jobs["pending"]["status"] == "queued"
    changed = copy.deepcopy(jobs)
    changed["pending"]["command"].append("--changed")
    with pytest.raises(ValueError, match="training arguments"):
        adopt(previous, changed)
    monkeypatch.setitem(adopt.__globals__, "process_start_ticks", lambda *args: 123)
    with pytest.raises(ValueError, match="Previous dispatcher"):
        adopt(previous, jobs)


def test_process_identity_rejects_command_mismatch_and_missing_pid():
    ticks = QUEUE["process_start_ticks"]
    assert ticks(os.getpid()) > 0
    with pytest.raises(ValueError, match="recorded command"):
        ticks(os.getpid(), ["unrelated", "process"])
    assert ticks(999999999) is None


def test_twenty_hz_successor_plan_preserves_baseline_and_five_new_controls(tmp_path):
    plan = json.loads((ROOT / "configs/experiments/zhang_20hz_controls.json").read_text())
    cases = {row["name"]: row for row in plan["runs"]}
    assert set(cases) == {"hz50", "shac", "ahac", "plateau", "navrl_height"}
    assert plan["retained_baseline"].endswith("zhang_main_20261029/unified_20hz")
    assert plan["num_envs"] == 256 and plan["report_episodes"] == 1024
    for name, row in cases.items():
        values = PROTOCOL["parse_train_args"](QUEUE["train_tokens"](row))
        assert values["variant"] == "dcol"
        assert values["timing"] == ("unified_50hz" if name == "hz50" else "unified_20hz")
        assert values["height_loss_mode"] == ("navrl_range" if name == "navrl_height" else "target_exp")
        assert values["coef_height_boundary"] == 1
        assert values["velocity_loss_mode"] == ("per_step" if name in ("shac", "ahac") else "full_window")


def test_queue_accepts_clean_pinned_feature_branch(monkeypatch):
    monkeypatch.setattr(
        QUEUE["subprocess"],
        "check_output",
        lambda args, **kw: {
            "rev-parse": "pinned",
            "status": "",
            "branch": "feature",
        }[args[1]],
    )
    QUEUE["check_source"](ROOT, "pinned")
