"""Default and explicit-stage entry points resolve before any simulator starts."""

import json
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class BeforeSimulation(Exception):
    pass


@pytest.mark.parametrize(
    "tokens,stage,count,randomization,fresh,lr",
    [
        ([], "finetune_s1", 32, True, True, 0.001),
        (["--stage", "pretrain_s1", "--fresh_stage"], "pretrain_s1", 16, False, True, 0.001),
        (["--stage", "pretrain_s0"], "pretrain_s0", 0, False, False, 0.001),
        (["--checkpoint", "/retained/model.pt"], "finetune_s1", 32, True, False, 0.00005),
    ],
)
def test_training_defaults_and_explicit_stages(tokens, stage, count, randomization, fresh, lr, tmp_path, monkeypatch):
    module = runpy.run_path(str(ROOT / "scripts/train_zhang.py"))
    entry = module["main"]
    monkeypatch.setitem(entry.__globals__, "add_launcher_args", lambda parser: parser.add_argument("--device"))

    def stop(cfg, args):
        expected_static = {"pretrain_s0": 240, "pretrain_s1": 360, "finetune_s1": 480}[stage]
        assert (cfg.training_stage, cfg.num_static_obstacles, cfg.num_dynamic_obstacles) == (
            stage,
            expected_static,
            count,
        )
        assert cfg.sim_to_real.enabled is randomization
        assert cfg.training_report_episodes == 1024
        assert args.fresh_stage is fresh and args.lr == lr
        raise BeforeSimulation

    monkeypatch.setitem(entry.__globals__, "launch_simulation", stop)
    monkeypatch.setattr(sys, "argv", ["train_zhang.py", "--log_dir", str(tmp_path / "run"), "--device", "cpu", *tokens])
    with pytest.raises(BeforeSimulation):
        entry()
    resolved = json.loads((tmp_path / "run/config.json").read_text())
    assert resolved["environment"]["dynamic_obstacles"] == count
    assert resolved["direct_stage_start"] is fresh


def test_default_queue_selects_the_four_rectangular_controls(monkeypatch):
    entry = runpy.run_path(str(ROOT / "scripts/run_comparison_queue.py"))["main"]

    def parse(parser):
        args = parser.parse_args(["--commit", "fixture"])
        plan = json.loads(args.plan.read_text())
        assert plan["launch_order"] == [
            "baseline_direct",
            "baseline_curriculum",
            "reshaped_direct",
            "reshaped_curriculum",
        ]
        assert plan["num_envs"] == plan["report_episodes"] == 1024
        assert plan["continue_final_stage"] is True
        assert [row["fixed_stage"] for row in plan["runs"]] == [True, False, True, False]
        raise BeforeSimulation

    monkeypatch.setitem(entry.__globals__, "parse_physics_args", parse)
    with pytest.raises(BeforeSimulation):
        entry()
