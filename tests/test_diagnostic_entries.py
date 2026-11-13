"""Diagnostic commands use the same checked Newton/Warp runtime as training."""

import contextlib
import importlib.util
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from isaaclab_navrl.zhang.config import ZhangTrainCfg

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_entry(monkeypatch, name):
    app = types.ModuleType("isaaclab.app")
    app.add_launcher_args = lambda parser: parser.add_argument("--device", default="cpu")
    app.launch_simulation = lambda *_args: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "isaaclab.app", app)
    spec = importlib.util.spec_from_file_location(f"_runtime_entry_{name}", SCRIPTS / f"{name}.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


@pytest.mark.parametrize("name", ["benchmark_camera", "profile_step", "diagnose_zhang"])
def test_diagnostics_construct_and_identify_the_checked_runtime(monkeypatch, tmp_path, name):
    entry = load_entry(monkeypatch, name)
    cfg = types.SimpleNamespace(scene=types.SimpleNamespace(), sim=types.SimpleNamespace())
    cfg.configure_scene = Mock()
    configs = types.ModuleType("isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg")
    configs.NavRLEnvCfg = configs.ZhangNavRLEnvCfg = lambda: cfg
    monkeypatch.setitem(sys.modules, configs.__name__, configs)
    monkeypatch.setitem(sys.modules, "isaaclab_navrl.tasks", types.ModuleType("isaaclab_navrl.tasks"))
    env = types.SimpleNamespace(close=Mock())
    factory = Mock(return_value=env)
    monkeypatch.setattr(entry, "create_navigation_env", factory)

    class IdentifiedRuntime(Exception):
        pass

    metadata = Mock(side_effect=IdentifiedRuntime)
    monkeypatch.setattr(entry, "runtime_metadata", metadata)
    tokens = [name, "--output", str(tmp_path / "result")]
    if name == "diagnose_zhang":
        checkpoint = tmp_path / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint fixture")
        payload = {
            "format_version": 3,
            "method": "zhang",
            "completed_updates": 2,
            "num_envs": 4,
            "train_cfg": asdict(ZhangTrainCfg(geometry_backend="native")),
            "environment": {
                "seed": 7,
                "terrain_seed": 8,
                "training_stage": "pretrain_s0",
                "static_obstacles": 240,
                "dynamic_obstacles": 0,
                "sim_to_real": {},
            },
        }
        monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: payload)
        monkeypatch.syspath_prepend(str(SCRIPTS))
        actor_critic = types.ModuleType("isaaclab_navrl.zhang.actor_critic")
        actor_critic.CostActorCritic = None
        runner = types.ModuleType("isaaclab_navrl.zhang.runner")
        runner.ZhangTrainer = None
        monkeypatch.setitem(sys.modules, actor_critic.__name__, actor_critic)
        monkeypatch.setitem(sys.modules, runner.__name__, runner)
        tokens += ["--checkpoint", str(checkpoint)]
    monkeypatch.setattr(sys, "argv", tokens)
    with pytest.raises(IdentifiedRuntime):
        entry.main()
    task, actual_cfg, args = factory.call_args.args
    assert task == (
        "Isaac-NavRL-Hummingbird-Direct-v0" if name == "profile_step"
        else "Isaac-NavRL-Zhang-Hummingbird-Direct-v0"
    )
    assert actual_cfg is cfg
    assert (args.physics_preset, args.frontend) == ("newton_mjwarp", "warp")
    metadata.assert_called_once_with(env)
    env.close.assert_called_once_with()
    if name == "diagnose_zhang":
        identity = json.loads((tmp_path / "result" / "diagnostic_identity.json").read_text())
        assert identity["source_update"] == 2


@pytest.mark.parametrize("name", ["benchmark_camera", "profile_step", "diagnose_zhang"])
@pytest.mark.parametrize("selector", [["physics=isaacsim_physx"], ["--frontend", "torch"], ["--frontend", "warp"]])
def test_diagnostics_reject_retired_selectors_before_constructing_runtime(monkeypatch, tmp_path, name, selector):
    entry = load_entry(monkeypatch, name)
    factory = Mock()
    monkeypatch.setattr(entry, "create_navigation_env", factory)
    tokens = [name, "--output", str(tmp_path / "result"), *selector]
    if name == "diagnose_zhang":
        tokens += ["--checkpoint", str(tmp_path / "unused.pt")]
    monkeypatch.setattr(sys, "argv", tokens)
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
    factory.assert_not_called()
