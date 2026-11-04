"""Native preset composition, unchanged defaults and exact checkpoint contracts."""

import argparse
import copy
from types import SimpleNamespace

import gymnasium as gym
import pytest
from isaaclab_tasks.utils.preset_cli import enumerate_task_presets

import aerialnav.tasks  # noqa: F401
from aerialnav.physics import parse_physics_args, resolve_navigation_config
from aerialnav.pointnet.checkpoint import policy_contract, validate_checkpoint


def compose(tokens=(), training=True, task="Isaac-AerialNav-Depth-v0"):
    args = parse_physics_args(argparse.ArgumentParser(), list(tokens), allow_presets=True)
    env, recipe = resolve_navigation_config(task, args, training=training)
    env.configure_scene()
    return env, recipe


def test_native_preset_discovery_keeps_one_navigation_environment():
    variants = enumerate_task_presets("Isaac-AerialNav-Depth-v0")
    assert "pointnet" in [x for names in variants.values() for x in names]
    assert "Isaac-AerialNav-PointCloud-v0" not in gym.registry
    default, recipe = compose()
    selected, point_recipe = compose(["presets=pointnet"])
    assert type(default) is type(selected)
    assert default.perception_cfg.mode == "depth" and default.scene.point_lidar is None
    assert selected.perception_cfg.mode == "pointnet" and selected.scene.point_lidar.ray_alignment == "base"
    assert recipe.perception == "depth" and point_recipe.perception == "lidar"
    assert point_recipe.timesteps == 160 and point_recipe.lr == 1e-4
    assert selected.action_space == default.action_space == 3
    assert selected.scene.lidar.ray_alignment == default.scene.lidar.ray_alignment == "yaw"
    ppo, _ = compose(training=False, task="Isaac-AerialNav-PPO-v0")
    assert ppo.method == "ppo" and ppo.scene.point_lidar is None
    with pytest.raises(ValueError, match="Unknown preset"):
        compose(["presets=pointnet"], training=False, task="Isaac-AerialNav-PPO-v0")
    with pytest.raises(ValueError, match="Unknown preset"):
        compose(["presets=not_a_preset"])


@pytest.mark.parametrize("change", ["alignment", "range", "outputs", "state_dim", "preset"])
def test_checkpoint_configuration_mismatch_is_not_silently_accepted(change):
    cfg, _ = compose(["presets=pointnet"])
    env = SimpleNamespace(cfg=cfg)
    contract = policy_contract(env)
    validate_checkpoint({"method": "pointnet", "policy_contract": contract}, env)
    wrong = copy.deepcopy(contract)
    if change == "alignment":
        wrong["sensor"]["ray_alignment"] = "yaw"
    elif change == "range":
        wrong["sensor"]["max_distance"] = 24
    elif change == "outputs":
        wrong["policy_output_dim"] = 6
    elif change == "state_dim":
        wrong["state_dim"] = 9
    else:
        wrong["preset"] = "depth"
    with pytest.raises(ValueError, match="configuration"):
        validate_checkpoint({"method": "pointnet", "policy_contract": wrong}, env)
    with pytest.raises(ValueError, match="configuration"):
        validate_checkpoint({"method": "pointnet"}, env)


def test_invalid_runtime_action_contract_is_rejected_before_saving():
    env = SimpleNamespace(cfg=SimpleNamespace(
        perception_cfg=SimpleNamespace(mode="pointnet", policy_outputs=6),
        scene=SimpleNamespace(point_lidar=object()), action_space=3,
    ))
    with pytest.raises(ValueError, match="three policy outputs"):
        policy_contract(env)


def test_domain_selectors_are_not_silently_ignored_by_other_entrypoints():
    with pytest.raises(SystemExit):
        parse_physics_args(argparse.ArgumentParser(), ["presets=pointnet"])
