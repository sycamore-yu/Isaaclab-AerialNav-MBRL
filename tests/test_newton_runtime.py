"""One registered Newton/Warp runtime, including construction failure cleanup."""

import argparse
import sys
from types import ModuleType, SimpleNamespace

import pytest

from isaaclab_navrl.physics import create_navigation_env, parse_runtime_args, runtime_metadata
from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import NavRLEnvCfg, ZhangNavRLEnvCfg


def test_drone_retains_authored_rigid_body_properties_for_newton_import():
    from isaaclab_navrl.control import make_drone_cfg

    spawn = make_drone_cfg("/World/Drone").spawn
    assert spawn.rigid_props.disable_gravity is False
    assert spawn.rigid_props.retain_accelerations is False
    assert spawn.rigid_props.linear_damping == spawn.rigid_props.angular_damping == 0.0
    assert spawn.rigid_props.max_linear_velocity == spawn.rigid_props.max_angular_velocity == 1000.0
    assert spawn.rigid_props.max_depenetration_velocity == 1.0
    assert spawn.articulation_props.self_collision_enabled is False


@pytest.mark.parametrize("config_type", [NavRLEnvCfg, ZhangNavRLEnvCfg])
def test_scene_configuration_is_concrete_newton_and_shared_geometry(config_type):
    from isaaclab_newton.physics import NewtonCfg

    cfg = config_type()
    cfg.training_curriculum = False
    cfg.num_dynamic_obstacles = 8
    cfg.configure_scene()
    assert isinstance(cfg.sim.physics, NewtonCfg)
    assert cfg.scene.replicate_physics and cfg.scene.env_spacing == 0
    assert not hasattr(cfg.scene, "dynamic_obstacles")
    assert len([key for key in vars(cfg.scene) if key.startswith("dynamic_visual_")]) == 8
    assert all(not target.track_mesh_transforms for target in cfg.scene.depth_camera.mesh_prim_paths)
    assert str(cfg.scene.lidar.class_type).endswith("NewtonNavRLRayCaster")
    assert str(cfg.scene.depth_camera.class_type).endswith("NewtonBatchedDepthCamera")
    cfg.num_dynamic_obstacles = 0
    cfg.configure_scene()
    assert not any(key.startswith("dynamic_visual_") for key in vars(cfg.scene))


def test_registration_has_no_alternate_environment():
    import gymnasium as gym

    import isaaclab_navrl.tasks  # noqa: F401

    for suffix, name in (("", "NavRLEnv"), ("-Zhang", "ZhangNavRLEnv")):
        spec = gym.spec(f"Isaac-NavRL{suffix}-Hummingbird-Direct-v0")
        assert spec.entry_point == f"isaaclab_navrl.tasks.navigation.navrl_env:{name}"
        assert spec.kwargs["warp_entry_point"] == spec.entry_point


def test_construction_validates_actual_runtime_and_closes_incompatible_environment(monkeypatch):
    base = type("DirectRLEnvWarp", (), {"__module__": "isaaclab_experimental.envs.direct_rl_env_warp"})
    warp_type = type("NavigationEnv", (base,), {})
    args = parse_runtime_args(argparse.ArgumentParser(), [])
    factory = ModuleType("isaaclab_rl.entrypoints.common")
    monkeypatch.setitem(sys.modules, "isaaclab_rl.entrypoints.common", factory)
    env = warp_type()
    env.unwrapped = env
    factory.create_isaaclab_env = lambda *a, **kw: env
    assert create_navigation_env("task", object(), args) is env
    assert runtime_metadata(env)["frontend"] == "warp"
    closed = []
    wrong = SimpleNamespace(close=lambda: closed.append(True))
    wrong.unwrapped = wrong
    factory.create_isaaclab_env = lambda *a, **kw: wrong
    with pytest.raises(RuntimeError, match="Warp"):
        create_navigation_env("task", object(), args)
    assert closed == [True]
