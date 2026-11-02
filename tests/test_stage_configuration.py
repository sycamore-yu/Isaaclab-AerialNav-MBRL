"""Stage application preserves explicit overrides and existing scene configuration."""

from copy import deepcopy

import pytest

from aerialnav.tasks.navigation.config.hummingbird.env_cfg import DiffNavEnvCfg, NavigationEnvCfg
from aerialnav.training import STAGE_CONFIG


def test_stage_switch_resets_override_and_builds_scene_from_new_settings():
    cfg = DiffNavEnvCfg()
    original = deepcopy(STAGE_CONFIG)
    override = STAGE_CONFIG["finetune_s1"] | {"dynamic_obstacles": 0}
    resolved = cfg.apply_training_stage("finetune_s1", stage_config=override)
    cfg.configure_scene()
    assert resolved == override
    assert cfg.num_static_obstacles == 320 and cfg.dynamic_capacity() == 0
    assert cfg.sim_to_real.enabled and cfg.episode_length_s == 40.0
    assert cfg.training_stage_config == override

    cfg.apply_training_stage("pretrain_s1")
    cfg.configure_scene()
    assert cfg.training_stage == "pretrain_s1" and cfg.training_stage_config is None
    assert cfg.num_static_obstacles == 240 and cfg.dynamic_capacity() == 120
    assert not cfg.sim_to_real.enabled and cfg.episode_length_s == 35.0
    assert cfg.scene.depth_camera.update_period == cfg.sim.dt * cfg.decimation
    cfg.apply_training_stage("finetune_s2")
    cfg.configure_scene()
    assert cfg.num_static_obstacles == 240 and cfg.dynamic_capacity() == 240
    assert cfg.sim_to_real.enabled and cfg.episode_length_s == 45.0
    assert STAGE_CONFIG == original


def test_ppo_explicit_static_stage_and_manual_scene_overrides_survive():
    cfg = NavigationEnvCfg()
    resolved = cfg.apply_training_stage(
        "pretrain_s0",
        stage_config=STAGE_CONFIG["pretrain_s0"]
        | {"static_obstacles": 300, "dynamic_obstacles": 0, "randomization": False},
    )
    cfg.configure_scene()
    assert cfg.num_static_obstacles == 300 and cfg.dynamic_capacity() == 0
    assert not cfg.sim_to_real.enabled and cfg.episode_length_s == 30.0
    assert resolved["static_obstacles"] == 300
    # Existing callers may still override counts directly before rebuilding the scene.
    cfg.num_static_obstacles, cfg.num_dynamic_obstacles = 12, 8
    cfg.configure_scene()
    assert cfg.num_static_obstacles == 12 and cfg.dynamic_capacity() == 8


def test_unknown_stage_does_not_change_configuration():
    cfg = NavigationEnvCfg()
    previous = (cfg.training_stage, cfg.num_static_obstacles, cfg.num_dynamic_obstacles)
    with pytest.raises(KeyError, match="unknown"):
        cfg.apply_training_stage("unknown")
    assert (cfg.training_stage, cfg.num_static_obstacles, cfg.num_dynamic_obstacles) == previous
