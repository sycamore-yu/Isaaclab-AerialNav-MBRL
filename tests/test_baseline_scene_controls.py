"""Scene controls reach training commands, reset sampling and episode reports."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from isaaclab_navrl.tasks.navigation.curriculum import TrainingCurriculum
from isaaclab_navrl.training import STAGE_CONFIG
from isaaclab_navrl.zhang.cli import parse_train_args
from isaaclab_navrl.zhang.events import randomize_zhang_parameters

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "tokens",
    [
        ["--num_dynamic_obstacles", "-1"],
        ["--target_speed_min", "2"],
        ["--target_speed_min", "6", "--target_speed_max", "2"],
        ["--target_speed_min", "2", "--target_speed_max", "nan"],
    ],
)
def test_invalid_scene_controls_are_rejected(tokens):
    with pytest.raises(ValueError):
        parse_train_args(tokens)


@pytest.mark.parametrize("stage", ["pretrain_s1", "finetune_s1"])
@pytest.mark.parametrize("count", [40, 60])
def test_dynamic_count_survives_curriculum_and_report(stage, count):
    cfg = {**STAGE_CONFIG[stage], "dynamic_obstacles": count}
    curriculum = TrainingCurriculum(1, "cpu", stage=stage, stage_config=cfg)
    assert curriculum.active_obstacles == count
    curriculum.observe(
        {
            "first_event_code": torch.ones(1, dtype=torch.long),
            "first_event_time_s": torch.ones(1),
            "terminated": torch.ones(1, dtype=torch.bool),
            "timeout": torch.zeros(1, dtype=torch.bool),
            "episode_return": torch.zeros(1),
            "step": torch.ones(1, dtype=torch.long),
        }
    )
    assert curriculum.pop_report()["dynamic_obstacles"] == count
    assert curriculum.state_dict()["stage_config"] == cfg
    expected = 120 if stage == "pretrain_s1" else 160
    assert STAGE_CONFIG[stage]["dynamic_obstacles"] == expected
    assert TrainingCurriculum(1, "cpu", stage=stage).active_obstacles == expected


def test_speed_control_samples_requested_range_on_partial_reset():
    env = SimpleNamespace(
        num_envs=4096,
        device="cpu",
        cfg=SimpleNamespace(seed=17, evaluation=False, training_stage="finetune_s1"),
        mass_scale=torch.ones(4096, 1),
        inertia_scale=torch.ones(4096, 1),
        perturbation=SimpleNamespace(thrust_scale=torch.ones(4096, 1), control_gain_scale=torch.ones(4096, 1)),
        apply_zhang_camera_randomization=lambda ids: None,
    )
    randomize_zhang_parameters(env, None, target_speed_range=(2, 6))
    speed = env.zhang_parameters["max_speed"]
    assert 2 <= speed.min() < 2.01 and 5.99 < speed.max() <= 6
    before = speed.clone()
    randomize_zhang_parameters(env, torch.tensor([0]), target_speed_range=(2, 6))
    torch.testing.assert_close(speed[1:], before[1:], rtol=0, atol=0)
    assert speed[0] != before[0]


def test_reconfiguring_scene_preserves_speed_event_parameters():
    import ast

    path = ROOT / "src/isaaclab_navrl/tasks/navigation/config/hummingbird/env_cfg.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ZhangNavRLEnvCfg")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "configure_scene")

    class Base:
        def configure_scene(self):
            self.events = SimpleNamespace(parameters=SimpleNamespace(params={}), disturbance=SimpleNamespace(params={}))

    # Run the actual configuration method with a base that recreates events as Isaac does.
    wrapper = ast.ClassDef(
        name="Config", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    namespace = {"Base": Base, "UsdPhysicsCollisionCfg": lambda **kw: kw, "ResolvableString": str}
    exec(compile(module, str(path), "exec"), namespace)
    cfg = namespace["Config"]()
    cfg.variant, cfg.timing = "dcol", "unified_20hz"
    cfg.geometry_backend = "native"
    cfg.height_target_range, cfg.target_speed_range = (0.5, 2.5), (2, 6)
    cfg.sim = SimpleNamespace()
    cfg.perception_cfg = SimpleNamespace(point_lidar=None)
    cfg.scene = SimpleNamespace(drone=SimpleNamespace(spawn=SimpleNamespace()), lidar=SimpleNamespace())
    for _ in range(2):
        cfg.configure_scene()
        assert cfg.events.parameters.params["target_speed_range"] == (2, 6)


def test_acceptance_validator_uses_requested_obstacle_count():
    from test_acceptance_protocol import report_fixture

    protocol = runpy.run_path(str(ROOT / "scripts/run_acceptance.py"))
    result = report_fixture(stage="pretrain_s1", num_envs=256)
    curriculum = result["curriculum"]
    curriculum["stage_config"]["dynamic_obstacles"] = 60
    curriculum["active_dynamic_obstacles"] = 60
    curriculum["latest_training_report"]["dynamic_obstacles"] = 60
    options = dict(
        method="zhang",
        stage="pretrain_s1",
        num_envs=256,
        parent_checkpoint=None,
        screening=True,
        fresh_start_stage="pretrain_s1",
    )
    assert not protocol["validate_stage_result"](result, **options, num_dynamic_obstacles=60)
    assert protocol["validate_stage_result"](result, **options)
