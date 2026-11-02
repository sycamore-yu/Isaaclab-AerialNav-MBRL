"""New task distribution, stage gates and analytical gradient reshaping."""

import math

import numpy as np
import pytest
import torch

from aerialnav.diffnav.gradient_field import reshape_avoidance_distance
from aerialnav.diffnav.termination import first_events
from aerialnav.tasks.navigation.task_logic import DynamicObstacleMotion, dynamic_layout, sample_navigation_route
from aerialnav.training import STAGE_CONFIG


def test_routes_preserve_uniform_speed_exact_distance_heading_and_bounds():
    generator = torch.Generator().manual_seed(19)
    speed = 1 + 5 * torch.rand(10000, generator=generator)
    speed[:2] = torch.tensor([1.0, 6.0])
    start, goal = sample_navigation_route(speed, generator)
    delta = goal - start
    torch.testing.assert_close(delta[:, :2].norm(dim=-1), 18 * speed, atol=2e-5, rtol=1e-6)
    for point in (start, goal):
        assert (point[:, 0].abs() <= 54.00001).all() and (point[:, 1].abs() <= 20.00001).all()
        assert ((point[:, 2] >= 0.5) & (point[:, 2] <= 2.5)).all()
    angle = torch.atan2(delta[:, 1], delta[:, 0]).remainder(math.pi)
    deviation = torch.minimum(angle, math.pi - angle)
    assert deviation.max() <= math.radians(8) + 1e-6
    assert 0.48 < (delta[:, 0] > 0).float().mean() < 0.52
    with pytest.raises(ValueError, match="at most 6"):
        sample_navigation_route(torch.tensor([6.01]), generator)


def test_new_curriculum_counts_density_and_timeouts():
    assert [(s["static_obstacles"], s["dynamic_obstacles"], s["episode_length_s"]) for s in STAGE_CONFIG.values()] == [
        (240, 0, 30),
        (240, 120, 35),
        (180, 180, 40),
        (320, 160, 40),
        (240, 240, 45),
    ]
    assert [s["static_obstacles"] / 4800 for s in STAGE_CONFIG.values()] == pytest.approx(
        [0.05, 0.05, 0.0375, 320 / 4800, 0.05]
    )
    assert [(s["static_obstacles"] + s["dynamic_obstacles"]) / 4800 for s in STAGE_CONFIG.values()] == pytest.approx(
        [0.05, 0.075, 0.075, 0.1, 0.1]
    )


@pytest.mark.parametrize("count", [0, 120, 160, 180, 240])
def test_dynamic_geometry_and_motion_stay_outside_clear_end_strips(count):
    positions, sizes = dynamic_layout(count, 7)
    assert np.array_equal(positions, dynamic_layout(count, 7)[0])
    motion = DynamicObstacleMotion(count, 7, "cpu", 0.05)
    for _ in range(300):
        motion.advance()
        assert (motion.position[:, 0].abs() + motion.size[:, 0] / 2 <= 54.00001).all()
        assert (motion.position[:, 1].abs() + motion.size[:, 1] / 2 <= 20.00001).all()
    if count:
        assert motion.position[:, 0].amax() - motion.position[:, 0].amin() > 50
        assert (motion.velocity.norm(dim=-1) <= 1.50001).all()


def test_map_xy_bounds_are_terminal_even_when_height_reset_is_disabled():
    state = torch.zeros(4, 13)
    state[:, 2] = 2
    state[:, 0] = torch.tensor([59.9, 60.1, 0.0, 0.0])
    state[:, 1] = torch.tensor([0.0, 0.0, 19.9, -20.1])
    terms = first_events(state, state[:, :3] + 10, torch.ones(4), height_bounds_reset=False)
    assert terms["bounds"].tolist() == [False, True, False, True]


def test_gradient_reshaping_preserves_values_norms_and_vertical_safety():
    query = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 1.0]]], requires_grad=True)
    normal = torch.tensor([[[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    distance = (query * normal).sum(-1) + torch.tensor([[0.3, 3.0, -0.7]])
    velocity = torch.tensor([[2.0, 0.0, 0.0]])
    reshaped = reshape_avoidance_distance(distance, query, velocity, velocity)
    assert torch.equal(distance, reshaped)
    gradient = torch.autograd.grad(reshaped.sum(), query)[0]
    delta = (math.pi / 2) * 0.9 * (1 - math.exp(-3))
    torch.testing.assert_close(gradient[0, 0], torch.tensor([-math.cos(delta), -math.sin(delta), 0.0]))
    torch.testing.assert_close(gradient[0, 1:], normal[0, 1:])
    torch.testing.assert_close(gradient.norm(dim=-1), normal.norm(dim=-1))


def test_gradient_reshaping_leaves_receding_obstacles_and_rest_finite():
    for velocity in (torch.tensor([[-1.0, 0.0, 0.0]]), torch.zeros(1, 3)):
        query = torch.zeros(1, 1, 3, requires_grad=True)
        distance = 0.3 - query[..., 0]
        reshaped = reshape_avoidance_distance(distance, query, velocity, torch.tensor([[1.0, 0.0, 0.0]]))
        gradient = torch.autograd.grad(reshaped.sum(), query)[0]
        assert torch.isfinite(gradient).all()
        if velocity[0, 0] < 0:
            torch.testing.assert_close(gradient, torch.tensor([[[-1.0, 0.0, 0.0]]]))


def test_avoidance_reshaping_keeps_collision_loss_and_forward_cost_unchanged():
    from test_diffnav_contract import loss_history

    from aerialnav.diffnav.config import DiffNavTrainCfg
    from aerialnav.diffnav.losses import diffnav_transition_costs

    query = torch.zeros(1, 10, 3, requires_grad=True)
    distance = 0.3 - query[..., 0]
    velocity = torch.tensor([[1.0, 0.0, 0.0]])
    shaped = reshape_avoidance_distance(distance, query, velocity, velocity)
    history = loss_history(torch.zeros(1, 1, 3))
    history["obstacle_signed_distance"] = [distance]
    cfg = DiffNavTrainCfg(
        variant="step", velocity_loss_mode="per_step", loss_timing="deferred", position_integration="original"
    )
    original, _, _ = diffnav_transition_costs(history, cfg)
    changed, _, _ = diffnav_transition_costs(history | {"avoidance_signed_distance": [shaped]}, cfg)
    for name in original:
        torch.testing.assert_close(changed[name], original[name], equal_nan=True)
    collision = torch.autograd.grad(changed["collision"], query, retain_graph=True)[0]
    assert (collision[..., 1:] == 0).all()
    avoidance = torch.autograd.grad(changed["obstacle_avoidance"], query)[0]
    assert avoidance[..., 1].abs().sum() > avoidance[..., 0].abs().sum()


def test_public_direct_profile_selects_the_confirmed_baseline():
    import json
    from pathlib import Path

    from aerialnav.diffnav.config import DiffNavTrainCfg
    from aerialnav.training import DEFAULT_STAGE, STAGE_CONFIG

    profile = json.loads((Path(__file__).parents[1] / "configs/baseline_direct.json").read_text())
    cfg = DiffNavTrainCfg()
    assert DEFAULT_STAGE == profile["stage"] == "finetune_s1"
    # The packaged checkpoint keeps its original 480/32 metadata.
    assert profile["static_obstacles"] == 480 and profile["dynamic_obstacles"] == 32
    assert STAGE_CONFIG[DEFAULT_STAGE]["static_obstacles"] == 320
    assert STAGE_CONFIG[DEFAULT_STAGE]["dynamic_obstacles"] == 160
    for name in (
        "variant",
        "collision_metric",
        "geometry_backend",
        "position_integration",
        "loss_timing",
        "avoidance_gradient",
        "velocity_window_s",
        "timesteps",
    ):
        assert getattr(cfg, name) == profile[name]
    assert cfg.max_iterations is None


def test_warp_frontend_enforces_the_same_rectangular_map_boundaries():
    import warp as wp

    from aerialnav.tasks.navigation.warp_task_kernels import WarpFirstEvents

    wp.init()
    state = torch.zeros(4, 13)
    state[:, 2] = 2
    state[:, 0] = torch.tensor([59.9, 60.1, 0.0, 0.0])
    state[:, 1] = torch.tensor([0.0, 0.0, 19.9, -20.1])
    events = WarpFirstEvents(root_state=state, target=state[:, :3] + 10, collision_gap=torch.ones(4))
    events.launch()
    assert events.terms["bounds"].tolist() == [False, True, False, True]


def test_static_heightfield_leaves_six_metre_end_strips_clear():
    from aerialnav.terrain import build_terrain_cfg, discrete_obstacles_height_field

    terrain = build_terrain_cfg(1, 600, 7).terrain_generator
    cfg = terrain.sub_terrains["obstacles"]
    cfg.size, cfg.seed = terrain.size, 7
    heights, _ = discrete_obstacles_height_field(0.0, cfg)
    assert heights.shape == (1200, 400)
    assert not heights[:60].any() and not heights[-60:].any()
    assert heights[60:-60].any()
