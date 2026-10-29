"""Continuous ground recovery changes geometry only below the ground plane."""

import pytest
import torch
from torch.nn import functional as F

from isaaclab_navrl.zhang.config import ZhangTrainCfg
from isaaclab_navrl.zhang.losses import zhang_transition_costs


def ground_history(query_height, terminal_height=None):
    """Analytic nearest points for actual static z=0 and shared ground z=.01."""
    count, batch, _ = query_height.shape
    selected_plane = torch.where(query_height.detach().abs() < (0.01 - query_height.detach()).abs(),
                                 torch.zeros_like(query_height), torch.full_like(query_height, 0.01))
    vectors = torch.stack((query_height * 0, query_height * 0, selected_plane - query_height), dim=-1)
    terminal_height = query_height[..., 0] if terminal_height is None else terminal_height
    position = torch.stack((terminal_height * 0, terminal_height * 0, terminal_height), dim=-1)
    zero = query_height.new_zeros(count, batch, 3)
    signed = query_height - 0.01
    return {
        "velocity": list(zero.unbind(0)), "target_velocity": list(zero.unbind(0)),
        "episode_id": [torch.zeros(batch, dtype=torch.long)] * count,
        "position": list(position.unbind(0)), "obstacle_vector": list(vectors.unbind(0)),
        "ground_signed_distance": list(signed.unbind(0)),
        "margin": [query_height.new_full((batch,), 0.3)] * count,
        "command": list(zero.unbind(0)), "previous_command": list(zero.unbind(0)),
        "velocity_prediction": list(zero.unbind(0)), "observed_velocity": list(zero.unbind(0)),
        "dcol_gap": list((signed[..., 1:] - 0.3).unbind(0)),
    }


def original_obstacle_terms(history, cfg):
    """Original unsigned distance and detached approach, without any correction."""
    distance = torch.stack(history["obstacle_vector"]).norm(dim=-1) - torch.stack(history["margin"])[..., None]
    approach = (-distance.diff(dim=-1) * (9 / cfg.policy_dt)).clamp_min(1).detach()
    gap = torch.stack(history["dcol_gap"]) if cfg.variant == "dcol" else distance[..., 1:]
    avoidance = (approach * (1 - distance[..., 1:]).relu().square()).mean(dim=-1)
    collision = (F.softplus(-32 * gap) * approach).mean(dim=-1)
    return avoidance, collision


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_all_above_ground_values_and_gradients_are_exactly_unchanged(dtype):
    base = torch.tensor([[0.5, 1.6], [0.8, 2.0], [1.0, 2.5]], dtype=dtype)
    # Descending look-ahead values exercise an approach weight greater than one.
    query = (base[..., None] + torch.linspace(0.1, -0.1, 10, dtype=dtype)).requires_grad_()
    history = ground_history(query, torch.full_like(base, 2.0))
    cfg = ZhangTrainCfg(variant="continuous")
    avoidance, collision = original_obstacle_terms(history, cfg)
    expected = cfg.coef_obj_avoidance * avoidance + cfg.coef_collide * collision
    losses, _, cost = zhang_transition_costs(history, cfg)
    torch.testing.assert_close(cost, expected, rtol=0, atol=0)
    torch.testing.assert_close(losses["obstacle_avoidance"], avoidance.mean(), rtol=0, atol=0)
    torch.testing.assert_close(losses["collision"], collision.mean(), rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(losses["total"], query, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.mean(), query)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_below_ground_both_obstacle_terms_and_height_restore_upward():
    height = torch.tensor([-0.1, -0.5, -1.5, -2.0], dtype=torch.float64, requires_grad=True)
    query = height[None, :, None].expand(3, -1, 10)
    history = ground_history(query)
    cfg = ZhangTrainCfg(variant="continuous")
    losses, _, cost = zhang_transition_costs(history, cfg)
    batch = len(height)
    avoidance_gradient = torch.autograd.grad(losses["obstacle_avoidance"], height, retain_graph=True)[0] * batch
    collision_gradient = torch.autograd.grad(losses["collision"], height, retain_graph=True)[0] * batch
    height_gradient = torch.autograd.grad(losses["height_boundary"], height, retain_graph=True)[0] * batch
    torch.testing.assert_close(avoidance_gradient, -2 * (1.31 - height.detach()), rtol=1e-12, atol=1e-12)
    logits = -32 * (height.detach() - 0.31)
    # F.softplus switches to the exactly linear branch above its default
    # threshold, so its derivative is exactly one there, not sigmoid(logit).
    softplus_derivative = torch.where(logits > 20, torch.ones_like(logits), torch.sigmoid(logits))
    torch.testing.assert_close(collision_gradient, -32 * softplus_derivative, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(height_gradient, torch.full_like(height, -2.0), rtol=0, atol=0)
    assert bool((avoidance_gradient < 0).all() & (collision_gradient < 0).all() & (height_gradient < 0).all())
    expected_gradient = (cfg.coef_obj_avoidance * avoidance_gradient + cfg.coef_collide * collision_gradient
                         + cfg.coef_height_boundary * height_gradient)
    actual_gradient = torch.autograd.grad(cost[0].sum(), height)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(losses["total"], cost.mean(), rtol=0, atol=0)


def test_lookahead_crossing_ground_keeps_the_original_detached_approach_formula():
    query = torch.tensor([[[0.2, 0.15, 0.1, 0.05, 0.02, -0.02, -0.1, -0.2, -0.3, -0.4]]],
                         dtype=torch.float64, requires_grad=True)
    history = ground_history(query, torch.full((1, 1), 2.0, dtype=torch.float64))
    cfg = ZhangTrainCfg(variant="continuous")
    losses, _, cost = zhang_transition_costs(history, cfg)
    # This descent crosses the plane inside the ten-query trace. Each ground
    # clearance, on either side, should be z-.01-margin with d(clearance)/dz=1.
    clearance = (query - 0.01) - 0.3
    approach = (-clearance.diff(dim=-1) * (9 / cfg.policy_dt)).clamp_min(1).detach()
    avoidance = (approach * (1 - clearance[..., 1:]).square()).mean(dim=-1)
    collision = (approach * F.softplus(-32 * clearance[..., 1:])).mean(dim=-1)
    expected = cfg.coef_obj_avoidance * avoidance + cfg.coef_collide * collision
    torch.testing.assert_close(cost, expected, rtol=1e-12, atol=1e-12)
    actual_gradient = torch.autograd.grad(losses["total"], query, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.mean(), query)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert actual_gradient[0, 0, 0] == 0  # The first query only sets a detached approach weight.
    assert bool((actual_gradient[..., 1:] < 0).all())


@pytest.mark.parametrize("query_z,terminal_z,should_penalize", [(2.0, -2.0, False), (-0.5, 2.0, True)])
def test_ground_geometry_uses_pre_step_queries_not_post_step_or_reset_height(query_z, terminal_z, should_penalize):
    query = torch.full((3, 1, 10), query_z, dtype=torch.float64, requires_grad=True)
    terminal = torch.full((3, 1), terminal_z, dtype=torch.float64, requires_grad=True)
    cfg = ZhangTrainCfg(variant="continuous", coef_height_boundary=0)
    losses, _, _ = zhang_transition_costs(ground_history(query, terminal), cfg)
    assert bool(losses["obstacle_avoidance"] > 0) is should_penalize
    assert bool(losses["collision"] > 1) is should_penalize
    query_gradient, terminal_gradient = torch.autograd.grad(losses["total"], (query, terminal))
    torch.testing.assert_close(terminal_gradient, torch.zeros_like(terminal), rtol=0, atol=0)
    if should_penalize:
        assert bool((query_gradient[..., 1:] < 0).all())


@pytest.mark.parametrize("variant", ("step", "dcol"))
def test_step_and_dcol_ignore_new_ground_field_and_preserve_original_geometry(variant):
    height = torch.tensor([-0.1, -0.5, 0.5, 2.0], dtype=torch.float64, requires_grad=True)
    query = height[None, :, None].expand(3, -1, 10)
    history = ground_history(query, height.new_full((3, len(height)), 2.0))
    cfg = ZhangTrainCfg(variant=variant)
    avoidance, collision = original_obstacle_terms(history, cfg)
    expected = cfg.coef_obj_avoidance * avoidance + cfg.coef_collide * collision
    losses, _, cost = zhang_transition_costs(history, cfg)
    torch.testing.assert_close(cost, expected, rtol=0, atol=0)
    without = dict(history)
    del without["ground_signed_distance"]
    no_field_losses, _, no_field_cost = zhang_transition_costs(without, cfg)
    torch.testing.assert_close(cost, no_field_cost, rtol=0, atol=0)
    for key in losses:
        torch.testing.assert_close(losses[key], no_field_losses[key], rtol=0, atol=0)
    gradient = torch.autograd.grad(losses["total"], height, retain_graph=True)[0]
    old_gradient = torch.autograd.grad(expected.mean(), height, retain_graph=True)[0]
    torch.testing.assert_close(gradient, old_gradient, rtol=0, atol=0)
    # Document the deliberately unchanged unsigned avoidance component in dcol.
    avoidance_gradient = torch.autograd.grad(losses["obstacle_avoidance"], height)[0]
    assert bool((avoidance_gradient[:2] > 0).all())


def test_continuous_requires_ground_queries_and_rejects_misaligned_samples():
    query = torch.full((3, 1, 10), 2.0)
    history = ground_history(query)
    cfg = ZhangTrainCfg(variant="continuous")
    missing = dict(history)
    del missing["ground_signed_distance"]
    with pytest.raises(ValueError, match="requires ground_signed_distance"):
        zhang_transition_costs(missing, cfg)
    wrong_shape = dict(history, ground_signed_distance=[torch.ones(1, 9)] * 3)
    with pytest.raises(ValueError, match="must match"):
        zhang_transition_costs(wrong_shape, cfg)
