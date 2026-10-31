"""Shared signed ground geometry preserves exterior values for every variant."""

import pytest
import torch
from torch.nn import functional as F

from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.diffnav.losses import diffnav_transition_costs


def ground_history(query_height, terminal_height=None):
    """Analytic nearest points for actual static z=0 and shared ground z=.01."""
    count, batch, _ = query_height.shape
    selected_plane = torch.where(
        query_height.detach().abs() < (0.01 - query_height.detach()).abs(),
        torch.zeros_like(query_height),
        torch.full_like(query_height, 0.01),
    )
    vectors = torch.stack((query_height * 0, query_height * 0, selected_plane - query_height), dim=-1)
    terminal_height = query_height[..., 0] if terminal_height is None else terminal_height
    position = torch.stack((terminal_height * 0, terminal_height * 0, terminal_height), dim=-1)
    zero = query_height.new_zeros(count, batch, 3)
    signed = query_height - 0.01
    return {
        "velocity": list(zero.unbind(0)),
        "target_velocity": list(zero.unbind(0)),
        "episode_id": [torch.zeros(batch, dtype=torch.long)] * count,
        "position": list(position.unbind(0)),
        "obstacle_vector": list(vectors.unbind(0)),
        "obstacle_signed_distance": list(signed.unbind(0)),
        "height_target": [query_height.new_full((batch,), 2.0)] * count,
        "height_range": [query_height.new_full((batch, 2), 2.0)] * count,
        "margin": [query_height.new_full((batch,), 0.3)] * count,
        "command": list(zero.unbind(0)),
        "previous_command": list(zero.unbind(0)),
        "velocity_prediction": list(zero.unbind(0)),
        "observed_velocity": list(zero.unbind(0)),
        "dcol_gap": list((signed[..., 1:] - 0.3).unbind(0)),
        **{
            name: [torch.zeros(batch, dtype=torch.bool, device=query_height.device) for _ in range(count)]
            for name in ("done", "collision", "bounds", "timeout", "invalid")
        },
        "episode_step": [
            torch.full((batch,), step + 1, dtype=torch.long, device=query_height.device) for step in range(count)
        ],
    }


def original_obstacle_terms(history, cfg):
    """Original unsigned distance and detached approach, without any correction."""
    distance = torch.stack(history["obstacle_vector"]).norm(dim=-1) - torch.stack(history["margin"])[..., None]
    approach = (-distance.diff(dim=-1) * (9 / cfg.policy_dt)).clamp_min(1).detach()
    # In this static-only exterior fixture the old DCOL ground/mesh minimum
    # also equals the unsigned-vector distance.  Derive it independently of
    # the new signed field rather than comparing the implementation to itself.
    gap = distance[..., 1:]
    avoidance = (approach * (1 - distance[..., 1:]).relu().square()).mean(dim=-1)
    collision = (F.softplus(-32 * gap) * approach).mean(dim=-1)
    return avoidance, collision


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("variant", ("step", "continuous", "dcol"))
def test_all_above_ground_values_and_gradients_are_exactly_unchanged(dtype, variant):
    base = torch.tensor([[0.5, 1.6], [0.8, 2.0], [1.0, 2.5]], dtype=dtype)
    # Descending look-ahead values exercise an approach weight greater than one.
    query = (base[..., None] + torch.linspace(0.1, -0.1, 10, dtype=dtype)).requires_grad_()
    history = ground_history(query, torch.full_like(base, 2.0))
    cfg = DiffNavTrainCfg(variant=variant)
    avoidance, collision = original_obstacle_terms(history, cfg)
    expected = cfg.coef_obj_avoidance * avoidance + cfg.coef_collide * collision
    losses, _, cost = diffnav_transition_costs(history, cfg)
    torch.testing.assert_close(cost, expected, rtol=0, atol=0)
    torch.testing.assert_close(losses["obstacle_avoidance"], avoidance.mean(), rtol=0, atol=0)
    torch.testing.assert_close(losses["collision"], collision.mean(), rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(losses["total"], query, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.mean(), query)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ("step", "continuous", "dcol"))
def test_below_ground_both_obstacle_terms_and_legacy_boundary_height_restore_upward(variant):
    height = torch.tensor([-0.1, -0.5, -1.5, -2.0], dtype=torch.float64, requires_grad=True)
    query = height[None, :, None].expand(3, -1, 10)
    history = ground_history(query)
    # Preserve this historical boundary-gradient contract explicitly; the
    # newly selected target-height modes have their own formula tests.
    cfg = DiffNavTrainCfg(variant=variant, height_loss_mode="boundary_huber")
    losses, _, cost = diffnav_transition_costs(history, cfg)
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
    expected_gradient = (
        cfg.coef_obj_avoidance * avoidance_gradient
        + cfg.coef_collide * collision_gradient
        + cfg.coef_height_boundary * height_gradient
    )
    actual_gradient = torch.autograd.grad(cost[0].sum(), height)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(losses["total"], cost.mean(), rtol=0, atol=0)


@pytest.mark.parametrize("variant", ("step", "continuous", "dcol"))
def test_lookahead_crossing_ground_keeps_the_original_detached_approach_formula(variant):
    query = torch.tensor(
        [[[0.2, 0.15, 0.1, 0.05, 0.02, -0.02, -0.1, -0.2, -0.3, -0.4]]], dtype=torch.float64, requires_grad=True
    )
    history = ground_history(query, torch.full((1, 1), 2.0, dtype=torch.float64))
    cfg = DiffNavTrainCfg(variant=variant)
    losses, _, cost = diffnav_transition_costs(history, cfg)
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
@pytest.mark.parametrize("variant", ("step", "continuous", "dcol"))
def test_ground_geometry_uses_pre_step_queries_not_post_step_or_reset_height(
    query_z,
    terminal_z,
    should_penalize,
    variant,
):
    query = torch.full((3, 1, 10), query_z, dtype=torch.float64, requires_grad=True)
    terminal = torch.full((3, 1), terminal_z, dtype=torch.float64, requires_grad=True)
    cfg = DiffNavTrainCfg(variant=variant, coef_height_boundary=0)
    losses, _, _ = diffnav_transition_costs(ground_history(query, terminal), cfg)
    assert bool(losses["obstacle_avoidance"] > 0) is should_penalize
    assert bool(losses["collision"] > 1) is should_penalize
    query_gradient, terminal_gradient = torch.autograd.grad(losses["total"], (query, terminal))
    torch.testing.assert_close(terminal_gradient, torch.zeros_like(terminal), rtol=0, atol=0)
    if should_penalize:
        assert bool((query_gradient[..., 1:] < 0).all())


@pytest.mark.parametrize("variant", ("step", "continuous", "dcol"))
def test_every_variant_uses_signed_ground_for_both_obstacle_terms(variant):
    height = torch.tensor([-0.1, -0.5, 0.5, 2.0], dtype=torch.float64, requires_grad=True)
    query = height[None, :, None].expand(3, -1, 10)
    history = ground_history(query, height.new_full((3, len(height)), 2.0))
    cfg = DiffNavTrainCfg(variant=variant)
    clearance = (query - 0.01) - 0.3
    approach = (-clearance.diff(dim=-1) * (9 / cfg.policy_dt)).clamp_min(1).detach()
    avoidance = (approach * (1 - clearance[..., 1:]).relu().square()).mean(dim=-1)
    collision = (approach * F.softplus(-32 * clearance[..., 1:])).mean(dim=-1)
    expected = cfg.coef_obj_avoidance * avoidance + cfg.coef_collide * collision
    losses, _, cost = diffnav_transition_costs(history, cfg)
    torch.testing.assert_close(cost, expected, rtol=0, atol=0)
    gradient = torch.autograd.grad(losses["total"], height, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.mean(), height, retain_graph=True)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    avoidance_gradient = torch.autograd.grad(losses["obstacle_avoidance"], height)[0]
    assert bool((avoidance_gradient[:2] < 0).all())


@pytest.mark.parametrize("variant", ("step", "continuous", "dcol"))
def test_every_variant_requires_signed_queries_and_rejects_misaligned_samples(variant):
    query = torch.full((3, 1, 10), 2.0)
    history = ground_history(query)
    cfg = DiffNavTrainCfg(variant=variant)
    missing = dict(history)
    del missing["obstacle_signed_distance"]
    with pytest.raises(ValueError, match="require obstacle_signed_distance"):
        diffnav_transition_costs(missing, cfg)
    wrong_shape = dict(history, obstacle_signed_distance=[torch.ones(1, 9)] * 3)
    with pytest.raises(ValueError, match="must match"):
        diffnav_transition_costs(wrong_shape, cfg)
    missing_target = dict(history)
    del missing_target["height_range"]
    with pytest.raises(ValueError, match="height_range"):
        diffnav_transition_costs(missing_target, cfg)
