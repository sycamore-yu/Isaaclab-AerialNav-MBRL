"""Independent velocity supervision must preserve geometry and transition timing."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from aerialnav.diffnav.actor_critic import short_horizon_cost_objective
from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.diffnav.dynamics import PointMassState, align_transition
from aerialnav.diffnav.losses import diffnav_transition_costs


def _history(velocity, target=None, episode=None, done=None):
    """Measured transitions, with separate terminal samples and episode labels."""
    count, batch, _ = velocity.shape
    target = torch.zeros_like(velocity) if target is None else target
    episode = torch.zeros(count, batch, dtype=torch.long, device=velocity.device) if episode is None else episode
    done = torch.zeros_like(episode, dtype=torch.bool) if done is None else done
    position = torch.zeros_like(velocity)
    position[..., 2] = 2.0
    vectors = velocity.new_zeros(count, batch, 10, 3)
    vectors[..., 0] = 10.0
    zero = torch.zeros_like(velocity)
    return {
        "velocity": list(velocity.unbind(0)),
        "target_velocity": list(target.unbind(0)),
        "episode_id": list(episode.unbind(0)),
        "done": list(done.unbind(0)),
        "collision": list(done.unbind(0)),
        "position": list(position.unbind(0)),
        "obstacle_signed_distance": list(vectors.norm(dim=-1).unbind(0)),
        "height_target": [velocity.new_full((batch,), 2.0)] * count,
        "height_range": [velocity.new_full((batch, 2), 2.0)] * count,
        "dcol_gap": [velocity.new_full((batch, 9), 9.7)] * count,
        "margin": [velocity.new_full((batch,), 0.3)] * count,
        "command": list(zero.unbind(0)),
        "previous_command": list(zero.unbind(0)),
        "velocity_prediction": list(zero.unbind(0)),
        "observed_velocity": list(zero.unbind(0)),
    }


def _velocity_only_cfg(**overrides):
    values = {
        "physics_dt": 0.01,
        "policy_dt": 0.1,
        "variant": "dcol",
        "velocity_loss_mode": "full_window",
        "coef_v_pred": 0.0,
        "coef_obj_avoidance": 0.0,
        "coef_collide": 0.0,
        "coef_d_acc": 0.0,
        "coef_d_jerk": 0.0,
        "coef_height_boundary": 0.0,
    }
    values.update(overrides)
    return DiffNavTrainCfg(**values)


def test_dcol_per_step_scores_every_short_episode_sample_and_ignores_window_cache():
    velocity = torch.tensor(
        [[[2.0, 0, 0]], [[0, 3.0, 0]], [[0, 0, 4.0]], [[5.0, 0, 0]]], dtype=torch.float64, requires_grad=True
    )
    target = velocity.new_tensor([[[1.0, 0, 0]], [[0, 1.0, 0]], [[0, 0, 1.0]], [[1.0, 0, 0]]])
    episode = torch.tensor([[4], [4], [5], [5]])
    done = torch.tensor([[False], [True], [False], [True]])
    history = _history(velocity, target, episode, done)
    old_velocity = velocity.new_full((20, 1, 3), 100.0, requires_grad=True)
    old_target = velocity.new_full((20, 1, 3), -100.0, requires_grad=True)
    prefix = {
        "velocity": old_velocity,
        "target_velocity": old_target,
        "episode_id": torch.full((20, 1), 4, dtype=torch.long),
    }
    cfg = _velocity_only_cfg(velocity_loss_mode="per_step")
    losses, cache, cost = diffnav_transition_costs(history, cfg, prefix)
    expected = velocity.new_tensor([[0.5], [1.5], [2.5], [3.5]])
    torch.testing.assert_close(cost, expected, rtol=0, atol=0)
    torch.testing.assert_close(losses["total"], expected.mean(), rtol=0, atol=0)
    torch.testing.assert_close(losses["total"], cost.mean(), rtol=0, atol=0)
    assert cache is None and losses["velocity_valid_samples"] == 4
    assert losses["velocity_sample_fraction"] == 1
    gradients = torch.autograd.grad(
        losses["total"], (velocity, old_velocity, old_target), retain_graph=True, allow_unused=True
    )
    expected_gradient = velocity.new_tensor([[[1.0, 0, 0]], [[0, 1.0, 0]], [[0, 0, 1.0]], [[1.0, 0, 0]]]) / 4
    torch.testing.assert_close(gradients[0], expected_gradient, rtol=0, atol=0)
    assert gradients[1:] == (None, None)
    # The collision terminal belongs to its own transition, without sending
    # this local cost into the next episode or another velocity sample.
    terminal_gradient = torch.autograd.grad(cost[1, 0], velocity, retain_graph=True)[0]
    expected_terminal_gradient = torch.zeros_like(velocity)
    expected_terminal_gradient[1, 0, 1] = 1
    torch.testing.assert_close(terminal_gradient, expected_terminal_gradient, rtol=0, atol=0)
    original, _, _ = diffnav_transition_costs(history, replace(cfg, velocity_loss_mode="full_window"))
    assert original["velocity"] == 0 and original["velocity_valid_samples"] == 0
    # SHAC receives the actual per-transition costs, without a sparse-window
    # H*N/K multiplier. Only its ordinary within-episode discount remains.
    actor = short_horizon_cost_objective(cost, torch.zeros_like(cost), done, discount=0.99)
    discounted = (expected * velocity.new_tensor([[1.0], [0.99], [1.0], [0.99]])).mean()
    torch.testing.assert_close(actor["discounted_cost"], discounted, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actor["objective"], discounted, rtol=1e-12, atol=1e-12)


def test_per_step_uses_terminal_velocity_before_same_step_reset():
    predicted_velocity = torch.tensor([[0.5, 0.7, 0.9]], dtype=torch.float64, requires_grad=True)
    zero = torch.zeros_like(predicted_velocity)
    predicted = PointMassState(zero, predicted_velocity, zero, zero)
    measured_velocity = predicted_velocity.new_tensor([[4.0, 0, 0]])
    reset_velocity = predicted_velocity.new_tensor([[100.0, 20.0, 0]], requires_grad=True)
    done = torch.tensor([True])
    terminal, following = align_transition(
        predicted,
        zero,
        measured_velocity,
        zero,
        done,
        zero,
        reset_velocity,
        0.1,
        zero,
    )
    torch.testing.assert_close(following.velocity, reset_velocity, rtol=0, atol=0)
    target = predicted_velocity.new_tensor([[[1.0, 0, 0]]])
    history = _history(terminal.velocity[None], target, done=done[None])
    losses, cache, cost = diffnav_transition_costs(history, _velocity_only_cfg(velocity_loss_mode="per_step"))
    torch.testing.assert_close(cost, predicted_velocity.new_tensor([[2.5]]), rtol=0, atol=0)
    assert cache is None and losses["velocity_sample_fraction"] == 1
    predicted_gradient, reset_gradient = torch.autograd.grad(
        losses["total"],
        (predicted_velocity, reset_velocity),
        allow_unused=True,
    )
    torch.testing.assert_close(predicted_gradient, predicted_velocity.new_tensor([[1.0, 0, 0]]), rtol=0, atol=0)
    assert reset_gradient is None


@pytest.mark.parametrize("variant", ("continuous", "dcol"))
def test_explicit_full_window_retains_known_values_gradients_and_detached_prefix(variant):
    # This is the pre-override W=2 regression case: one environment resets
    # inside the rollout; the other resets between the prefix and rollout.
    previous = torch.zeros(2, 2, 3, dtype=torch.float64)
    previous[..., 0] = previous.new_tensor([[10.0, 20.0], [12.0, 22.0]])
    previous.requires_grad_()
    previous_target = torch.zeros_like(previous)
    previous_target[..., 0] = previous.new_tensor([[3.0, 5.0], [5.0, 7.0]])
    previous_target.requires_grad_()
    velocity = torch.zeros(4, 2, 3, dtype=torch.float64)
    velocity[..., 0] = velocity.new_tensor([[2.0, 100.0], [4.0, 200.0], [6.0, 300.0], [8.0, 400.0]])
    velocity.requires_grad_()
    target = torch.zeros_like(velocity)
    target[..., 0] = velocity.new_tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    episode = torch.tensor([[0, 1], [0, 1], [1, 1], [1, 1]])
    prefix = {
        "velocity": previous,
        "target_velocity": previous_target,
        "episode_id": torch.zeros(2, 2, dtype=torch.long),
    }
    cfg = _velocity_only_cfg(variant=variant, velocity_window_s=0.2)
    assert cfg.velocity_loss_mode == "full_window" and cfg.resolved_velocity_loss_mode == "window"
    done = torch.tensor([[False, False], [True, False], [False, False], [False, False]])
    losses, cache, cost = diffnav_transition_costs(_history(velocity, target, episode, done), cfg, prefix)
    # The legacy integer-count division rounds H*N/K in float32 even for
    # float64 trajectories. Preserve that existing numerical contract.
    legacy_scale = torch.tensor(8 / 6, dtype=torch.float32).to(velocity)
    unscaled = velocity.new_tensor([[1.5, 0.0], [1.5, 139.5], [0.0, 229.5], [3.5, 319.5]])
    expected = unscaled * legacy_scale
    torch.testing.assert_close(cost, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(losses["velocity"], expected.mean(), rtol=1e-12, atol=1e-12)
    assert losses["velocity_valid_samples"] == 6 and losses["velocity_sample_fraction"] == 0.75
    gradients = torch.autograd.grad(losses["total"], (velocity, previous, previous_target), allow_unused=True)
    expected_gradient = torch.zeros_like(velocity)
    # Each scored two-sample average contributes 1/2 before the same rounded
    # endpoint scale and the eight-transition mean are applied.
    expected_gradient[..., 0] = velocity.new_tensor([[1.0, 0.5], [0.5, 1.0], [0.5, 1.0], [0.5, 0.5]]) * legacy_scale / 8
    torch.testing.assert_close(gradients[0], expected_gradient, rtol=1e-12, atol=1e-12)
    assert gradients[1:] == (None, None)
    torch.testing.assert_close(cache["velocity"], velocity[-2:].detach(), rtol=0, atol=0)
    torch.testing.assert_close(cache["target_velocity"], target[-2:], rtol=0, atol=0)
    assert all(not value.requires_grad for value in cache.values())


def _rich_history():
    velocity = torch.tensor(
        [[[2.0, 0, 0], [0, 0.5, 0]], [[0, 3.0, 0], [0, 0, 4.0]], [[1.0, 2.0, 0], [0, 0, 2.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    history = _history(
        velocity,
        episode=torch.tensor([[0, 0], [0, 1], [1, 1]]),
        done=torch.tensor([[False, True], [True, False], [False, False]]),
    )
    vectors = velocity.new_zeros(3, 2, 10, 3)
    vectors[..., 0] = velocity.new_tensor([[0.8, 1.5], [0.5, 0.9], [2.0, 1.2]])[..., None] + torch.linspace(
        0.1, -0.1, 10, dtype=velocity.dtype
    )
    vectors.requires_grad_()
    gap = velocity.new_tensor([[0.02, 0.4], [-0.1, 1.0], [0.3, -0.02]])[..., None].expand(-1, -1, 9).clone()
    gap.requires_grad_()
    command = velocity.new_tensor(
        [[[0.1, 0, 0], [0.2, 0, 0]], [[0.4, 0, 0], [0.5, 0, 0]], [[0.3, 0, 0], [0.9, 0, 0]]]
    ).requires_grad_()
    prediction = (command.detach() * 3).requires_grad_()
    position = torch.zeros_like(velocity)
    position[..., 2] = velocity.new_tensor([[0.4, 2.0], [3.8, 2.0], [-1.0, 5.0]])
    position.requires_grad_()
    for name, value in (
        ("obstacle_vector", vectors),
        ("dcol_gap", gap),
        ("command", command),
        ("velocity_prediction", prediction),
        ("position", position),
    ):
        history[name] = list(value.unbind(0))
    history["obstacle_signed_distance"] = list(vectors.norm(dim=-1).unbind(0))
    return velocity, history, (vectors, gap, command, prediction, position)


def test_step_variant_default_and_override_are_exactly_identical():
    velocity, history, _ = _rich_history()
    cfg = DiffNavTrainCfg(physics_dt=0.01, policy_dt=0.1, velocity_loss_mode="variant_default", variant="step")
    default, default_cache, default_cost = diffnav_transition_costs(history, cfg)
    explicit, explicit_cache, explicit_cost = diffnav_transition_costs(
        history,
        replace(cfg, velocity_loss_mode="per_step"),
    )
    assert default_cache is explicit_cache is None
    for name in default:
        torch.testing.assert_close(default[name], explicit[name], rtol=0, atol=0)
    torch.testing.assert_close(default_cost, explicit_cost, rtol=0, atol=0)
    default_gradient = torch.autograd.grad(default["total"], velocity, retain_graph=True)[0]
    explicit_gradient = torch.autograd.grad(explicit["total"], velocity)[0]
    torch.testing.assert_close(default_gradient, explicit_gradient, rtol=0, atol=0)


def test_dcol_override_preserves_all_six_other_costs_and_their_gradients():
    velocity, history, geometry_and_control = _rich_history()
    cfg = DiffNavTrainCfg(physics_dt=0.01, policy_dt=0.1, variant="dcol", velocity_loss_mode="full_window")
    original, _, original_cost = diffnav_transition_costs(history, cfg)
    explicit, cache, explicit_cost = diffnav_transition_costs(history, replace(cfg, velocity_loss_mode="per_step"))
    assert original["velocity"] == 0  # Three transitions cannot fill W20.
    assert cache is None and explicit["velocity_sample_fraction"] == 1
    for name in ("velocity_prediction", "obstacle_avoidance", "collision", "acceleration", "jerk", "height_boundary"):
        torch.testing.assert_close(original[name], explicit[name], rtol=0, atol=0)
    per_step = F.smooth_l1_loss(velocity.norm(dim=-1), torch.zeros_like(velocity[..., 0]), reduction="none")
    torch.testing.assert_close(explicit_cost - original_cost, cfg.coef_v * per_step, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(explicit["total"], explicit_cost.mean(), rtol=0, atol=0)
    original_gradients = torch.autograd.grad(original["total"], geometry_and_control, retain_graph=True)
    explicit_gradients = torch.autograd.grad(explicit["total"], geometry_and_control)
    for first, second in zip(original_gradients, explicit_gradients, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
