"""Episode targets, terminal alignment and signed inputs for DiffNav transition costs."""

import math
from dataclasses import replace

import pytest
import torch

from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.diffnav.dynamics import PointMassState, align_transition
from aerialnav.diffnav.losses import diffnav_transition_costs, height_target_loss


def _history(position, height_target):
    count, batch, _ = position.shape
    zero = torch.zeros_like(position)
    events = torch.zeros(count, batch, dtype=torch.bool, device=position.device)
    history = {
        "position": list(position.unbind(0)),
        "height_target": list(height_target.unbind(0)),
        "velocity": list(zero.unbind(0)),
        "target_velocity": list(zero.unbind(0)),
        "velocity_prediction": list(zero.unbind(0)),
        "observed_velocity": list(zero.unbind(0)),
        "command": list(zero.unbind(0)),
        "previous_command": list(zero.unbind(0)),
        "episode_id": list(torch.zeros_like(events, dtype=torch.long).unbind(0)),
        "episode_step": list(torch.arange(1, count + 1, device=position.device)[:, None].expand(-1, batch).unbind(0)),
        "obstacle_signed_distance": [position.new_full((batch, 10), 10.0)] * count,
        "collision_gap": [position.new_full((batch, 9), 9.7)] * count,
        "margin": [position.new_full((batch,), 0.3)] * count,
    }
    for name in ("done", "collision", "bounds", "timeout", "invalid"):
        history[name] = list(events.unbind(0))
    return history


def _height_only_cfg(**overrides):
    values = {
        "variant": "step",
        "coef_v": 0.0,
        "coef_v_pred": 0.0,
        "height_loss_mode": "target_exp",
        "coef_obj_avoidance": 0.0,
        "coef_collide": 0.0,
        "coef_d_acc": 0.0,
        "coef_d_jerk": 0.0,
    }
    values.update(overrides)
    return DiffNavTrainCfg(**values)


def test_target_exp_matches_requested_formula_and_has_no_added_far_field_term():
    errors = [-1000.0, -2.0, -1e-10, 0.0, 1e-10, 0.5, 2.0, 1000.0]
    position = torch.tensor([[3.0, -2.0, value] for value in errors], dtype=torch.float64, requires_grad=True)
    target = torch.zeros(len(errors), dtype=torch.float64, requires_grad=True)
    values = height_target_loss(position, target, reduction="none")
    expected = position.new_tensor([-math.expm1(-abs(value)) for value in errors])
    torch.testing.assert_close(values, expected, rtol=1e-14, atol=0)
    torch.testing.assert_close(height_target_loss(position, target), expected.mean(), rtol=1e-14, atol=0)
    torch.testing.assert_close(height_target_loss(position, target, reduction="sum"), expected.sum())
    gradient, target_gradient = torch.autograd.grad(values.sum(), (position, target), allow_unused=True)
    expected_gradient = torch.zeros_like(position)
    expected_gradient[:, 2] = position.new_tensor(
        [math.copysign(math.exp(-abs(value)), value) if value else 0.0 for value in errors]
    )
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-14, atol=0)
    assert target_gradient is None
    assert values[0] == values[-1] == 1 and gradient[0, 2] == gradient[-1, 2] == 0


def test_target_huber_uses_a_one_meter_shoulder_and_detaches_the_target():
    errors = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    position = torch.tensor([[0.0, 0.0, value + 1.5] for value in errors], dtype=torch.float64, requires_grad=True)
    target = position.new_full((len(errors),), 1.5, requires_grad=True)
    values = height_target_loss(position, target, "target_huber", reduction="none")
    torch.testing.assert_close(values, position.new_tensor([1.5, 0.5, 0.125, 0.0, 0.125, 0.5, 1.5]), rtol=0, atol=0)
    gradient, target_gradient = torch.autograd.grad(values.sum(), (position, target), allow_unused=True)
    expected_gradient = torch.zeros_like(position)
    expected_gradient[:, 2] = position.new_tensor([-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0])
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    assert target_gradient is None


def test_terminal_height_keeps_old_episode_target_and_gradient_before_autoreset():
    predicted_position = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    zero = torch.zeros_like(predicted_position)
    prediction = PointMassState(predicted_position, zero, zero, zero)
    measured_position = predicted_position.new_tensor([[0.0, 0.0, 0.25], [0.0, 0.0, 3.9]])
    reset_position = predicted_position.new_tensor([[0.0, 0.0, 2.4], [0.0, 0.0, 0.6]], requires_grad=True)
    episode_target_buffer = predicted_position.new_tensor([0.5, 2.5], requires_grad=True)
    target_snapshot = episode_target_buffer.detach().clone()
    terminal, following = align_transition(
        prediction,
        measured_position,
        zero,
        zero,
        torch.ones(2, dtype=torch.bool),
        reset_position,
        zero,
        0.1,
    )
    with torch.no_grad():
        episode_target_buffer.copy_(predicted_position.new_tensor([2.4, 0.6]))
    history = _history(terminal.position[None], target_snapshot[None])
    history["done"] = history["collision"] = [torch.ones(2, dtype=torch.bool)]
    losses, _, cost = diffnav_transition_costs(history, _height_only_cfg())
    expected = predicted_position.new_tensor([[-math.expm1(-0.25), -math.expm1(-1.4)]])
    torch.testing.assert_close(cost, expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(following.position, reset_position, rtol=0, atol=0)
    predicted_gradient, reset_gradient, target_gradient = torch.autograd.grad(
        losses["total"],
        (predicted_position, reset_position, episode_target_buffer),
        retain_graph=True,
        allow_unused=True,
    )
    expected_gradient = torch.zeros_like(predicted_position)
    expected_gradient[:, 2] = predicted_position.new_tensor([-math.exp(-0.25), math.exp(-1.4)]) / 2
    torch.testing.assert_close(predicted_gradient, expected_gradient, rtol=1e-14, atol=1e-14)
    assert reset_gradient is None and target_gradient is None
    torch.testing.assert_close(
        torch.autograd.grad(following.position.sum(), predicted_position)[0],
        torch.zeros_like(predicted_position),
        rtol=0,
        atol=0,
    )


def test_boundary_mode_keeps_the_previous_values_without_requiring_a_target():
    position = torch.tensor(
        [[[0.0, 0.0, 0.4], [0.0, 0.0, 2.0]], [[0.0, 0.0, 3.8], [0.0, 0.0, 5.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    history = _history(position, position.new_zeros(2, 2))
    del history["height_target"]
    losses, _, cost = diffnav_transition_costs(history, _height_only_cfg(height_loss_mode="boundary_huber"))
    torch.testing.assert_close(cost, position.new_tensor([[0.18, 0.0], [0.18, 2.5]]), rtol=0, atol=1e-14)
    expected_gradient = torch.zeros_like(position)
    expected_gradient[..., 2] = position.new_tensor([[-1.2, 0.0], [1.2, 2.0]]) / 4
    torch.testing.assert_close(torch.autograd.grad(losses["total"], position)[0], expected_gradient, rtol=0, atol=1e-14)


@pytest.mark.parametrize("velocity_mode", ("variant_default", "per_step", "truncated_window"))
def test_height_is_scored_on_every_transition_independent_of_velocity_windows(velocity_mode):
    position = torch.zeros(3, 2, 3, dtype=torch.float64)
    position[..., 2] = position.new_tensor([[0.0, 3.0], [0.5, 3.1], [1.8, 0.2]])
    position.requires_grad_()
    target = position.new_tensor([[0.5, 2.5], [0.5, 2.5], [2.5, 2.5]])
    history = _history(position, target)
    history["episode_id"] = list(torch.tensor([[0, 10], [0, 10], [1, 10]]).unbind(0))
    history["episode_step"] = list(torch.tensor([[1, 1], [2, 2], [1, 3]]).unbind(0))
    done = torch.tensor([[False, False], [True, False], [False, True]])
    history["done"] = history["collision"] = list(done.unbind(0))
    cfg = _height_only_cfg(variant="terminal", velocity_loss_mode=velocity_mode, velocity_window_s=2.0)
    losses, _, cost = diffnav_transition_costs(history, cfg)
    error = position.detach()[..., 2] - target
    expected = -torch.expm1(-error.abs())
    torch.testing.assert_close(cost, expected, rtol=0, atol=0)
    expected_gradient = torch.zeros_like(position)
    expected_gradient[..., 2] = error.sign() * torch.exp(-error.abs()) / 6
    torch.testing.assert_close(
        torch.autograd.grad(losses["total"], position)[0], expected_gradient, rtol=1e-14, atol=1e-14
    )


def test_height_mode_changes_only_its_term_and_preserves_other_cost_gradients():
    position = torch.zeros(3, 2, 3, dtype=torch.float64)
    position[..., 2] = position.new_tensor([[0.4, 2.0], [3.8, 1.0], [5.0, -1.0]])
    position.requires_grad_()
    target = position.new_tensor([[0.5, 2.5], [0.5, 2.5], [1.5, 2.5]])
    history = _history(position, target)
    velocity = (torch.arange(18, dtype=position.dtype).reshape(3, 2, 3) / 10).requires_grad_()
    command = (torch.arange(18, dtype=position.dtype).reshape(3, 2, 3) / 100).requires_grad_()
    prediction = (velocity.detach() / 2).requires_grad_()
    signed = torch.linspace(2.0, -0.5, 60, dtype=position.dtype).reshape(3, 2, 10).requires_grad_()
    history.update(
        {
            "velocity": list(velocity.unbind(0)),
            "command": list(command.unbind(0)),
            "velocity_prediction": list(prediction.unbind(0)),
            "obstacle_signed_distance": list(signed.unbind(0)),
            "collision_gap": list((signed[..., 1:] - 0.3).unbind(0)),
        }
    )
    cfg = DiffNavTrainCfg(
        variant="terminal", velocity_loss_mode="per_step", coef_height_boundary=0.7, height_loss_mode="target_exp"
    )
    baseline, _, baseline_cost = diffnav_transition_costs(history, cfg)
    nonheight_inputs = (velocity, command, prediction, signed)
    baseline_gradients = torch.autograd.grad(baseline["total"], nonheight_inputs, retain_graph=True)
    for mode in ("target_huber", "boundary_huber"):
        selected, _, selected_cost = diffnav_transition_costs(history, replace(cfg, height_loss_mode=mode))
        for name in ("velocity", "velocity_prediction", "obstacle_avoidance", "collision", "acceleration", "jerk"):
            torch.testing.assert_close(selected[name], baseline[name], rtol=0, atol=0)
        torch.testing.assert_close(
            (selected_cost - baseline_cost).mean(),
            cfg.coef_height_boundary * (selected["height_boundary"] - baseline["height_boundary"]),
            rtol=1e-12,
            atol=1e-12,
        )
        gradients = torch.autograd.grad(selected["total"], nonheight_inputs, retain_graph=True)
        for actual, expected in zip(gradients, baseline_gradients, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ("step", "continuous", "terminal"))
def test_signed_obstacle_distance_is_used_without_unsigned_or_ground_fallback(variant):
    position = torch.zeros(1, 2, 3, dtype=torch.float64)
    position[..., 2] = 2.0
    history = _history(position, position.new_full((1, 2), 2.0))
    signed = position.new_tensor([[[0.2] * 10, [-0.2] * 10]], requires_grad=True)
    # Deliberately contradictory legacy inputs must not enter the loss graph.
    unsigned_vector = position.new_full((1, 2, 10, 3), 100.0, requires_grad=True)
    legacy_ground = position.new_full((1, 2, 10), -100.0, requires_grad=True)
    history.update(
        {
            "obstacle_signed_distance": list(signed.unbind(0)),
            "obstacle_vector": list(unsigned_vector.unbind(0)),
            "ground_signed_distance": list(legacy_ground.unbind(0)),
            "collision_gap": list((signed[..., 1:] - 0.3).unbind(0)),
        }
    )
    cfg = _height_only_cfg(variant=variant, coef_obj_avoidance=1.0, coef_collide=1.0)
    losses, _, cost = diffnav_transition_costs(history, cfg)
    gap = [-0.1, -0.5]
    avoidance = position.new_tensor([[(1.0 - value) ** 2 for value in gap]])
    collision = position.new_tensor([[math.log1p(math.exp(-32 * value)) for value in gap]])
    torch.testing.assert_close(cost, avoidance + collision, rtol=1e-12, atol=1e-12)
    actual, unsigned_gradient, legacy_gradient = torch.autograd.grad(
        losses["obstacle_avoidance"],
        (signed, unsigned_vector, legacy_ground),
        allow_unused=True,
    )
    expected = torch.zeros_like(signed)
    expected[:, :, 1:] = position.new_tensor([-2.2, -3.0])[None, :, None] / 18
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    assert unsigned_gradient is None and legacy_gradient is None


def test_missing_or_misaligned_targets_and_signed_samples_fail_explicitly():
    position = torch.zeros(2, 1, 3, dtype=torch.float64)
    history = _history(position, position.new_full((2, 1), 1.5))
    cfg = _height_only_cfg(coef_height_boundary=0)
    for mode in ("target_exp", "target_huber"):
        missing = {name: value for name, value in history.items() if name != "height_target"}
        with pytest.raises(ValueError, match="pre-step episode height_target"):
            diffnav_transition_costs(missing, replace(cfg, height_loss_mode=mode))
    with pytest.raises(ValueError, match="leading sample dimensions"):
        diffnav_transition_costs(dict(history, height_target=[position.new_ones(1, 1)] * 2), cfg)
    with pytest.raises(ValueError, match="height_loss_mode"):
        diffnav_transition_costs(history, replace(cfg, height_loss_mode="unknown"))
    with pytest.raises(ValueError, match="require obstacle_signed_distance"):
        diffnav_transition_costs(
            {name: value for name, value in history.items() if name != "obstacle_signed_distance"}, cfg
        )
    with pytest.raises(ValueError, match=r"\[T, N, 10\]"):
        diffnav_transition_costs(dict(history, obstacle_signed_distance=[position.new_ones(1, 9)] * 2), cfg)
    with pytest.raises(ValueError, match="mode must"):
        height_target_loss(position, position.new_zeros(2, 1), "unknown")
    with pytest.raises(ValueError, match="reduction must"):
        height_target_loss(position, position.new_zeros(2, 1), reduction="unknown")
