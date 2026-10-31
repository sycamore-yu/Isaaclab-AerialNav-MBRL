"""Terminal window completion preserves episode timing and the original full windows."""

from dataclasses import replace

import pytest
import torch

from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.diffnav.dynamics import PointMassState, align_transition
from aerialnav.diffnav.losses import diffnav_transition_costs


def _cfg(window=4, **overrides):
    values = {
        "physics_dt": 0.01,
        "policy_dt": 0.1,
        "variant": "dcol",
        "velocity_loss_mode": "truncated_window",
        "velocity_window_s": window * 0.1,
        "coef_v_pred": 0.0,
        "coef_obj_avoidance": 0.0,
        "coef_collide": 0.0,
        "coef_d_acc": 0.0,
        "coef_d_jerk": 0.0,
        "coef_height_boundary": 0.0,
    }
    values.update(overrides)
    return DiffNavTrainCfg(**values)


def _history(velocity, episode_step, *, target=None, episode=None, **events):
    count, batch, _ = velocity.shape
    step = torch.as_tensor(episode_step, dtype=torch.long, device=velocity.device)
    episode = torch.zeros_like(step) if episode is None else torch.as_tensor(episode, device=velocity.device)
    target = torch.zeros_like(velocity) if target is None else target
    zero = torch.zeros_like(velocity)
    position = zero.clone()
    position[..., 2] = 2.0
    vector = velocity.new_zeros(count, batch, 10, 3)
    vector[..., 0] = 10.0
    history = {
        "velocity": list(velocity.unbind(0)),
        "target_velocity": list(target.unbind(0)),
        "episode_id": list(episode.unbind(0)),
        "episode_step": list(step.unbind(0)),
        "position": list(position.unbind(0)),
        "obstacle_signed_distance": list(vector.norm(dim=-1).unbind(0)),
        "dcol_gap": [velocity.new_full((batch, 9), 9.7)] * count,
        "height_target": [velocity.new_full((batch,), 2.0)] * count,
        "height_range": [velocity.new_full((batch, 2), 2.0)] * count,
        "margin": [velocity.new_full((batch,), 0.3)] * count,
        "command": list(zero.unbind(0)),
        "previous_command": list(zero.unbind(0)),
        "velocity_prediction": list(zero.unbind(0)),
        "observed_velocity": list(zero.unbind(0)),
    }
    for name in ("done", "collision", "bounds", "timeout", "invalid"):
        value = events.get(name, torch.zeros(count, batch, dtype=torch.bool, device=velocity.device))
        history[name] = list(torch.as_tensor(value, dtype=torch.bool, device=velocity.device).unbind(0))
    return history


@pytest.mark.parametrize("reason", ("collision", "bounds", "timeout"))
def test_short_terminal_uses_actual_length_start_target_and_one_episode(reason):
    velocity = torch.tensor([[[2.0, 0, 0]], [[4.0, 0, 0]], [[50.0, 0, 0]]], dtype=torch.float64, requires_grad=True)
    target = velocity.new_tensor([[[1.0, 0, 0]], [[100.0, 0, 0]], [[0.0, 0, 0]]])
    terminal = [[False], [True], [False]]
    history = _history(
        velocity, [[1], [2], [1]], target=target, episode=[[0], [0], [1]], done=terminal, **{reason: terminal}
    )
    losses, _, cost = diffnav_transition_costs(history, _cfg())
    # mean(2,4)-target_start(1)=2, SmoothL1=1.5; only its terminal
    # endpoint is charged H*N/K=3. The new episode's 50 never participates.
    torch.testing.assert_close(cost, velocity.new_tensor([[0.0], [4.5], [0.0]]), rtol=0, atol=0)
    torch.testing.assert_close(losses["velocity"], velocity.new_tensor(1.5), rtol=0, atol=0)
    assert losses["velocity_short_terminal_samples"] == 1 and losses["velocity_full_window_samples"] == 0
    assert losses["velocity_valid_samples"] == 1
    gradient = torch.autograd.grad(losses["total"], velocity)[0]
    expected = torch.zeros_like(velocity)
    expected[:2, :, 0] = 0.5
    torch.testing.assert_close(gradient, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "events,count",
    [
        ({"collision": [[True]]}, 0),
        ({"done": [[True]]}, 0),
        ({"done": [[True]], "collision": [[True]], "invalid": [[True]]}, 0),
        ({"done": [[True]], "bounds": [[True]], "timeout": [[True]], "collision": [[True]]}, 1),
    ],
)
def test_terminal_requires_real_reset_and_eligible_reason_without_invalid(events, count):
    velocity = torch.tensor([[[2.0, 0, 0]]], dtype=torch.float64, requires_grad=True)
    losses, _, cost = diffnav_transition_costs(_history(velocity, [[1]], **events), _cfg())
    assert losses["velocity_short_terminal_samples"] == count
    torch.testing.assert_close(cost, velocity.new_tensor([[count * 1.5]]), rtol=0, atol=0)


def test_full_and_short_samples_share_denominator_and_keep_their_endpoints():
    velocity = torch.zeros(4, 2, 3, dtype=torch.float64)
    velocity[..., 0] = velocity.new_tensor([[1.0, 4.0], [2.0, 8.0], [3.0, 100.0], [4.0, 200.0]])
    velocity.requires_grad_()
    history = _history(
        velocity,
        [[1, 1], [2, 2], [3, 1], [4, 2]],
        episode=[[0, 0], [0, 0], [0, 1], [0, 1]],
        done=[[False, False], [False, True], [False, False], [False, False]],
        collision=[[False, False], [False, True], [False, False], [False, False]],
    )
    losses, _, cost = diffnav_transition_costs(history, _cfg(window=3))
    # Full loss 2.5 and short loss 5.5 share K=2, not two separate means.
    expected = velocity.new_tensor([[0.0, 0.0], [0.0, 22.0], [0.0, 0.0], [10.0, 0.0]])
    torch.testing.assert_close(cost, expected, rtol=0, atol=0)
    torch.testing.assert_close(losses["total"], velocity.new_tensor(4.0), rtol=0, atol=0)
    torch.testing.assert_close(losses["total"], cost.mean(), rtol=0, atol=0)
    assert losses["velocity_full_window_samples"] == losses["velocity_short_terminal_samples"] == 1
    assert losses["velocity_valid_samples"] == 2 and losses["velocity_sample_fraction"] == 0.25
    expected_gradient = torch.zeros_like(velocity)
    expected_gradient[1:, 0, 0] = 1 / 6
    expected_gradient[:2, 1, 0] = 0.25
    torch.testing.assert_close(
        torch.autograd.grad(losses["total"], velocity)[0], expected_gradient, rtol=1e-12, atol=1e-12
    )


def test_short_episode_crosses_update_with_detached_prefix_and_no_update_end_score():
    previous = torch.tensor([[[1.0, 0, 0]], [[3.0, 0, 0]]], dtype=torch.float64, requires_grad=True)
    first, cache, _ = diffnav_transition_costs(_history(previous, [[1], [2]]), _cfg())
    assert first["velocity_valid_samples"] == 0 and first["velocity_short_terminal_samples"] == 0
    assert all(not value.requires_grad for value in cache.values())
    # Also enforce detach on consumption, even if a caller supplies a live prefix.
    cache["velocity"] = previous
    previous_target = torch.zeros_like(previous, requires_grad=True)
    cache["target_velocity"] = previous_target
    current = torch.tensor([[[5.0, 0, 0]], [[80.0, 0, 0]]], dtype=torch.float64, requires_grad=True)
    terminal = [[True], [False]]
    history = _history(current, [[3], [1]], episode=[[0], [1]], done=terminal, bounds=terminal)
    losses, _, cost = diffnav_transition_costs(history, _cfg(), cache)
    torch.testing.assert_close(cost, current.new_tensor([[5.0], [0.0]]), rtol=0, atol=0)
    torch.testing.assert_close(losses["velocity"], current.new_tensor(2.5), rtol=0, atol=0)
    gradient, old_gradient, old_target_gradient = torch.autograd.grad(
        losses["total"],
        (current, previous, previous_target),
        allow_unused=True,
    )
    torch.testing.assert_close(gradient, current.new_tensor([[[1 / 3, 0, 0]], [[0.0, 0, 0]]]), rtol=1e-12, atol=1e-12)
    assert old_gradient is None and old_target_gradient is None


def test_exact_first_window_terminal_completes_once_without_changing_source_offset():
    velocity = torch.tensor([[[1.0, 0, 0]], [[2.0, 0, 0]], [[3.0, 0, 0]]], dtype=torch.float64, requires_grad=True)
    terminal = [[False], [False], [True]]
    history = _history(velocity, [[1], [2], [3]], done=terminal, collision=terminal)
    cfg = _cfg(window=3)
    original, _, _ = diffnav_transition_costs(history, replace(cfg, velocity_loss_mode="full_window"))
    complete, _, cost = diffnav_transition_costs(history, cfg)
    assert original["velocity_valid_samples"] == 0  # Original offset is retained.
    assert complete["velocity_full_window_samples"] == 0 and complete["velocity_short_terminal_samples"] == 1
    torch.testing.assert_close(cost, velocity.new_tensor([[0.0], [0.0], [4.5]]), rtol=0, atol=0)
    expected_gradient = torch.zeros_like(velocity)
    expected_gradient[..., 0] = 1 / 3
    torch.testing.assert_close(
        torch.autograd.grad(complete["total"], velocity)[0], expected_gradient, rtol=1e-12, atol=1e-12
    )
    # A later exactly-W episode already has a full score. The previous invalid
    # terminal is deliberately ineligible, and neither endpoint is double counted.
    later = torch.cat((velocity.new_tensor([[[9.0, 0, 0]]]), velocity.detach())).requires_grad_()
    history = _history(
        later,
        [[1], [1], [2], [3]],
        episode=[[0], [1], [1], [1]],
        done=[[True], [False], [False], [True]],
        invalid=[[True], [False], [False], [False]],
        bounds=[[False], [False], [False], [True]],
    )
    original, _, original_cost = diffnav_transition_costs(history, replace(cfg, velocity_loss_mode="full_window"))
    complete, _, cost = diffnav_transition_costs(history, cfg)
    assert complete["velocity_full_window_samples"] == 1 and complete["velocity_short_terminal_samples"] == 0
    torch.testing.assert_close(cost, original_cost, rtol=0, atol=0)
    first_gradient = torch.autograd.grad(original["total"], later, retain_graph=True)[0]
    next_gradient = torch.autograd.grad(complete["total"], later)[0]
    torch.testing.assert_close(first_gradient, next_gradient, rtol=0, atol=0)


def test_long_episode_with_bounded_prefix_is_not_misclassified_as_short():
    cfg = _cfg()
    cache = None
    for start in range(1, 10, 2):
        velocity = torch.full((2, 1, 3), 2.0, dtype=torch.float64, requires_grad=True)
        terminal = [[False], [start == 9]]
        history = _history(velocity, [[start], [start + 1]], done=terminal, timeout=terminal)
        original, _, original_cost = diffnav_transition_costs(
            history,
            replace(cfg, velocity_loss_mode="full_window"),
            cache,
        )
        complete, cache, cost = diffnav_transition_costs(history, cfg, cache)
        assert complete["velocity_short_terminal_samples"] == 0
        torch.testing.assert_close(cost, original_cost, rtol=0, atol=0)
        torch.testing.assert_close(complete["velocity"], original["velocity"], rtol=0, atol=0)
        assert cache["velocity"].shape[0] <= 4
    assert complete["velocity_full_window_samples"] == 2


@pytest.mark.parametrize("steps,episode", [([[3]], [[0]]), ([[1], [2]], [[0], [1]])])
def test_terminal_completion_rejects_missing_episode_samples(steps, episode):
    count = len(steps)
    velocity = torch.ones(count, 1, 3, dtype=torch.float64)
    terminal = torch.zeros(count, 1, dtype=torch.bool)
    terminal[-1] = True
    history = _history(velocity, steps, episode=episode, done=terminal, collision=terminal)
    with pytest.raises(ValueError, match="episode_step disagrees"):
        diffnav_transition_costs(history, _cfg())


def test_truncated_requires_terminal_metadata_only_for_its_own_mode():
    velocity = torch.ones(2, 1, 3, dtype=torch.float64)
    history = _history(velocity, [[1], [2]])
    del history["timeout"]
    with pytest.raises(ValueError, match="requires terminal history"):
        diffnav_transition_costs(history, _cfg())
    diffnav_transition_costs(history, _cfg(velocity_loss_mode="full_window"))
    diffnav_transition_costs(history, _cfg(velocity_loss_mode="per_step"))


def test_terminal_velocity_alignment_excludes_the_reset_state():
    predicted_velocity = torch.tensor([[0.5, 0.7, 0.9]], dtype=torch.float64, requires_grad=True)
    zero = torch.zeros_like(predicted_velocity)
    predicted = PointMassState(zero, predicted_velocity, zero, zero)
    measured_velocity = predicted_velocity.new_tensor([[4.0, 0, 0]])
    reset_velocity = predicted_velocity.new_tensor([[100.0, 20.0, 0]], requires_grad=True)
    terminal, _ = align_transition(
        predicted, zero, measured_velocity, zero, torch.tensor([True]), zero, reset_velocity, 0.1, zero
    )
    history = _history(
        terminal.velocity[None],
        [[1]],
        target=predicted_velocity.new_tensor([[[1.0, 0, 0]]]),
        done=[[True]],
        collision=[[True]],
    )
    losses, _, cost = diffnav_transition_costs(history, _cfg())
    torch.testing.assert_close(cost, predicted_velocity.new_tensor([[2.5]]), rtol=0, atol=0)
    derivative, reset_derivative = torch.autograd.grad(
        losses["total"], (predicted_velocity, reset_velocity), allow_unused=True
    )
    torch.testing.assert_close(derivative, predicted_velocity.new_tensor([[1.0, 0, 0]]), rtol=0, atol=0)
    assert reset_derivative is None


def test_other_six_costs_and_geometry_gradients_remain_exactly_unchanged():
    velocity = torch.tensor([[[2.0, 0, 0]], [[4.0, 0, 0]]], dtype=torch.float64, requires_grad=True)
    history = _history(velocity, [[1], [2]], done=[[False], [True]], bounds=[[False], [True]])
    vector = velocity.new_zeros(2, 1, 10, 3)
    vector[..., 0] = torch.linspace(1.2, 0.8, 10, dtype=velocity.dtype)
    vector.requires_grad_()
    gap = velocity.new_tensor([[-0.02], [0.2]])[..., None].expand(-1, -1, 9).clone().requires_grad_()
    command = velocity.new_tensor([[[0.1, 0.2, 0]], [[0.3, 0.2, 0.1]]]).requires_grad_()
    prediction = velocity.new_tensor([[[0.4, 0.1, 0]], [[0.3, 0.9, 0.7]]]).requires_grad_()
    position = velocity.new_tensor([[[0.0, 0, 3.8]], [[0.0, 0, 4.1]]]).requires_grad_()
    variables = (vector, gap, command, prediction, position)
    for name, value in zip(
        ("obstacle_vector", "dcol_gap", "command", "velocity_prediction", "position"), variables, strict=True
    ):
        history[name] = list(value.unbind(0))
    history["obstacle_signed_distance"] = list(vector.norm(dim=-1).unbind(0))
    cfg = DiffNavTrainCfg(
        physics_dt=0.01, policy_dt=0.1, variant="dcol", velocity_loss_mode="truncated_window", velocity_window_s=0.4
    )
    original, _, original_cost = diffnav_transition_costs(history, replace(cfg, velocity_loss_mode="full_window"))
    complete, _, cost = diffnav_transition_costs(history, cfg)
    for name in ("velocity_prediction", "obstacle_avoidance", "collision", "acceleration", "jerk", "height_boundary"):
        torch.testing.assert_close(complete[name], original[name], rtol=0, atol=0)
    torch.testing.assert_close(cost - original_cost, velocity.new_tensor([[0.0], [5.0]]), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(complete["total"], cost.mean(), rtol=0, atol=0)
    original_gradients = torch.autograd.grad(original["total"], variables, retain_graph=True)
    complete_gradients = torch.autograd.grad(complete["total"], variables)
    for first, second in zip(original_gradients, complete_gradients, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
