"""Temporal critic features preserve only the intended rollout derivatives."""

import torch

from aerialnav.diffnav.critic_state import CriticHistory, critic_observation
from aerialnav.diffnav.dynamics import PointMassState
from aerialnav.tasks.navigation.randomization import DifferentiableDelay


def test_window_reset_cuts_only_affected_environment_and_rollout_detach_cuts_old_graph():
    velocity = torch.ones(2, 3, requires_grad=True)
    history = CriticHistory(3, velocity)
    history.push(velocity, torch.ones_like(velocity), torch.tensor([True, False]))
    history.features().sum().backward()
    assert not velocity.grad[0].any() and velocity.grad[1].abs().sum() > 0
    history.detach()
    next_velocity = torch.ones(2, 3, requires_grad=True)
    history.push(next_velocity, torch.ones_like(next_velocity), torch.zeros(2, dtype=torch.bool))
    old_gradient = velocity.grad.clone()
    history.features().sum().backward()
    torch.testing.assert_close(velocity.grad, old_gradient, rtol=0, atol=0)
    assert next_velocity.grad.abs().sum() > 0


def test_value_state_retains_position_hidden_and_pending_action_but_not_sensor_target_gradients():
    position = torch.ones(2, 3, requires_grad=True)
    hidden = torch.ones(2, 192, requires_grad=True)
    pending = torch.ones(2, 3, requires_grad=True)
    zero = torch.zeros_like(position)
    state = PointMassState(position, zero, zero, zero)
    depth = torch.ones(2, 1, 12, 16, requires_grad=True)
    target = torch.full((2, 3), 3.0, requires_grad=True)
    observation = {
        "target": target,
        "depth_features": depth,
        "quaternion": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        "input_delay_steps": torch.zeros(2, dtype=torch.long),
    }
    action_delay, observation_delay = DifferentiableDelay(2), DifferentiableDelay(5)
    action_delay.push(pending, torch.zeros(2, dtype=torch.long))
    features = critic_observation(
        state,
        observation,
        hidden,
        zero,
        {"margin": torch.full((2,), 0.3)},
        CriticHistory(20, position),
        action_delay,
        observation_delay,
        torch.zeros(2),
        400,
        torch.zeros(2, dtype=torch.long),
    )
    features.sum().backward()
    assert position.grad.abs().sum() > 0 and hidden.grad.abs().sum() > 0 and pending.grad.abs().sum() > 0
    assert depth.grad is None and target.grad is None
