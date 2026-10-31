"""Behavior checks for policy perturbations and differentiable action delays."""

import pytest
import torch

from aerialnav.tasks.navigation.randomization import (
    DifferentiableDelay,
    PolicyPerturbation,
    SimToRealCfg,
)


def _action_delay(num_envs: int, lags: torch.Tensor) -> PolicyPerturbation:
    cfg = SimToRealCfg(
        enabled=True,
        observation_noise=False,
        input_delay=False,
        action_delay=True,
        action_delay_range_s=(0, 0.1),
    )
    perturbation = PolicyPerturbation(cfg, num_envs, "cpu", seed=417, policy_dt=0.05)
    perturbation.reset(torch.arange(num_envs))
    perturbation.action_buffer.set_time_lag(lags)
    return perturbation


def _grad_or_zero(gradient: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(value) if gradient is None else gradient


def test_default_latency_in_seconds_and_observation_age_after_local_reset():
    perturbation = PolicyPerturbation(SimToRealCfg(enabled=True), 256, "cpu", 417, policy_dt=0.05)
    perturbation.reset(torch.arange(256))
    assert set(perturbation.observation_buffer.time_lags.tolist()) == {0, 1, 2}
    assert set(perturbation.action_buffer.time_lags.tolist()) == {0, 1}
    perturbation.observation_buffer.set_time_lag(2)
    for step, expected in enumerate((0.0, 0.05, 0.1, 0.1)):
        delivered = perturbation.observe(step, lambda step=step: torch.full((256, 1), float(step)))
        torch.testing.assert_close(perturbation.observation_age_s, torch.full((256,), expected))
        assert delivered[0, 0] == max(0, step - 2)
        assert perturbation.latest_observation[0, 0] == step
    perturbation.reset(torch.tensor([0]))
    perturbation.observation_buffer.set_time_lag(2)
    delivered = perturbation.observe(4, lambda: torch.full((256, 1), 4.0))
    assert perturbation.observation_age_s[0] == 0 and delivered[0, 0] == 4
    torch.testing.assert_close(perturbation.observation_age_s[1:], torch.full((255,), 0.1))
    assert delivered[1, 0] == 2


def test_action_delay_selects_same_commands_and_credits_historical_actions():
    """Each loss credits the command actually executed after a 0, 1, or 2 step lag."""
    lags = torch.tensor([0, 1, 2], dtype=torch.long)
    execution = _action_delay(3, lags)
    differentiable = DifferentiableDelay(history_length=2)
    commands = [torch.full((3, 3), float(10 * (step + 1)), requires_grad=True) for step in range(5)]
    expected_gradients = [torch.zeros_like(command) for command in commands]
    losses = []

    for step, command in enumerate(commands):
        executed = execution.execute(command)
        selected = differentiable.push(command, lags, zero_until_ready=True)
        expected = torch.zeros_like(command)
        weights = (step + 1) * torch.tensor([[1.0], [2.0], [3.0]])

        for env_id, lag in enumerate(lags.tolist()):
            if step >= lag:
                expected[env_id] = commands[step - lag].detach()[env_id]
                expected_gradients[step - lag][env_id] += weights[env_id]

        torch.testing.assert_close(executed, expected)
        torch.testing.assert_close(selected.detach(), executed)
        losses.append((selected * weights).sum())

    gradients = torch.autograd.grad(sum(losses), commands, allow_unused=True)
    for command, gradient, expected in zip(commands, gradients, expected_gradients, strict=True):
        torch.testing.assert_close(_grad_or_zero(gradient, command), expected)


def test_partial_reset_removes_previous_episode_values_and_gradients():
    """Resetting one environment preserves the other environment's pending command."""
    lags = torch.tensor([1, 1], dtype=torch.long)
    execution = _action_delay(2, lags)
    differentiable = DifferentiableDelay(history_length=2)
    old_first = torch.tensor([[10.0] * 3, [20.0] * 3], requires_grad=True)
    old_last = torch.tensor([[11.0] * 3, [21.0] * 3], requires_grad=True)

    for command in (old_first, old_last):
        execution.execute(command)
        differentiable.push(command, lags, zero_until_ready=True)

    execution.reset(torch.tensor([0]))
    execution.action_buffer.set_time_lag(lags)
    differentiable.reset(torch.tensor([True, False]))
    new_first = torch.tensor([[100.0] * 3, [200.0] * 3], requires_grad=True)
    new_second = torch.tensor([[101.0] * 3, [201.0] * 3], requires_grad=True)
    losses = []

    for command, expected in (
        (new_first, torch.tensor([[0.0] * 3, [21.0] * 3])),
        (new_second, torch.tensor([[100.0] * 3, [200.0] * 3])),
    ):
        executed = execution.execute(command)
        selected = differentiable.push(command, lags, zero_until_ready=True)
        torch.testing.assert_close(executed, expected)
        torch.testing.assert_close(selected.detach(), expected)
        losses.append(selected.sum())

    commands = (old_first, old_last, new_first, new_second)
    gradients = torch.autograd.grad(sum(losses), commands, allow_unused=True)
    expected_gradients = (
        torch.tensor([[0.0] * 3, [0.0] * 3]),
        torch.tensor([[0.0] * 3, [1.0] * 3]),
        torch.tensor([[1.0] * 3, [1.0] * 3]),
        torch.tensor([[0.0] * 3, [0.0] * 3]),
    )
    for command, gradient, expected in zip(commands, gradients, expected_gradients, strict=True):
        torch.testing.assert_close(_grad_or_zero(gradient, command), expected)


def test_repeated_observation_read_does_not_sample_or_advance_delay():
    cfg = SimToRealCfg(
        enabled=True,
        observation_noise=False,
        input_delay=True,
        action_delay=False,
        input_delay_range_s=(0.05, 0.05),
    )
    perturbation = PolicyPerturbation(cfg, 2, "cpu", seed=417, policy_dt=0.05)
    perturbation.reset(torch.arange(2))
    perturbation.observation_buffer.set_time_lag(1)
    sampled_values = []

    def sample(value: float) -> torch.Tensor:
        sampled_values.append(value)
        return torch.full((2, 3), value)

    first = perturbation.observe(10, lambda: sample(100.0))
    repeated = perturbation.observe(10, lambda: sample(900.0))
    next_step = perturbation.observe(11, lambda: sample(200.0))
    following_step = perturbation.observe(12, lambda: sample(300.0))

    assert sampled_values == [100.0, 200.0, 300.0]
    torch.testing.assert_close(first, torch.full((2, 3), 100.0))
    torch.testing.assert_close(repeated, torch.full((2, 3), 100.0))
    torch.testing.assert_close(next_step, torch.full((2, 3), 100.0))
    torch.testing.assert_close(following_step, torch.full((2, 3), 200.0))


@pytest.mark.parametrize("enabled,observation_noise", [(False, True), (True, False)])
def test_disabled_observation_noise_preserves_input(enabled, observation_noise):
    cfg = SimToRealCfg(enabled=enabled, observation_noise=observation_noise)
    perturbation = PolicyPerturbation(cfg, 2, "cpu", seed=417, policy_dt=0.05)
    position = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, 0.5]])
    velocity = torch.tensor([[0.1, -0.2, 0.3], [1.0, 2.0, -3.0]])
    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
    originals = tuple(value.clone() for value in (position, velocity, quaternion))

    observed = perturbation.state(position, velocity, quaternion)

    for actual, expected in zip(observed, originals, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip((position, velocity, quaternion), originals, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_state_noise_has_requested_scale_and_preserves_true_state():
    """Noise has physical units, while the observed attitude remains a unit quaternion."""
    num_envs = 32768
    cfg = SimToRealCfg(
        enabled=True,
        observation_noise=True,
        input_delay=False,
        action_delay=False,
        position_std=0.02,
        velocity_std=0.05,
        attitude_std_deg=1.0,
    )
    perturbation = PolicyPerturbation(cfg, num_envs, "cpu", seed=8642, policy_dt=0.05)
    position = torch.full((num_envs, 3), 2.0)
    velocity = torch.full((num_envs, 3), -0.5)
    quaternion = torch.zeros((num_envs, 4))
    quaternion[:, 3] = 1.0
    originals = tuple(value.clone() for value in (position, velocity, quaternion))

    noisy_position, noisy_velocity, noisy_quaternion = perturbation.state(position, velocity, quaternion)

    for noisy, true, standard_deviation in (
        (noisy_position, originals[0], 0.02),
        (noisy_velocity, originals[1], 0.05),
    ):
        noise = noisy - true
        assert torch.all(noise.mean(dim=0).abs() < 0.03 * standard_deviation)
        torch.testing.assert_close(
            noise.std(dim=0, unbiased=False),
            torch.full((3,), standard_deviation),
            rtol=0.03,
            atol=0,
        )

    torch.testing.assert_close(noisy_quaternion.norm(dim=-1), torch.ones(num_envs), rtol=0, atol=2e-6)
    assert not torch.equal(noisy_quaternion, originals[2])
    for actual, expected in zip((position, velocity, quaternion), originals, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
