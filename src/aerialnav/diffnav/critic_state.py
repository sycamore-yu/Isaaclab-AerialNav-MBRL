"""Augmented critic observations; the actor still receives only depth/odometry.

Fixed physical-unit scales are not running normalization. Velocity-window and
pending-action memory retain derivatives inside a rollout. Scene occlusion and
the unobserved PhysX contact state still make this an approximate value state;
no exact Markov or differentiable-contact claim is made.
"""

import torch

from .model import quaternion_matrix


class CriticHistory:
    def __init__(self, window, example):
        self.window = window
        self.samples = [example.new_zeros(example.shape[0], 7) for _ in range(window)]

    def detach(self):
        self.samples = [sample.detach() for sample in self.samples]

    def push(self, velocity, target_velocity, done):
        sample = torch.cat(
            (velocity / 10, target_velocity.detach() / 10, velocity.new_ones(velocity.shape[0], 1)), dim=-1
        )
        self.samples = self.samples[1:] + [sample]
        self.reset(done)

    def reset(self, done):
        self.samples = [torch.where(done[:, None], torch.zeros_like(value), value) for value in self.samples]

    def features(self):
        return torch.cat(self.samples, dim=-1)


def _delay_features(delay, example, width, scales):
    count = delay.history_length + 1
    padding = [example.new_zeros(example.shape[0], width) for _ in range(count - len(delay.history))]
    values = padding + delay.history
    scale = example.new_tensor(scales)
    age = (
        example.new_zeros(example.shape[0])
        if delay.ages is None
        else delay.ages.detach().clone().clamp_max(count).to(example) / count
    )
    return torch.cat([*[value / scale for value in values], age[:, None]], dim=-1)


def critic_observation(
    state,
    observation,
    hidden,
    previous_command,
    parameters,
    history,
    action_delay,
    observation_delay,
    episode_steps,
    max_episode_steps,
    action_lags,
):
    """Use h_t before its actor call, and h_(t+1) for the resulting next state.

    Both algorithms use pre-reset endpoint features and mask absorbing terminals. Live endpoints keep state, hidden,
    temporal-memory and delayed-command paths.
    All mutable environment buffers are cloned or consumed into fresh tensors.
    """
    scales = {"max_speed": 10, "pitch_ctl_delay": 12, "yaw_ctl_delay": 12, "wind": 10, "dg": 10}
    randomization = [
        value.detach().reshape(len(hidden), -1).clone() / scales.get(name, 1)
        for name, value in sorted(parameters.items())
    ]
    return torch.cat(
        (
            (observation["target"].detach() - state.position) / 50,
            state.velocity / 10,
            state.acceleration / 20,
            state.actuator / 20,
            previous_command / 20,
            quaternion_matrix(observation["quaternion"].detach()).flatten(1),
            (1 - episode_steps.detach().clone().float() / max_episode_steps)[:, None],
            observation["depth_features"].detach().clone().flatten(1),
            hidden,
            history.features(),
            _delay_features(action_delay, hidden, 3, (20, 20, 20)),
            _delay_features(observation_delay, hidden, 6, (50, 50, 50, 10, 10, 10)),
            observation["input_delay_steps"].detach().clone()[:, None] / max(observation_delay.history_length, 1),
            action_lags.detach().clone()[:, None] / max(action_delay.history_length, 1),
            *randomization,
        ),
        dim=-1,
    )
