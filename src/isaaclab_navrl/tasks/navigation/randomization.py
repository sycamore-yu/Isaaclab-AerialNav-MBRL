"""Shared policy perturbations and reset events for the two action interfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.math import quat_from_angle_axis, quat_mul


@dataclass
class SimToRealCfg:
    enabled: bool = False
    observation_noise: bool = True
    input_delay: bool = True
    action_delay: bool = True
    control_response: bool = True
    action_difference: bool = True
    mass: bool = True
    inertia: bool = True
    thrust: bool = True
    position_std: float = 0.02
    velocity_std: float = 0.05
    attitude_std_deg: float = 1.0
    lidar_std: float = 0.02
    depth_std: float = 0.02
    input_delay_range_s: tuple[float, float] = (0.0, 0.1)
    action_delay_range_s: tuple[float, float] = (0.0, 0.05)
    control_gain_range: tuple[float, float] = (0.6, 1.4)
    mass_scale_range: tuple[float, float] = (0.9, 1.1)
    inertia_scale_range: tuple[float, float] = (0.9, 1.1)
    thrust_scale_range: tuple[float, float] = (0.9, 1.1)
    action_difference_weight: float = 0.01


class PolicyPerturbation:
    """Advance official delay buffers once per policy step, never per sensor read."""

    def __init__(self, cfg, num_envs, device, seed, *, policy_dt, nominal_action_delay=0):
        self.cfg, self.device, self.num_envs = cfg, device, num_envs
        if not math.isfinite(policy_dt) or policy_dt <= 0:
            raise ValueError("policy_dt must be finite and positive")
        self.policy_dt = policy_dt
        self.input_delay_steps = self._step_range(cfg.input_delay_range_s)
        self.action_delay_steps = self._step_range(cfg.action_delay_range_s)
        self.nominal_action_delay = nominal_action_delay
        self.generator = torch.Generator(device=device).manual_seed(seed + 3109)
        self.observation_buffer = DelayBuffer(self.input_delay_steps[1], num_envs, device)
        self.action_buffer = DelayBuffer(max(self.action_delay_steps[1], nominal_action_delay), num_envs, device)
        self.control_gain_scale = torch.ones(num_envs, 1, device=device)
        self.thrust_scale = torch.ones_like(self.control_gain_scale)
        self._observation_key = None
        self._observation = None
        self.latest_observation = None
        self._observation_age_s = torch.zeros(num_envs, device=device)

    def _step_range(self, bounds):
        low, high = bounds
        if not all(math.isfinite(x) for x in bounds) or not 0 <= low <= high:
            raise ValueError("Delay bounds must be finite, nonnegative, and ordered")
        steps = (math.ceil(low / self.policy_dt - 1e-9), math.floor(high / self.policy_dt + 1e-9))
        if steps[0] > steps[1]:
            raise ValueError("Delay range contains no policy-clock tick")
        return steps

    @property
    def observation_age_s(self):
        """Actual age, including the initial buffer fill and local resets."""
        return self._observation_age_s

    @property
    def noisy(self):
        return self.cfg.enabled and self.cfg.observation_noise

    @torch.no_grad()
    def reset(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        for buffer, active, bounds, nominal in (
            (self.observation_buffer, self.cfg.input_delay, self.input_delay_steps, 0),
            (self.action_buffer, self.cfg.action_delay, self.action_delay_steps, self.nominal_action_delay),
        ):
            buffer.reset(ids)
            if self.cfg.enabled and active:
                lag = torch.randint(
                    bounds[0], bounds[1] + 1, (len(ids),), device=self.device, generator=self.generator, dtype=torch.int
                )
                buffer.time_lags[ids] = lag
            else:
                buffer.time_lags[ids] = nominal
        for target, active, bounds in (
            (self.control_gain_scale, self.cfg.control_response, self.cfg.control_gain_range),
            (self.thrust_scale, self.cfg.thrust, self.cfg.thrust_scale_range),
        ):
            if self.cfg.enabled and active:
                target[ids] = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
                    len(ids), 1, device=self.device, generator=self.generator
                )
            else:
                target[ids] = 1
        self._observation_key = None

    def normal(self, value, std):
        return value + torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=self.generator) * std

    def state(self, position, velocity, quaternion):
        if not self.noisy:
            return position, velocity, quaternion
        position = self.normal(position, self.cfg.position_std)
        velocity = self.normal(velocity, self.cfg.velocity_std)
        # Isotropic rotation-vector noise; never add noise to quaternion components.
        rotation = self.normal(torch.zeros_like(position), math.radians(self.cfg.attitude_std_deg))
        angle = rotation.norm(dim=-1)
        axis = rotation / angle[:, None].clamp_min(1e-12)
        quaternion = quat_mul(quaternion, quat_from_angle_axis(angle, axis))
        return position, velocity, quaternion

    @torch.no_grad()
    def observe(self, key, sample):
        if self._observation_key != key:
            self.latest_observation = sample().clone()
            tick = self.latest_observation.new_full((self.num_envs, 1), key)
            delivered = self.observation_buffer.compute(torch.cat((self.latest_observation, tick), dim=-1)).clone()
            self._observation = delivered[:, :-1]
            self._observation_age_s = (key - delivered[:, -1]) * self.policy_dt
            self._observation_key = key
        return self._observation

    @torch.no_grad()
    def execute(self, command):
        delayed = self.action_buffer.compute(command)
        # Hold the reset command at zero until the first issued command arrives.
        available = self.action_buffer.num_pushes > self.action_buffer.time_lags
        return torch.where(available[:, None], delayed, torch.zeros_like(delayed))


class DifferentiableDelay:
    """Immutable rollout history selected with the physical DelayBuffer's lags.

    Official DelayBuffer owns latency and physical execution. Its mutable ring
    storage cannot be retained in a BPTT graph, so only gradient history lives here.
    """

    def __init__(self, history_length):
        self.history_length = history_length
        self.history = []
        self.ages = None

    def push(self, value, lags, *, zero_until_ready=False):
        if self.ages is None:
            self.ages = torch.zeros(value.shape[0], dtype=torch.long, device=value.device)
        self.history = (self.history + [value])[-(self.history_length + 1) :]
        lag = torch.minimum(lags.long(), self.ages).clamp_max(len(self.history) - 1)
        stacked = torch.stack(self.history)
        selected = stacked[len(self.history) - 1 - lag, torch.arange(len(value), device=value.device)]
        if zero_until_ready:
            valid = (self.ages >= lags).reshape(-1, *([1] * (value.ndim - 1)))
            selected = torch.where(valid, selected, torch.zeros_like(selected))
        self.ages = self.ages + 1
        return selected

    def reset(self, done):
        if self.ages is None:
            return
        self.ages = torch.where(done, torch.zeros_like(self.ages), self.ages)
        self.history = [
            torch.where(done.reshape(-1, *([1] * (item.ndim - 1))), torch.zeros_like(item), item)
            for item in self.history
        ]

    def detach(self):
        self.history = [item.detach() for item in self.history]


@torch.no_grad()
def capture_nominal_dynamics(env, env_ids):
    """Read the loaded asset's masses and inertia before any reset randomization."""
    drone = env.scene["drone"]
    env.nominal_body_mass = drone.data.body_mass.torch.clone()
    env.nominal_body_inertia = drone.data.body_inertia.torch.clone()
    env.nominal_dynamics = {
        "source": "loaded asset before randomization",
        "physics_backend": drone.__backend_name__,
        "body_names": list(drone.body_names),
        "body_mass_kg": env.nominal_body_mass[0].cpu().tolist(),
        "body_inertia_kg_m2": env.nominal_body_inertia[0].cpu().tolist(),
        "total_mass_kg": float(env.nominal_body_mass[0].sum()),
    }


@torch.no_grad()
def reset_sim_to_real(env, env_ids):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.long()
    env.perturbation.reset(ids)
    controller = env.controller
    controller.velocity_gain[ids] = controller.nominal_velocity_gain * env.perturbation.control_gain_scale[ids]
    for name in ("KF", "KM"):
        controller.rotor_parameters[name][ids] = (
            getattr(controller.rotor_group, name) * env.perturbation.thrust_scale[ids]
        )
    masses = env.drone.data.body_mass.torch
    inertias = env.drone.data.body_inertia.torch
    env.mass_scale[ids] = (masses[ids].sum(-1) / env.nominal_body_mass[ids].sum(-1))[:, None]
    base = env.drone.body_names.index("base_link")
    diag = [0, 4]
    env.inertia_scale[ids] = (
        inertias[ids, base][:, diag].sum(-1) / env.nominal_body_inertia[ids, base][:, diag].sum(-1)
    )[:, None]
