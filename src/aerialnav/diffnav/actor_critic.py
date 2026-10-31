"""Cost-form SHAC/AHAC value learning, independent of the PhysX rollout.

The runner supplies DiffNav's actual per-transition costs and augmented critic
states. Both algorithms supply zero next values at absorbing terminals and the
original pre-reset endpoint value at time truncations. At an ordinary rollout endpoint,
the frozen critic retains position and recurrent-memory input derivatives.

SHAC uses one critic and an exponential-moving-average target. AHAC uses two
critics with a frozen per-rollout copy, not a delayed EMA target. Cost values
reverse the sign of reward values: the maximum twin value is the pessimistic
bootstrap for both actor and TD targets, matching the reward-min convention.
An optional softplus value output supplies a nonnegative cost-value prior. The
default linear output and the actor/TD objectives remain unchanged.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


def _check_trajectory(costs, next_values, trace_cut):
    if costs.ndim != 2 or not all(costs.shape):
        raise ValueError("costs must have nonempty [time, environment] dimensions")
    if next_values.shape != costs.shape or trace_cut.shape != costs.shape:
        raise ValueError("next_values and trace_cut must match the cost trajectory")
    if trace_cut.dtype != torch.bool:
        raise ValueError("trace_cut must be a boolean tensor")
    if next_values.device != costs.device or trace_cut.device != costs.device:
        raise ValueError("trajectory tensors must use the same device")


def _check_discount(discount, td_lambda=None):
    if not math.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError("discount must be finite and between zero and one")
    if td_lambda is not None and (not math.isfinite(td_lambda) or not 0 <= td_lambda <= 1):
        raise ValueError("td_lambda must be finite and between zero and one")


@torch.no_grad()
def td_lambda_cost_targets(costs, next_values, trace_cut, *, discount=0.99, td_lambda=0.95):
    """Return detached TD(lambda) targets without crossing a reset boundary.

    ``next_values[t]`` is zero for an absorbing terminal and uses the pre-reset
    endpoint for a time limit. ``trace_cut`` stops the lambda recursion, without
    erasing the supplied timeout bootstrap or connecting the following episode.
    The last sample always bootstraps from its own supplied next value.
    """
    _check_trajectory(costs, next_values, trace_cut)
    _check_discount(discount, td_lambda)
    targets = torch.empty_like(costs)
    continuation = next_values[-1]
    for step in reversed(range(costs.shape[0])):
        continuation = torch.where(trace_cut[step], next_values[step], continuation)
        continuation = costs[step] + discount * ((1.0 - td_lambda) * next_values[step] + td_lambda * continuation)
        targets[step] = continuation
    return targets


def short_horizon_cost_objective(costs, next_values, trace_cut, *, discount=0.99):
    """Differentiable cost returns with a bootstrap at every segment endpoint.

    A segment ends at an environment reset or at the common rollout horizon.
    Discount powers restart after resets. The supplied reset next values must
    be detached, while ordinary final next values must retain the state graph.
    Division by the number of collected transitions follows the SHAC rollout
    objective and keeps the direct cost in the DiffNav per-transition scale.
    """
    _check_trajectory(costs, next_values, trace_cut)
    _check_discount(discount)
    weight = torch.ones_like(costs[0])
    cost_sum = costs.new_zeros(())
    bootstrap_sum = costs.new_zeros(())
    for step in range(costs.shape[0]):
        cost_sum = cost_sum + (weight * costs[step]).sum()
        boundary = torch.ones_like(trace_cut[step]) if step == costs.shape[0] - 1 else trace_cut[step]
        endpoint_value = torch.where(boundary, next_values[step], torch.zeros_like(next_values[step]))
        bootstrap_sum = bootstrap_sum + (weight * discount * endpoint_value).sum()
        if step == costs.shape[0] - 1:
            last_weight = weight
        weight = torch.where(boundary, torch.ones_like(weight), weight * discount)
    divisor = costs.numel()
    direct = cost_sum / divisor
    bootstrap = bootstrap_sum / divisor
    # Diagnostics only: retain the exact existing objective accumulation above.
    # All earlier endpoints are reset boundaries. Only a non-reset final state
    # supplies a live bootstrap derivative in the runner's rollout contract.
    with torch.no_grad():
        live_endpoint = (
            last_weight
            * discount
            * torch.where(trace_cut[-1], torch.zeros_like(next_values[-1]), next_values[-1].detach())
        ).sum() / divisor
        reset_bootstrap = bootstrap.detach() - live_endpoint
    return {
        "objective": direct + bootstrap,
        "discounted_cost": direct,
        "bootstrap": bootstrap,
        "bootstrap_reset": reset_bootstrap,
        "bootstrap_live_endpoint": live_endpoint,
    }


@dataclass
class AHACSettings:
    """Explicit metric-clearance adaptation, not the paper's contact-stiffness detector.

    The short-horizon actor/critic objective follows AHAC. The synchronous
    horizon feedback below is a documented task adaptation: it does not copy
    the sign-inconsistent horizon update in the pinned reference implementation.
    Distances and ``risk_clearance``/``risk_scale`` are in metres.
    """

    initial_horizon: int = 32
    min_horizon: int = 16
    max_horizon: int = 300
    risk_clearance: float = 0.10
    risk_scale: float = 0.05
    risk_budget: float = 0.10
    dual_lr: float = 0.10
    initial_dual: float = 1.0
    max_dual: float = 100.0
    shrink_factor: float = 0.8
    growth_steps: int = 4
    safe_budget_fraction: float = 0.5
    stable_updates_required: int = 3
    max_terminal_fraction: float = 0.05
    unstable_gradient_ratio: float = 2.0
    max_model_velocity_error: float = 0.5

    def validate(self):
        for name in ("initial_horizon", "min_horizon", "max_horizon", "growth_steps", "stable_updates_required"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.min_horizon <= self.initial_horizon <= self.max_horizon:
            raise ValueError("AHAC requires min_horizon <= initial_horizon <= max_horizon")
        if not math.isfinite(self.risk_clearance) or self.risk_clearance < 0:
            raise ValueError("risk_clearance must be finite and nonnegative")
        if not 0 <= self.max_terminal_fraction <= 1:
            raise ValueError("max_terminal_fraction must be within [0, 1]")
        for name in (
            "risk_scale",
            "risk_budget",
            "dual_lr",
            "max_dual",
            "unstable_gradient_ratio",
            "max_model_velocity_error",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.initial_dual) or not 0 <= self.initial_dual <= self.max_dual:
            raise ValueError("initial_dual must be within [0, max_dual]")
        for name in ("shrink_factor", "safe_budget_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be within (0, 1)")


class AdaptiveHorizonController:
    """One shared rollout horizon, a projected dual, and live risk gradients.

    Read ``horizon`` once before collecting a parallel rollout. Add ``penalty``
    to its actor objective, then call ``update`` only after actor backward.
    Risk never directly resets an environment or truncates an individual graph.
    The critic continues to fit the original DiffNav costs, without this separate
    Lagrange term. Controller state contains only scalars, never an actor graph.
    """

    def __init__(self, settings=None):
        self.settings = AHACSettings() if settings is None else copy.deepcopy(settings)
        self.settings.validate()
        self._horizon = self.settings.initial_horizon
        self._dual = float(self.settings.initial_dual)
        self.completed_updates = 0
        self.stable_updates = 0

    @property
    def horizon(self):
        """Number of policy steps for every environment in the next rollout."""
        return self._horizon

    @property
    def dual(self):
        return self._dual

    def risk(self, gaps):
        """Map metre-valued ``[T, N, K]`` gaps to live ``[T, N]`` geometry risk.

        The nearest of K samples defines risk. Positive infinity means no
        obstacle and maps to zero. This is not a contact Jacobian or stiffness.
        """
        if gaps.ndim != 3 or not all(gaps.shape) or not torch.is_floating_point(gaps):
            raise ValueError("gaps must be a nonempty floating [time, environment, sample] tensor")
        minimum_gap = gaps.amin(dim=-1)
        return F.softplus((self.settings.risk_clearance - minimum_gap) / self.settings.risk_scale)

    @staticmethod
    def _check_risk(risk):
        if risk.ndim != 2 or not all(risk.shape) or not torch.is_floating_point(risk):
            raise ValueError("risk must be a nonempty floating [time, environment] tensor")

    def penalty(self, risk):
        """Penalize the worst trajectory's temporal peak, with no batch dilution."""
        self._check_risk(risk)
        trajectory_peak = risk.amax(dim=0)
        return self._dual * (trajectory_peak.amax() - self.settings.risk_budget).relu()

    @torch.no_grad()
    def update(self, risk, *, terminal_fraction=0.0, gradient_ratio=0.0, model_velocity_error=0.0):
        """Shrink on any unsafe signal; grow only after consecutive stable updates."""
        self._check_risk(risk)
        if risk.shape[0] != self._horizon:
            raise ValueError("risk time dimension must equal the shared rollout horizon")
        mean, minimum, maximum, over_budget = (
            torch.stack(
                (
                    risk.detach().mean(),
                    risk.detach().amin(),
                    risk.detach().amax(),
                    (risk.detach() > self.settings.risk_budget).to(risk.dtype).mean(),
                )
            )
            .cpu()
            .tolist()
        )
        values = (mean, minimum, maximum, terminal_fraction, gradient_ratio, model_velocity_error)
        if not all(math.isfinite(v) and v >= 0 for v in values) or terminal_fraction > 1:
            raise FloatingPointError("AHAC risk and stability signals must be finite and nonnegative")
        old_horizon, old_dual = self._horizon, self._dual
        self._dual = min(
            self.settings.max_dual, max(0.0, self._dual + self.settings.dual_lr * (maximum - self.settings.risk_budget))
        )
        unsafe = (
            maximum > self.settings.risk_budget
            or terminal_fraction > self.settings.max_terminal_fraction
            or gradient_ratio > self.settings.unstable_gradient_ratio
            or model_velocity_error > self.settings.max_model_velocity_error
        )
        safe = (
            maximum < self.settings.risk_budget * self.settings.safe_budget_fraction
            and terminal_fraction <= self.settings.max_terminal_fraction
            and gradient_ratio <= 1
            and model_velocity_error <= self.settings.max_model_velocity_error
        )
        self.stable_updates = self.stable_updates + 1 if safe and not unsafe else 0
        if unsafe:
            self._horizon = max(self.settings.min_horizon, math.floor(old_horizon * self.settings.shrink_factor))
        elif self.stable_updates >= self.settings.stable_updates_required:
            self._horizon = min(self.settings.max_horizon, old_horizon + self.settings.growth_steps)
            self.stable_updates = 0
        self.completed_updates += 1
        return {
            "ahac_horizon_used": old_horizon,
            "ahac_horizon_next": self._horizon,
            "ahac_horizon_change": self._horizon - old_horizon,
            "ahac_dual_before": old_dual,
            "ahac_dual_after": self._dual,
            "ahac_risk_mean": mean,
            "ahac_risk_max": maximum,
            "ahac_constraint_risk": maximum,
            "ahac_risk_over_budget_fraction": over_budget,
            "ahac_risk_budget": self.settings.risk_budget,
            "ahac_terminal_trajectory_fraction": terminal_fraction,
            "ahac_actor_gradient_clip_ratio": gradient_ratio,
            "ahac_model_velocity_error_mps": model_velocity_error,
            "ahac_stable_update_streak": self.stable_updates,
            "ahac_unsafe": float(unsafe),
            "ahac_completed_updates": self.completed_updates,
        }

    def state_dict(self):
        return {
            "format_version": 2,
            "adaptation": "metric_clearance_peak_stability_feedback",
            "settings": asdict(self.settings),
            "horizon": self._horizon,
            "dual": self._dual,
            "completed_updates": self.completed_updates,
            "stable_updates": self.stable_updates,
        }

    def load_state_dict(self, state):
        if (
            state["format_version"] != 2
            or state["adaptation"] != "metric_clearance_peak_stability_feedback"
            or state["settings"] != asdict(self.settings)
        ):
            raise ValueError("incompatible AHAC adaptation checkpoint")
        horizon, dual, updates = state["horizon"], state["dual"], state["completed_updates"]
        if type(horizon) is not int or not self.settings.min_horizon <= horizon <= self.settings.max_horizon:
            raise ValueError("checkpoint horizon is outside configured bounds")
        if not math.isfinite(dual) or not 0 <= dual <= self.settings.max_dual:
            raise ValueError("checkpoint dual is outside configured bounds")
        if type(updates) is not int or updates < 0:
            raise ValueError("checkpoint update count must be a nonnegative integer")
        stable = state["stable_updates"]
        if type(stable) is not int or not 0 <= stable < self.settings.stable_updates_required:
            raise ValueError("checkpoint stable update streak is invalid")
        self._horizon, self._dual, self.completed_updates = horizon, float(dual), updates
        self.stable_updates = stable


def critic_split_indices(shape, fraction, device, generator):
    """Reserve whole environments where possible; never fit held-out samples."""
    time, environments = shape
    if time * environments < 2:
        raise ValueError("critic validation requires at least two trajectory samples")
    grid = torch.arange(time * environments, device=device).reshape(time, environments)
    if environments > 1:
        order = torch.randperm(environments, device=device, generator=generator)
        count = min(environments - 1, max(1, math.ceil(fraction * environments)))
        return grid[:, order[count:]].reshape(-1), grid[:, order[:count]].reshape(-1)
    order = torch.randperm(time, device=device, generator=generator)
    count = min(time - 1, max(1, math.ceil(fraction * time)))
    return order[count:], order[:count]


@dataclass
class ActorCriticSettings:
    """Value-learning options; policy and simulator settings stay in the runner."""

    algorithm: Literal["shac", "ahac"] = "shac"
    discount: float = 0.99
    td_lambda: float = 0.95
    hidden_sizes: tuple[int, ...] = (256, 256)
    critic_lr: float = 0.0005
    critic_minibatches: int = 4
    critic_max_optimizer_steps: int = 64
    critic_validation_fraction: float = 0.1
    critic_iterations: int = 16
    critic_max_iterations: int = 64
    critic_min_iterations: int = 6
    critic_convergence_window: int = 5
    critic_relative_tolerance: float = 0.005
    critic_max_grad_norm: float = 5.0
    target_tau: float = 0.6
    seed: int = 0
    value_transform: Literal["linear", "softplus"] = "linear"

    def validate(self):
        if self.algorithm not in ("shac", "ahac"):
            raise ValueError("algorithm must be shac or ahac")
        if self.value_transform not in ("linear", "softplus"):
            raise ValueError("value_transform must be linear or softplus")
        _check_discount(self.discount, self.td_lambda)
        if not 0 < self.critic_validation_fraction < 1:
            raise ValueError("critic_validation_fraction must be within (0, 1)")
        if self.discount == 1:
            raise ValueError("cost value fitting requires discount < 1")
        if not self.hidden_sizes or any(type(size) is not int or size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        for name in (
            "critic_minibatches",
            "critic_max_optimizer_steps",
            "critic_iterations",
            "critic_max_iterations",
            "critic_min_iterations",
            "critic_convergence_window",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.critic_min_iterations > self.critic_max_iterations:
            raise ValueError("critic_min_iterations cannot exceed critic_max_iterations")
        for name in ("critic_lr", "critic_relative_tolerance", "critic_max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.target_tau) or not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must be in (0, 1]")


class CostValueNetwork(nn.Module):
    """An independent scalar value function over the runner's augmented state."""

    def __init__(self, input_dim, hidden_sizes=(256, 256), *, value_transform="linear"):
        super().__init__()
        if value_transform not in ("linear", "softplus"):
            raise ValueError("value_transform must be linear or softplus")
        self.value_transform = value_transform
        layers = []
        previous = input_dim
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, width), nn.ELU()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, observation):
        raw = self.network(observation).squeeze(-1)
        return F.softplus(raw, beta=1, threshold=20) if self.value_transform == "softplus" else raw


class CostActorCritic:
    """Own critics, frozen value snapshots, optimizer and TD(lambda) fitting.

    Call ``actor_value`` on live endpoint observations and ``bootstrap_value``
    on detached next observations during a rollout. After actor backward and
    its optimizer step, call ``fit``. Target weights must not change before all
    uses of the current actor graph have finished.
    """

    def __init__(self, input_dim, settings=None, *, device="cpu"):
        self.settings = ActorCriticSettings() if settings is None else copy.deepcopy(settings)
        self.settings.validate()
        if type(input_dim) is not int or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        self.input_dim = input_dim
        self.device = torch.device(device)
        count = 1 if self.settings.algorithm == "shac" else 2
        # Initializing the critic must not move the policy/environment RNG.
        initialization_rng = torch.Generator(device="cpu").manual_seed(self.settings.seed)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(initialization_rng.get_state())
            self.critics = nn.ModuleList(
                [
                    CostValueNetwork(
                        input_dim, self.settings.hidden_sizes, value_transform=self.settings.value_transform
                    )
                    for _ in range(count)
                ]
            ).to(self.device)
        self.device = next(self.critics.parameters()).device
        self.frozen_critics = copy.deepcopy(self.critics).requires_grad_(False).eval()
        self.optimizer = torch.optim.AdamW(self.critics.parameters(), lr=self.settings.critic_lr)
        self._shuffle_rng = torch.Generator(device=self.device).manual_seed(self.settings.seed + 1)
        self.completed_fits = 0

    def _check_observation(self, observation):
        if observation.shape[-1] != self.input_dim or observation.device != self.device:
            raise ValueError("critic observation has the wrong feature dimension or device")

    def actor_value(self, observation):
        """Frozen weights, live input derivatives; max of the AHAC cost twins."""
        self._check_observation(observation)
        values = torch.stack([critic(observation) for critic in self.frozen_critics], dim=-1)
        return values.max(dim=-1).values

    @torch.no_grad()
    def bootstrap_value(self, observation):
        """Detached TD target; max cost is the reward-min twin-critic convention."""
        self._check_observation(observation)
        values = torch.stack([critic(observation) for critic in self.frozen_critics], dim=-1)
        return values.max(dim=-1).values

    @torch.no_grad()
    def update_frozen_critics(self):
        """Update only after the actor graph and critic fitting have completed."""
        tau = self.settings.target_tau if self.settings.algorithm == "shac" else 1.0
        for frozen, live in zip(self.frozen_critics.parameters(), self.critics.parameters(), strict=True):
            frozen.lerp_(live, tau)

    def fit(self, observation, costs, next_values, trace_cut):
        """Fit detached cost returns, then prepare the next rollout's value copy.

        Each pass uses a fixed number of minibatches, independent of H. Total
        optimizer steps have a hard cap. AHAC convergence is judged on held-out
        trajectories (or disjoint samples in single-environment unit probes).
        """
        _check_trajectory(costs, next_values, trace_cut)
        self._check_observation(observation)
        if observation.shape[:-1] != costs.shape:
            raise ValueError("critic observations must have shape [time, environment, feature]")
        features = observation.detach().reshape(-1, self.input_dim)
        targets = td_lambda_cost_targets(
            costs, next_values, trace_cut, discount=self.settings.discount, td_lambda=self.settings.td_lambda
        ).reshape(-1)
        if not bool(torch.isfinite(features).all() & torch.isfinite(targets).all()):
            raise FloatingPointError("non-finite critic observation or target")
        train, validation = critic_split_indices(
            costs.shape, self.settings.critic_validation_fraction, self.device, self._shuffle_rng
        )
        iterations = (
            self.settings.critic_iterations
            if self.settings.algorithm == "shac"
            else self.settings.critic_max_iterations
        )
        losses, validation_losses = [], []
        optimizer_steps = 0
        grad_norm = targets.new_zeros(())
        for _ in range(iterations):
            if optimizer_steps >= self.settings.critic_max_optimizer_steps:
                break
            shuffled = train[torch.randperm(len(train), device=self.device, generator=self._shuffle_rng)]
            pass_loss, fitted = targets.new_zeros(()), 0
            for selected in torch.tensor_split(shuffled, min(self.settings.critic_minibatches, len(shuffled))):
                if optimizer_steps >= self.settings.critic_max_optimizer_steps:
                    break
                prediction = torch.stack([critic(features[selected]) for critic in self.critics], dim=-1)
                loss = (prediction - targets[selected, None]).square().mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    self.critics.parameters(), self.settings.critic_max_grad_norm, error_if_nonfinite=True
                )
                self.optimizer.step()
                optimizer_steps += 1
                fitted += len(selected)
                pass_loss = pass_loss + loss.detach() * len(selected)
                grad_norm = torch.maximum(grad_norm, norm.detach())
            losses.append(float(pass_loss / fitted))
            with torch.no_grad():
                estimate = torch.stack([critic(features[validation]) for critic in self.critics], dim=-1)
                validation_losses.append(float((estimate - targets[validation, None]).square().mean()))
            window = self.settings.critic_convergence_window
            if (
                self.settings.algorithm == "ahac"
                and len(losses) >= self.settings.critic_min_iterations
                and len(validation_losses) > window
            ):
                recent = validation_losses[-window - 1 :]
                changes = [
                    abs(current - previous) / max(abs(previous), 1e-8)
                    for previous, current in zip(recent[:-1], recent[1:], strict=True)
                ]
                if max(changes) <= self.settings.critic_relative_tolerance or min(recent[1:]) >= recent[0]:
                    break
        self.update_frozen_critics()
        self.completed_fits += 1
        # These are the frozen next values supplied by this rollout, before the
        # fit above. Reductions are detached and transferred together once.
        next_detached = next_values.detach()
        next_minimum, next_mean, next_negative_fraction = (
            torch.stack(
                (
                    next_detached.amin(),
                    next_detached.mean(),
                    (next_detached < 0).to(next_detached.dtype).mean(),
                )
            )
            .cpu()
            .tolist()
        )
        return {
            "critic_loss": losses[-1],
            "critic_first_pass_loss": losses[0],
            "critic_optimizer_steps": optimizer_steps,
            "critic_optimizer_step_cap": self.settings.critic_max_optimizer_steps,
            "critic_validation_td_mse": validation_losses[-1],
            "critic_validation_first_pass_td_mse": validation_losses[0],
            "critic_training_samples": len(train),
            "critic_validation_samples": len(validation),
            "critic_minibatches": min(self.settings.critic_minibatches, len(train)),
            "critic_iterations": len(losses),
            "critic_max_grad_norm_observed": float(grad_norm),
            "critic_target_mean": float(targets.mean()),
            "critic_target_std": float(targets.std(unbiased=False)),
            "critic_completed_fits": self.completed_fits,
            "critic_next_value_min": next_minimum,
            "critic_next_value_mean": next_mean,
            "critic_next_value_negative_fraction": next_negative_fraction,
        }

    def state_dict(self):
        return {
            "format_version": 2,
            "input_dim": self.input_dim,
            "settings": asdict(self.settings),
            "critics": self.critics.state_dict(),
            "frozen_critics": self.frozen_critics.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "shuffle_rng_state": self._shuffle_rng.get_state(),
            "completed_fits": self.completed_fits,
        }

    def load_state_dict(self, state):
        if (
            state["format_version"] != 2
            or state["input_dim"] != self.input_dim
            or state["settings"]["algorithm"] != self.settings.algorithm
            or tuple(state["settings"]["hidden_sizes"]) != tuple(self.settings.hidden_sizes)
        ):
            raise ValueError("incompatible actor-critic checkpoint")
        if state["settings"].get("value_transform", "linear") != self.settings.value_transform:
            raise ValueError("incompatible critic value_transform in checkpoint")
        self.critics.load_state_dict(state["critics"], strict=True)
        self.frozen_critics.load_state_dict(state["frozen_critics"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self._shuffle_rng.set_state(state["shuffle_rng_state"].cpu())
        self.completed_fits = int(state["completed_fits"])
