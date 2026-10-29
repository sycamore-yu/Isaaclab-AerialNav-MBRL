"""Explicit units and defaults for the approved Zhang migration."""

import math
from dataclasses import dataclass
from typing import Literal

ZHANG_VARIANTS = ("step", "continuous", "dcol")
ZHANG_ALGORITHMS = ("bptt", "shac", "ahac")


@dataclass
class ZhangTrainCfg:
    variant: Literal["step", "continuous", "dcol"] = "step"
    algorithm: Literal["bptt", "shac", "ahac"] = "bptt"
    timesteps: int = 150
    max_iterations: int | None = None
    stop_after_acceptance: bool = False
    physics_dt: float = 0.01
    policy_dt: float = 0.1
    velocity_loss_mode: Literal["variant_default", "per_step"] = "variant_default"
    velocity_window_s: float = 2.0
    grad_decay: float = 0.4
    lr: float = 0.001
    cosine_period_updates: int = 400
    min_lr_ratio: float = 0.01
    lr_schedule: Literal["cosine", "plateau", "constant"] = "cosine"
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    plateau_threshold: float = 0.005
    plateau_cooldown: int = 2
    minimum_lr: float = 1e-6
    observation_normalization: bool = False
    normalization_variance_floor: float = 1e-4
    recovery_enabled: bool = True
    discount: float = 0.99
    td_lambda: float = 0.95
    critic_lr: float = 0.0005
    critic_iterations: int = 16
    critic_max_iterations: int = 64
    critic_value_transform: Literal["linear", "softplus"] = "linear"
    ahac_min_horizon: int = 16
    ahac_max_horizon: int = 150
    ahac_initial_horizon: int = 32
    ahac_risk_clearance: float = 0.1
    ahac_risk_scale: float = 0.05
    ahac_risk_budget: float = 0.1
    ahac_dual_lr: float = 0.1
    max_grad_norm: float = 5.0
    coef_v: float = 1.0
    coef_v_pred: float = 2.0
    coef_obj_avoidance: float = 3.0
    coef_collide: float = 7.5
    coef_d_acc: float = 0.01
    coef_d_jerk: float = 0.001
    coef_height_boundary: float = 1.0
    boundary_margin: float = 0.5
    boundary_scale: float = 0.5
    height_bounds: tuple[float, float] = (0.2, 4.0)
    seed: int = 0
    gradient_diagnostic_updates: tuple[int, ...] = ()

    @property
    def resolved_velocity_loss_mode(self) -> Literal["per_step", "window"]:
        """Select velocity supervision without changing collision geometry or critic memory."""
        if self.velocity_loss_mode == "variant_default":
            return "per_step" if self.variant == "step" else "window"
        if self.velocity_loss_mode == "per_step":
            return "per_step"
        raise ValueError("velocity_loss_mode must be variant_default or per_step")

    @property
    def velocity_window_steps(self) -> int:
        steps = round(self.velocity_window_s / self.policy_dt)
        if steps < 1 or abs(steps * self.policy_dt - self.velocity_window_s) > 1e-7:
            raise ValueError("velocity_window_s must be a positive integer number of policy steps")
        return steps

    def validate(self) -> None:
        if self.variant not in ZHANG_VARIANTS:
            raise ValueError(f"variant must be one of {ZHANG_VARIANTS}")
        if self.algorithm not in ZHANG_ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ZHANG_ALGORITHMS}")
        _ = self.resolved_velocity_loss_mode
        if self.critic_value_transform not in ("linear", "softplus"):
            raise ValueError("critic_value_transform must be linear or softplus")
        if self.algorithm == "bptt" and self.critic_value_transform != "linear":
            raise ValueError("A non-linear critic_value_transform requires SHAC or AHAC")
        if self.algorithm != "bptt" and self.variant != "dcol":
            raise ValueError("The SHAC/AHAC comparisons require the dcol configuration")
        if self.algorithm != "bptt" and self.gradient_diagnostic_updates:
            raise ValueError("Seven-term gradient diagnostics currently require the unbootstrapped BPTT objective")
        if self.lr_schedule not in ("cosine", "plateau", "constant"):
            raise ValueError("lr_schedule must be cosine, plateau, or constant")
        if any(type(value) is not bool for value in (self.observation_normalization, self.recovery_enabled)):
            raise ValueError("normalization and recovery switches must be boolean")
        if any(not math.isfinite(value) or value <= 0 for value in (
            self.lr, self.minimum_lr, self.normalization_variance_floor, self.critic_lr
        )):
            raise ValueError("learning rates and normalization variance floor must be finite and positive")
        if not 0 < self.min_lr_ratio <= 1 or not 0 < self.plateau_factor < 1:
            raise ValueError("Invalid scheduler factor")
        if not 0 <= self.discount < 1 or not 0 <= self.td_lambda <= 1:
            raise ValueError("discount must be in [0,1) and td_lambda in [0,1]")
        if any(type(value) is not int or value < 0 for value in (self.plateau_patience, self.plateau_cooldown)):
            raise ValueError("plateau report patience and cooldown must be nonnegative integers")
        if not math.isfinite(self.plateau_threshold) or self.plateau_threshold < 0:
            raise ValueError("plateau_threshold must be finite and nonnegative")
        if any(type(value) is not int or value < 1 for value in (
            self.critic_iterations, self.critic_max_iterations,
            self.ahac_min_horizon, self.ahac_max_horizon, self.ahac_initial_horizon
        )):
            raise ValueError("critic iterations and AHAC horizons must be positive integers")
        if not 2 <= self.ahac_min_horizon <= self.ahac_initial_horizon <= self.ahac_max_horizon:
            raise ValueError("AHAC requires 2 <= min <= initial <= max horizon")
        if any(not math.isfinite(value) or value <= 0 for value in (
            self.ahac_risk_scale, self.ahac_risk_budget, self.ahac_dual_lr
        )) or not math.isfinite(self.ahac_risk_clearance) or self.ahac_risk_clearance < 0:
            raise ValueError("Invalid AHAC DCOL risk parameters")
        if self.physics_dt != 0.01 or self.policy_dt != 0.1:
            raise ValueError("The approved task requires 100 Hz physics and a 10 Hz policy")
        if self.timesteps < 2 or (self.max_iterations is not None and self.max_iterations < 1):
            raise ValueError("Training requires at least two rollout steps and one update")
        if self.cosine_period_updates < 1:
            raise ValueError("cosine_period_updates must be positive")
        if not 0 < self.grad_decay <= 1 or self.max_grad_norm <= 0:
            raise ValueError("grad_decay must be in (0, 1] and max_grad_norm must be positive")
        lower, upper = self.height_bounds
        if not all(math.isfinite(value) for value in (
            lower, upper, self.boundary_margin, self.boundary_scale, self.coef_height_boundary
        )):
            raise ValueError("height boundary parameters must be finite")
        if self.boundary_margin < 0 or lower + self.boundary_margin >= upper - self.boundary_margin:
            raise ValueError("height_bounds and boundary_margin must leave a positive interior interval")
        if self.boundary_scale <= 0 or self.coef_height_boundary < 0:
            raise ValueError("boundary_scale must be positive and coef_height_boundary must be nonnegative")
        if (any(type(update) is not int or update < 1 for update in self.gradient_diagnostic_updates)
                or len(set(self.gradient_diagnostic_updates)) != len(self.gradient_diagnostic_updates)):
            raise ValueError("gradient_diagnostic_updates must contain unique positive integers")
        _ = self.velocity_window_steps
