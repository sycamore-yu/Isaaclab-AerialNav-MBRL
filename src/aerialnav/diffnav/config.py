"""Explicit units and defaults for the approved DiffNav migration."""

import math
from dataclasses import dataclass
from typing import Literal

from aerialnav.physics import DEFAULT_FRONTEND, DEFAULT_PHYSICS, validate_runtime

DIFFNAV_VARIANTS = ("step", "continuous", "terminal")
DIFFNAV_ALGORITHMS = ("bptt", "shac", "ahac")


@dataclass
class DiffNavTrainCfg:
    supported_perceptions = ("depth",)
    variant: Literal["step", "continuous", "terminal"] = "terminal"
    algorithm: Literal["bptt", "shac", "ahac"] = "bptt"
    recurrent_model: Literal["gru", "sru_lstm_gated"] = "gru"
    timesteps: int = 150
    max_iterations: int | None = None
    stop_after_acceptance: bool = False
    physics: Literal["isaacsim_physx", "newton_mjwarp"] = DEFAULT_PHYSICS
    frontend: Literal["torch", "warp"] = DEFAULT_FRONTEND
    physics_dt: float = 0.05
    policy_dt: float = 0.05
    velocity_loss_mode: Literal["variant_default", "per_step", "truncated_window", "full_window", "ema"] | None = None
    collision_metric: Literal["sdf"] = "sdf"
    position_integration: Literal["original", "trapezoid"] | None = None
    geometry_backend: Literal["native", "curobo_v2"] = "curobo_v2"
    avoidance_gradient: Literal["original", "reshaped"] = "original"
    perception: Literal["depth"] = "depth"
    approach_mode: Literal["finite_difference", "relative"] = "finite_difference"
    lateral_avoidance: float = 0.0
    risk_speed_floor: float = 1.0
    velocity_window_s: float = 2.0
    grad_decay: float = 0.4
    lr: float = 0.001
    optimizer: Literal["adamw", "adam"] = "adamw"
    weight_decay: float = 0.01
    cosine_period_updates: int = 400
    min_lr_ratio: float = 0.01
    lr_schedule: Literal["cosine", "plateau", "constant"] = "cosine"
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    plateau_threshold: float = 0.005
    plateau_cooldown: int = 2
    minimum_lr: float = 1e-6
    observation_normalization: bool = False
    normalization_variance_floor: float = 1e-4
    recovery_enabled: bool = True
    discount: float = 0.99
    td_lambda: float = 0.95
    critic_lr: float = 0.0005
    failure_terminal_cost: float = 0.0
    critic_target_tau: float = 0.6
    critic_iterations: int = 16
    critic_max_iterations: int = 64
    critic_value_transform: Literal["linear", "softplus"] = "linear"
    ahac_min_horizon: int = 16
    ahac_max_horizon: int = 300
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
    height_loss_mode: Literal["target_exp", "target_huber", "boundary_huber", "goal_height_band"] = "goal_height_band"
    height_target_range: tuple[float, float] = (0.5, 2.5)
    boundary_margin: float = 0.5
    boundary_scale: float = 0.5
    height_bounds: tuple[float, float] = (0.2, 4.0)
    seed: int = 0
    gradient_diagnostic_updates: tuple[int, ...] = ()
    gradient_projection: Literal["none", "height_priority"] = "none"
    loss_timing: Literal["deferred", "pre_reset"] | None = None

    def __post_init__(self):
        # Older public weights used this name for terminal sampling.
        if self.variant == "dcol":
            self.variant = "terminal"
        if self.velocity_loss_mode is None:
            self.velocity_loss_mode = "per_step" if self.algorithm in ("shac", "ahac") else "full_window"
        if self.position_integration is None:
            self.position_integration = (
                "trapezoid" if self.algorithm == "bptt" and self.velocity_loss_mode == "full_window" else "original"
            )
        if self.loss_timing is None:
            self.loss_timing = (
                "pre_reset" if self.algorithm == "bptt" and self.velocity_loss_mode == "full_window" else "deferred"
            )

    @property
    def resolved_velocity_loss_mode(self) -> Literal["per_step", "window", "truncated_window", "ema"]:
        """Select velocity supervision without changing collision geometry or critic memory."""
        if self.velocity_loss_mode == "variant_default":
            return "per_step" if self.variant == "step" else "truncated_window"
        if self.velocity_loss_mode == "full_window":
            return "window"
        if self.velocity_loss_mode in ("per_step", "truncated_window", "ema"):
            return self.velocity_loss_mode
        raise ValueError("velocity_loss_mode must be variant_default, per_step, truncated_window, full_window, or ema")

    @property
    def velocity_window_steps(self) -> int:
        steps = round(self.velocity_window_s / self.policy_dt)
        if steps < 1 or abs(steps * self.policy_dt - self.velocity_window_s) > 1e-7:
            raise ValueError("velocity_window_s must be a positive integer number of policy steps")
        return steps

    def validate(self) -> None:
        self.validate_collision_metric()
        if self.avoidance_gradient not in ("original", "reshaped"):
            raise ValueError("avoidance_gradient must be original or reshaped")
        if self.avoidance_gradient == "reshaped" and (
            self.algorithm != "bptt" or self.approach_mode != "finite_difference"
        ):
            raise ValueError("reshaped avoidance requires BPTT and finite_difference approach mode")
        if self.geometry_backend not in ("native", "curobo_v2"):
            raise ValueError("geometry_backend must be native or curobo_v2")
        if self.geometry_backend == "curobo_v2" and (
            self.collision_metric != "sdf" or self.approach_mode != "finite_difference"
        ):
            raise ValueError("cuRoboV2 requires SDF and finite_difference approach mode")
        if self.position_integration not in ("original", "trapezoid"):
            raise ValueError("position_integration must be original or trapezoid")
        if self.perception not in self.supported_perceptions:
            raise ValueError("Original DiffNav uses depth observations")
        if self.loss_timing not in ("deferred", "pre_reset"):
            raise ValueError("loss_timing must be deferred or pre_reset")
        if self.loss_timing == "pre_reset" and (
            self.algorithm != "bptt" or self.resolved_velocity_loss_mode != "window"
        ):
            raise ValueError("pre_reset loss timing currently requires BPTT with full_window velocity")
        validate_runtime(self.physics, self.frontend)
        if self.variant not in DIFFNAV_VARIANTS:
            raise ValueError(f"variant must be one of {DIFFNAV_VARIANTS}")
        if self.algorithm not in DIFFNAV_ALGORITHMS:
            raise ValueError(f"algorithm must be one of {DIFFNAV_ALGORITHMS}")
        if self.recurrent_model not in ("gru", "sru_lstm_gated"):
            raise ValueError("recurrent_model must be gru or sru_lstm_gated")
        if self.recurrent_model == "sru_lstm_gated" and self.algorithm != "bptt":
            raise ValueError("SRU-LSTM-Gated currently requires --algorithm bptt; SHAC/AHAC are not supported")
        _ = self.resolved_velocity_loss_mode
        if self.approach_mode not in ("finite_difference", "relative"):
            raise ValueError("approach_mode must be finite_difference or relative")
        if not math.isfinite(self.lateral_avoidance) or self.lateral_avoidance < 0:
            raise ValueError("lateral_avoidance must be finite and nonnegative")
        if not math.isfinite(self.risk_speed_floor) or not 0 < self.risk_speed_floor <= 1:
            raise ValueError("risk_speed_floor must be in (0, 1]")
        if (self.lateral_avoidance or self.risk_speed_floor < 1) and self.approach_mode != "relative":
            raise ValueError("lateral avoidance and risk speed require relative approach")
        if self.optimizer not in ("adamw", "adam"):
            raise ValueError("optimizer must be adamw or adam")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.height_loss_mode not in ("target_exp", "target_huber", "boundary_huber", "goal_height_band"):
            raise ValueError("height_loss_mode must be target_exp, target_huber, boundary_huber, or goal_height_band")
        if (
            len(self.height_target_range) != 2
            or not all(math.isfinite(x) for x in self.height_target_range)
            or not self.height_bounds[0]
            < self.height_target_range[0]
            < self.height_target_range[1]
            < self.height_bounds[1]
        ):
            raise ValueError("height_target_range must be a finite increasing interval inside the height bounds")
        if self.gradient_projection not in ("none", "height_priority"):
            raise ValueError("gradient_projection must be none or height_priority")
        if self.gradient_projection != "none" and self.algorithm != "bptt":
            raise ValueError("Height-priority projection currently requires the unbootstrapped BPTT objective")
        if self.critic_value_transform not in ("linear", "softplus"):
            raise ValueError("critic_value_transform must be linear or softplus")
        if self.algorithm == "bptt" and self.critic_value_transform != "linear":
            raise ValueError("A non-linear critic_value_transform requires SHAC or AHAC")
        if not math.isfinite(self.failure_terminal_cost) or self.failure_terminal_cost < 0:
            raise ValueError("failure_terminal_cost must be finite and nonnegative")
        if self.failure_terminal_cost and self.algorithm == "bptt":
            raise ValueError("failure_terminal_cost requires SHAC or AHAC")
        if not math.isfinite(self.critic_target_tau) or not 0 < self.critic_target_tau <= 1:
            raise ValueError("critic_target_tau must be in (0, 1]")
        if self.algorithm != "bptt" and self.variant != "terminal":
            raise ValueError("The SHAC/AHAC comparisons require the terminal configuration")
        if self.lr_schedule not in ("cosine", "plateau", "constant"):
            raise ValueError("lr_schedule must be cosine, plateau, or constant")
        if any(type(value) is not bool for value in (self.observation_normalization, self.recovery_enabled)):
            raise ValueError("normalization and recovery switches must be boolean")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.lr, self.minimum_lr, self.normalization_variance_floor, self.critic_lr)
        ):
            raise ValueError("learning rates and normalization variance floor must be finite and positive")
        if not 0 < self.min_lr_ratio <= 1 or not 0 < self.plateau_factor < 1:
            raise ValueError("Invalid scheduler factor")
        if not 0 <= self.discount < 1 or not 0 <= self.td_lambda <= 1:
            raise ValueError("discount must be in [0,1) and td_lambda in [0,1]")
        if any(type(value) is not int or value < 0 for value in (self.plateau_patience, self.plateau_cooldown)):
            raise ValueError("plateau report patience and cooldown must be nonnegative integers")
        if not math.isfinite(self.plateau_threshold) or self.plateau_threshold < 0:
            raise ValueError("plateau_threshold must be finite and nonnegative")
        if any(
            type(value) is not int or value < 1
            for value in (
                self.critic_iterations,
                self.critic_max_iterations,
                self.ahac_min_horizon,
                self.ahac_max_horizon,
                self.ahac_initial_horizon,
            )
        ):
            raise ValueError("critic iterations and AHAC horizons must be positive integers")
        if not 2 <= self.ahac_min_horizon <= self.ahac_initial_horizon <= self.ahac_max_horizon:
            raise ValueError("AHAC requires 2 <= min <= initial <= max horizon")
        if (
            any(
                not math.isfinite(value) or value <= 0
                for value in (self.ahac_risk_scale, self.ahac_risk_budget, self.ahac_dual_lr)
            )
            or not math.isfinite(self.ahac_risk_clearance)
            or self.ahac_risk_clearance < 0
        ):
            raise ValueError("Invalid AHAC DCOL risk parameters")
        if (self.physics_dt, self.policy_dt) not in ((0.01, 0.1), (0.05, 0.05), (0.02, 0.02)):
            raise ValueError("Supported timing is 100/10 Hz, synchronized 20 Hz, or synchronized 50 Hz")
        if self.timesteps < 2 or (self.max_iterations is not None and self.max_iterations < 1):
            raise ValueError("Training requires at least two rollout steps and one update")
        if self.cosine_period_updates < 1:
            raise ValueError("cosine_period_updates must be positive")
        if not 0 < self.grad_decay <= 1 or self.max_grad_norm <= 0:
            raise ValueError("grad_decay must be in (0, 1] and max_grad_norm must be positive")
        lower, upper = self.height_bounds
        if not all(
            math.isfinite(value)
            for value in (lower, upper, self.boundary_margin, self.boundary_scale, self.coef_height_boundary)
        ):
            raise ValueError("height boundary parameters must be finite")
        if self.boundary_margin < 0 or lower + self.boundary_margin >= upper - self.boundary_margin:
            raise ValueError("height_bounds and boundary_margin must leave a positive interior interval")
        if self.boundary_scale <= 0 or self.coef_height_boundary < 0:
            raise ValueError("boundary_scale must be positive and coef_height_boundary must be nonnegative")
        if any(type(update) is not int or update < 1 for update in self.gradient_diagnostic_updates) or len(
            set(self.gradient_diagnostic_updates)
        ) != len(self.gradient_diagnostic_updates):
            raise ValueError("gradient_diagnostic_updates must contain unique positive integers")
        _ = self.velocity_window_steps

    def validate_collision_metric(self):
        if self.collision_metric != "sdf":
            raise ValueError(
                "Current DiffNav uses SDF only; use the explicit historical DCOL reference for comparisons"
            )
