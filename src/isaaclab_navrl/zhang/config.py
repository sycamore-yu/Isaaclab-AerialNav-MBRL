"""Explicit units and defaults for the approved Zhang migration."""

import math
from dataclasses import dataclass
from typing import Literal

ZHANG_VARIANTS = ("step", "continuous", "dcol", "truncated_window")


@dataclass
class ZhangTrainCfg:
    variant: Literal["step", "continuous", "dcol", "truncated_window"] = "step"
    timesteps: int = 150
    max_iterations: int | None = None
    stop_after_acceptance: bool = False
    physics_dt: float = 0.01
    policy_dt: float = 0.1
    velocity_window_s: float = 2.0
    grad_decay: float = 0.4
    lr: float = 0.001
    cosine_period_updates: int = 400
    min_lr_ratio: float = 0.01
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
    def velocity_window_steps(self) -> int:
        steps = round(self.velocity_window_s / self.policy_dt)
        if steps < 1 or abs(steps * self.policy_dt - self.velocity_window_s) > 1e-7:
            raise ValueError("velocity_window_s must be a positive integer number of policy steps")
        return steps

    def validate(self) -> None:
        if self.variant not in ZHANG_VARIANTS:
            raise ValueError(f"variant must be one of {ZHANG_VARIANTS}")
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
