"""PointNet formula weights and explicitly project-selected migration parameters."""

import math
from dataclasses import dataclass

from aerialnav.diffnav.config import DiffNavTrainCfg

# Historical metadata is accepted on load; new checkpoints use technical identifiers.
CHECKPOINT_METHODS = ("pointnet", "liu")
OBSERVATION_CONTRACTS = ("body_xyz_lidar_validity_v2", "liu_body_xyz_lidar_validity_v2")


@dataclass
class PointNetTrainCfg(DiffNavTrainCfg):
    supported_perceptions = ("lidar",)
    resume_runtime_fields = ("point_microbatch_envs", "point_precision", "point_encoder_backend")
    point_microbatch_envs: int = 64
    point_precision: str = "bf16"
    point_encoder_backend: str = "compile"
    perception: str = "lidar"
    variant: str = "step"
    timesteps: int = 160
    loss_timing: str = "deferred"
    lr: float = 1e-4
    lr_schedule: str = "constant"
    coef_v_pred: float = 0.0
    recovery_enabled: bool = False
    stop_after_acceptance: bool = True
    huber_delta: float = 1.0
    velocity_norm_weight: float = 0.8
    velocity_axis_weight: float = 0.6
    jerk_mean_weight: float = 1.0
    jerk_variance_weight: float = 1.0

    def validate(self):
        super().validate()
        if self.point_microbatch_envs not in (16, 32, 64, 128):
            raise ValueError("point_microbatch_envs must be 16, 32, 64 or 128")
        if self.point_precision not in ("fp32", "bf16") or self.point_encoder_backend not in (
            "torch",
            "compile",
            "triton",
        ):
            raise ValueError("Unsupported PointNet precision or encoder backend")
        if self.algorithm != "bptt" or self.variant != "step" or self.recurrent_model != "gru" or self.observation_normalization:
            raise ValueError("PointNet requires step BPTT, GRU and no depth normalization")
        if self.height_loss_mode != "goal_height_band" or self.approach_mode != "finite_difference":
            raise ValueError("PointNet requires the S0 height constraint and finite-difference closing speed")
        if self.loss_timing != "deferred" or self.velocity_loss_mode != "full_window":
            raise ValueError("PointNet uses deferred causal window losses")
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive")
        for name in ("velocity_norm_weight", "velocity_axis_weight", "jerk_mean_weight", "jerk_variance_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
