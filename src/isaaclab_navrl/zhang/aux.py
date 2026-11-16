"""Training-only ego-frame dynamic obstacle perception head and supervision.

Supervises the recurrent hidden state ``h`` to decode forward-FOV dynamic
obstacle proximity and relative velocity ``(proximity, rel_vx, rel_vy)`` across
horizontal angular sectors. Inference never calls this module or requires any
privileged obstacle state.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

DEFAULT_AUX_SECTORS = 8
DEFAULT_AUX_HALF_FOV_RAD = math.radians(36.0)
DEFAULT_AUX_MAX_RANGE_M = 12.0
DEFAULT_AUX_MAX_SPEED_MPS = 4.0
DEFAULT_AUX_VERTICAL_CLEARANCE_M = 2.0


@torch.no_grad()
def ego_dynamic_sector_targets(
    drone_position: torch.Tensor,
    drone_velocity: torch.Tensor,
    heading: torch.Tensor,
    obstacle_position: torch.Tensor,
    obstacle_velocity: torch.Tensor,
    obstacle_size: torch.Tensor,
    *,
    num_sectors: int = DEFAULT_AUX_SECTORS,
    half_fov_rad: float = DEFAULT_AUX_HALF_FOV_RAD,
    max_range_m: float = DEFAULT_AUX_MAX_RANGE_M,
    max_speed_mps: float = DEFAULT_AUX_MAX_SPEED_MPS,
    vertical_clearance_m: float = DEFAULT_AUX_VERTICAL_CLEARANCE_M,
) -> torch.Tensor:
    """Compute [N, num_sectors, 3] targets: (proximity in [0,1], rel_vx, rel_vy in [-1,1]).

    Args:
        drone_position: [N, 3] world position.
        drone_velocity: [N, 3] world velocity.
        heading: [N, 3, 3] body/heading-to-world rotation matrix (+X forward, +Y left, +Z up).
        obstacle_position: [M, 3] shared dynamic obstacle world centers.
        obstacle_velocity: [M, 3] shared dynamic obstacle world velocities.
        obstacle_size: [M, 3] obstacle dimensions (width_x, width_y, height_z).
    """
    batch_size = drone_position.shape[0]
    targets = drone_position.new_zeros((batch_size, num_sectors, 3))
    if obstacle_position.numel() == 0:
        return targets

    rel_world = obstacle_position[None, :, :] - drone_position[:, None, :]
    rel_ego = torch.einsum("nmi,nij->nmj", rel_world, heading)
    rel_vel_world = obstacle_velocity[None, :, :] - drone_velocity[:, None, :]
    rel_vel_ego = torch.einsum("nmi,nij->nmj", rel_vel_world, heading)

    half_height = 0.5 * obstacle_size[:, 2]
    z_gap = (rel_world[..., 2].abs() - half_height[None, :]).clamp_min(0.0)
    z_valid = z_gap <= vertical_clearance_m

    radius_xy = 0.5 * obstacle_size[:, :2].amax(dim=-1)
    center_dist_xy = rel_ego[..., :2].norm(dim=-1)
    surface_dist = (center_dist_xy - radius_xy[None, :]).clamp_min(0.0)

    angle = torch.atan2(rel_ego[..., 1], rel_ego[..., 0])
    in_fov = z_valid & (rel_ego[..., 0] > 0.0) & (angle.abs() <= half_fov_rad) & (surface_dist < max_range_m)

    normalized_angle = (angle + half_fov_rad) / (2.0 * half_fov_rad)
    sector = (normalized_angle * num_sectors).floor().long().clamp(0, num_sectors - 1)

    inf = torch.full_like(surface_dist, float("inf"))
    masked_dist = torch.where(in_fov, surface_dist, inf)

    for s in range(num_sectors):
        sector_dist = torch.where(sector == s, masked_dist, inf)
        nearest_dist, nearest_idx = sector_dist.min(dim=-1)
        active = torch.isfinite(nearest_dist)
        proximity = torch.where(active, (1.0 - nearest_dist / max_range_m).clamp(0.0, 1.0), 0.0)
        gather_idx = nearest_idx[:, None, None].expand(-1, 1, 2)
        nearest_vel = rel_vel_ego[..., :2].gather(1, gather_idx)[:, 0, :]
        scaled_vel = (nearest_vel / max_speed_mps).clamp(-1.0, 1.0)
        targets[:, s, 0] = proximity
        targets[:, s, 1] = torch.where(active, scaled_vel[:, 0], 0.0)
        targets[:, s, 2] = torch.where(active, scaled_vel[:, 1], 0.0)

    return targets


def dynamic_aux_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute proximity + masked relative-velocity Smooth-L1 loss over [T, N, S, 3] or [N, S, 3]."""
    if prediction.shape != target.shape or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must have matching [..., S, 3] shape")
    pred_proximity = torch.sigmoid(prediction[..., 0])
    target_proximity = target[..., 0]
    proximity_loss = F.smooth_l1_loss(pred_proximity, target_proximity, beta=0.1)

    occupied = target_proximity > 0.05
    if occupied.any():
        pred_vel = torch.tanh(prediction[..., 1:3])[occupied]
        target_vel = target[..., 1:3][occupied]
        velocity_loss = F.smooth_l1_loss(pred_vel, target_vel, beta=0.1)
    else:
        velocity_loss = prediction[..., 1:3].sum() * 0.0

    return proximity_loss + velocity_loss
