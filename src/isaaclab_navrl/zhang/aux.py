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
import warp as wp

DEFAULT_AUX_SECTORS = 8
DEFAULT_AUX_HALF_FOV_RAD = math.radians(36.0)
DEFAULT_AUX_MAX_RANGE_M = 12.0
DEFAULT_AUX_MAX_SPEED_MPS = 4.0
DEFAULT_AUX_VERTICAL_CLEARANCE_M = 2.0


@wp.kernel(enable_backward=False)
def _ego_dynamic_sector_targets_kernel(
    drone_position: wp.array2d(dtype=wp.float32),
    drone_velocity: wp.array2d(dtype=wp.float32),
    heading: wp.array3d(dtype=wp.float32),
    obstacle_position: wp.array2d(dtype=wp.float32),
    obstacle_velocity: wp.array2d(dtype=wp.float32),
    obstacle_size: wp.array2d(dtype=wp.float32),
    num_sectors: int,
    half_fov_rad: float,
    max_range_m: float,
    max_speed_mps: float,
    vertical_clearance_m: float,
    out_targets: wp.array3d(dtype=wp.float32),
):
    n = wp.tid()
    num_obs = obstacle_position.shape[0]
    px = drone_position[n, 0]
    py = drone_position[n, 1]
    pz = drone_position[n, 2]
    vx = drone_velocity[n, 0]
    vy = drone_velocity[n, 1]
    vz = drone_velocity[n, 2]

    # heading[n, :, 0] is ego +X in world; heading[n, :, 1] is ego +Y in world.
    h00 = heading[n, 0, 0]
    h10 = heading[n, 1, 0]
    h20 = heading[n, 2, 0]
    h01 = heading[n, 0, 1]
    h11 = heading[n, 1, 1]
    h21 = heading[n, 2, 1]

    for s in range(num_sectors):
        best_dist = max_range_m
        best_vx = wp.float32(0.0)
        best_vy = wp.float32(0.0)
        found = wp.int32(0)
        for m in range(num_obs):
            rx = obstacle_position[m, 0] - px
            ry = obstacle_position[m, 1] - py
            rz = obstacle_position[m, 2] - pz
            z_gap = wp.max(wp.abs(rz) - 0.5 * obstacle_size[m, 2], 0.0)
            if z_gap <= vertical_clearance_m:
                ex = rx * h00 + ry * h10 + rz * h20
                if ex > 0.0:
                    ey = rx * h01 + ry * h11 + rz * h21
                    angle = wp.atan2(ey, ex)
                    if wp.abs(angle) <= half_fov_rad:
                        rad_xy = 0.5 * wp.max(obstacle_size[m, 0], obstacle_size[m, 1])
                        dist = wp.max(wp.sqrt(ex * ex + ey * ey) - rad_xy, 0.0)
                        if dist < max_range_m:
                            norm_ang = (angle + half_fov_rad) / (2.0 * half_fov_rad)
                            sec = wp.clamp(int(wp.floor(norm_ang * float(num_sectors))), 0, num_sectors - 1)
                            if sec == s and dist < best_dist:
                                best_dist = dist
                                rvx_w = obstacle_velocity[m, 0] - vx
                                rvy_w = obstacle_velocity[m, 1] - vy
                                rvz_w = obstacle_velocity[m, 2] - vz
                                rvx_e = rvx_w * h00 + rvy_w * h10 + rvz_w * h20
                                rvy_e = rvx_w * h01 + rvy_w * h11 + rvz_w * h21
                                best_vx = wp.clamp(rvx_e / max_speed_mps, -1.0, 1.0)
                                best_vy = wp.clamp(rvy_e / max_speed_mps, -1.0, 1.0)
                                found = 1
        if found != 0:
            out_targets[n, s, 0] = wp.clamp(1.0 - best_dist / max_range_m, 0.0, 1.0)
            out_targets[n, s, 1] = best_vx
            out_targets[n, s, 2] = best_vy
        else:
            out_targets[n, s, 0] = 0.0
            out_targets[n, s, 1] = 0.0
            out_targets[n, s, 2] = 0.0


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

    if (
        drone_position.is_cuda
        and drone_position.dtype == torch.float32
        and obstacle_position.is_cuda
        and obstacle_position.dtype == torch.float32
    ):
        wp.init()
        wp.launch(
            _ego_dynamic_sector_targets_kernel,
            dim=batch_size,
            inputs=[
                wp.from_torch(drone_position.contiguous(), dtype=wp.float32),
                wp.from_torch(drone_velocity.contiguous(), dtype=wp.float32),
                wp.from_torch(heading.contiguous(), dtype=wp.float32),
                wp.from_torch(obstacle_position.contiguous(), dtype=wp.float32),
                wp.from_torch(obstacle_velocity.contiguous(), dtype=wp.float32),
                wp.from_torch(obstacle_size.contiguous(), dtype=wp.float32),
                int(num_sectors),
                float(half_fov_rad),
                float(max_range_m),
                float(max_speed_mps),
                float(vertical_clearance_m),
            ],
            outputs=[wp.from_torch(targets, dtype=wp.float32)],
            device=wp.device_from_torch(drone_position.device),
            stream=wp.stream_from_torch(drone_position.device),
        )
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
