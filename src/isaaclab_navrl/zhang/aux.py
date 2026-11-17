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
DEFAULT_V4_AUX_SECTORS = 6
DEFAULT_V4_AUX_HALF_FOV_RAD = math.radians(30.0)
DEFAULT_AUX_MAX_RANGE_M = 12.0
DEFAULT_AUX_MAX_SPEED_MPS = 4.0
DEFAULT_AUX_VERTICAL_CLEARANCE_M = 2.0
DEFAULT_MAX_INV_TTC = 2.5
DEFAULT_LOS_TOLERANCE_M = 0.6


@wp.kernel(enable_backward=False)
def _ego_inv_ttc_los_targets_kernel(
    drone_position: wp.array2d(dtype=wp.float32),
    drone_velocity: wp.array2d(dtype=wp.float32),
    heading: wp.array3d(dtype=wp.float32),
    obstacle_position: wp.array2d(dtype=wp.float32),
    obstacle_velocity: wp.array2d(dtype=wp.float32),
    obstacle_size: wp.array2d(dtype=wp.float32),
    depth_map: wp.array3d(dtype=wp.float32),
    has_depth: int,
    num_sectors: int,
    half_fov_rad: float,
    max_range_m: float,
    max_speed_mps: float,
    vertical_clearance_m: float,
    max_inv_ttc: float,
    los_tolerance_m: float,
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

    h00 = heading[n, 0, 0]
    h10 = heading[n, 1, 0]
    h20 = heading[n, 2, 0]
    h01 = heading[n, 0, 1]
    h11 = heading[n, 1, 1]
    h21 = heading[n, 2, 1]
    h02 = heading[n, 0, 2]
    h12 = heading[n, 1, 2]
    h22 = heading[n, 2, 2]

    dh = depth_map.shape[1]
    dw = depth_map.shape[2]
    fx_d = float(dw) / (2.0 * wp.tan(half_fov_rad))
    cx_d = 0.5 * float(dw)
    cy_d = 0.5 * float(dh)

    for s in range(num_sectors):
        best_risk = wp.float32(0.0)
        best_vx = wp.float32(0.0)
        best_vy = wp.float32(0.0)
        for m in range(num_obs):
            rx = obstacle_position[m, 0] - px
            ry = obstacle_position[m, 1] - py
            rz = obstacle_position[m, 2] - pz
            z_gap = wp.max(wp.abs(rz) - 0.5 * obstacle_size[m, 2], 0.0)
            if z_gap <= vertical_clearance_m:
                ex = rx * h00 + ry * h10 + rz * h20
                if ex > 0.0:
                    ey = rx * h01 + ry * h11 + rz * h21
                    ez = rx * h02 + ry * h12 + rz * h22
                    angle = wp.atan2(ey, ex)
                    if wp.abs(angle) <= half_fov_rad:
                        rad_xy = 0.5 * wp.max(obstacle_size[m, 0], obstacle_size[m, 1])
                        center_xy = wp.sqrt(ex * ex + ey * ey)
                        dist_xy = wp.max(center_xy - rad_xy, 0.0)
                        if dist_xy < max_range_m:
                            norm_ang = (angle + half_fov_rad) / (2.0 * half_fov_rad)
                            sec = wp.clamp(int(wp.floor(norm_ang * float(num_sectors))), 0, num_sectors - 1)
                            if sec == s:
                                visible = 1
                                if has_depth != 0:
                                    u_obs = wp.clamp(int(wp.floor(cx_d - fx_d * (ey / wp.max(ex, 1.0e-3)))), 0, dw - 1)
                                    v_obs = wp.clamp(int(wp.floor(cy_d - fx_d * (ez / wp.max(ex, 1.0e-3)))), 0, dh - 1)
                                    min_obs_d = wp.float32(1.0e6)
                                    for dv in range(-1, 2):
                                        for du in range(-1, 2):
                                            vv = wp.clamp(v_obs + dv, 0, dh - 1)
                                            uu = wp.clamp(u_obs + du, 0, dw - 1)
                                            dv_val = depth_map[n, vv, uu]
                                            if wp.isfinite(dv_val) and dv_val > 0.0 and dv_val < min_obs_d:
                                                min_obs_d = dv_val
                                    if ex - rad_xy > min_obs_d + los_tolerance_m:
                                        visible = 0
                                if visible != 0:
                                    rvx_w = obstacle_velocity[m, 0] - vx
                                    rvy_w = obstacle_velocity[m, 1] - vy
                                    rvz_w = obstacle_velocity[m, 2] - vz
                                    norm_r = wp.max(wp.sqrt(rx * rx + ry * ry + rz * rz), 1.0e-3)
                                    v_close = wp.max(-(rx * rvx_w + ry * rvy_w + rz * rvz_w) / norm_r, 0.0)
                                    gap = wp.max(center_xy - rad_xy, 0.25)
                                    inv_ttc = v_close / gap
                                    if inv_ttc > best_risk:
                                        best_risk = inv_ttc
                                        rvx_e = rvx_w * h00 + rvy_w * h10 + rvz_w * h20
                                        rvy_e = rvx_w * h01 + rvy_w * h11 + rvz_w * h21
                                        best_vx = wp.clamp(rvx_e / max_speed_mps, -1.0, 1.0)
                                        best_vy = wp.clamp(rvy_e / max_speed_mps, -1.0, 1.0)
        risk_norm = wp.clamp(best_risk / max_inv_ttc, 0.0, 1.0)
        if risk_norm > 0.0:
            out_targets[n, s, 0] = risk_norm
            out_targets[n, s, 1] = best_vx
            out_targets[n, s, 2] = best_vy
        else:
            out_targets[n, s, 0] = 0.0
            out_targets[n, s, 1] = 0.0
            out_targets[n, s, 2] = 0.0


@wp.kernel(enable_backward=False)
def _forward_closing_risk_kernel(
    drone_position: wp.array2d(dtype=wp.float32),
    drone_velocity: wp.array2d(dtype=wp.float32),
    heading: wp.array3d(dtype=wp.float32),
    obstacle_position: wp.array2d(dtype=wp.float32),
    obstacle_velocity: wp.array2d(dtype=wp.float32),
    obstacle_size: wp.array2d(dtype=wp.float32),
    half_fov_rad: float,
    max_range_m: float,
    out_risk: wp.array(dtype=wp.float32),
):
    n = wp.tid()
    num_obs = obstacle_position.shape[0]
    px = drone_position[n, 0]
    py = drone_position[n, 1]
    pz = drone_position[n, 2]
    vx = drone_velocity[n, 0]
    vy = drone_velocity[n, 1]
    vz = drone_velocity[n, 2]

    h00 = heading[n, 0, 0]
    h10 = heading[n, 1, 0]
    h20 = heading[n, 2, 0]
    h01 = heading[n, 0, 1]
    h11 = heading[n, 1, 1]
    h21 = heading[n, 2, 1]

    best = wp.float32(0.0)
    for m in range(num_obs):
        rx = obstacle_position[m, 0] - px
        ry = obstacle_position[m, 1] - py
        rz = obstacle_position[m, 2] - pz
        z_gap = wp.max(wp.abs(rz) - 0.5 * obstacle_size[m, 2], 0.0)
        if z_gap <= 1.5:
            ex = rx * h00 + ry * h10 + rz * h20
            ey = rx * h01 + ry * h11 + rz * h21
            rad_xy = 0.5 * wp.max(obstacle_size[m, 0], obstacle_size[m, 1])
            center_xy = wp.sqrt(ex * ex + ey * ey)
            surf_gap = center_xy - rad_xy
            angle = wp.atan2(ey, ex)
            if ex > -rad_xy and wp.abs(angle) <= half_fov_rad and surf_gap < max_range_m:
                rvx_w = obstacle_velocity[m, 0] - vx
                rvy_w = obstacle_velocity[m, 1] - vy
                rvz_w = obstacle_velocity[m, 2] - vz
                norm_r = wp.max(wp.sqrt(rx * rx + ry * ry + rz * rz), 1.0e-3)
                v_close = wp.max(-(rx * rvx_w + ry * rvy_w + rz * rvz_w) / norm_r, 0.0)
                if v_close > 0.0:
                    inv_ttc = v_close / wp.max(surf_gap, 0.3)
                    risk_m = wp.max(inv_ttc - 0.35, 0.0) * wp.exp(-wp.max(surf_gap, 0.0) / 3.0)
                    if risk_m > best:
                        best = risk_m
    out_risk[n] = best


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
    inv_ttc_mode: bool = False,
    depth_map: torch.Tensor | None = None,
    max_inv_ttc: float = DEFAULT_MAX_INV_TTC,
    los_tolerance_m: float = DEFAULT_LOS_TOLERANCE_M,
) -> torch.Tensor:
    """Compute [N, num_sectors, 3] dynamic obstacle sector targets.

    In legacy mode (``inv_ttc_mode=False``), selects nearest obstacle and encodes
    ``(proximity, rel_vx, rel_vy)``. In v4 mode (``inv_ttc_mode=True``), selects
    the LoS-visible obstacle with highest positive inverse TTC and encodes
    ``(risk_score, rel_vx, rel_vy)``.
    """
    batch_size = drone_position.shape[0]
    targets = drone_position.new_zeros((batch_size, num_sectors, 3))
    if obstacle_position.numel() == 0:
        return targets

    if inv_ttc_mode:
        if (
            drone_position.is_cuda
            and drone_position.dtype == torch.float32
            and obstacle_position.is_cuda
            and obstacle_position.dtype == torch.float32
        ):
            wp.init()
            d_in = (
                depth_map[:, 0]
                if depth_map is not None and depth_map.ndim == 4
                else (depth_map if depth_map is not None else drone_position.new_zeros((batch_size, 1, 1)))
            )
            wp.launch(
                _ego_inv_ttc_los_targets_kernel,
                dim=batch_size,
                inputs=[
                    wp.from_torch(drone_position.contiguous(), dtype=wp.float32),
                    wp.from_torch(drone_velocity.contiguous(), dtype=wp.float32),
                    wp.from_torch(heading.contiguous(), dtype=wp.float32),
                    wp.from_torch(obstacle_position.contiguous(), dtype=wp.float32),
                    wp.from_torch(obstacle_velocity.contiguous(), dtype=wp.float32),
                    wp.from_torch(obstacle_size.contiguous(), dtype=wp.float32),
                    wp.from_torch(d_in.contiguous(), dtype=wp.float32),
                    1 if depth_map is not None else 0,
                    int(num_sectors),
                    float(half_fov_rad),
                    float(max_range_m),
                    float(max_speed_mps),
                    float(vertical_clearance_m),
                    float(max_inv_ttc),
                    float(los_tolerance_m),
                ],
                outputs=[wp.from_torch(targets, dtype=wp.float32)],
                device=wp.device_from_torch(drone_position.device),
                stream=wp.stream_from_torch(drone_position.device),
            )
            return targets
        return _ego_inv_ttc_los_targets_cpu(
            drone_position,
            drone_velocity,
            heading,
            obstacle_position,
            obstacle_velocity,
            obstacle_size,
            depth_map=depth_map,
            num_sectors=num_sectors,
            half_fov_rad=half_fov_rad,
            max_range_m=max_range_m,
            max_speed_mps=max_speed_mps,
            vertical_clearance_m=vertical_clearance_m,
            max_inv_ttc=max_inv_ttc,
            los_tolerance_m=los_tolerance_m,
        )

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


@torch.no_grad()
def _ego_inv_ttc_los_targets_cpu(
    drone_position: torch.Tensor,
    drone_velocity: torch.Tensor,
    heading: torch.Tensor,
    obstacle_position: torch.Tensor,
    obstacle_velocity: torch.Tensor,
    obstacle_size: torch.Tensor,
    *,
    depth_map: torch.Tensor | None,
    num_sectors: int,
    half_fov_rad: float,
    max_range_m: float,
    max_speed_mps: float,
    vertical_clearance_m: float,
    max_inv_ttc: float,
    los_tolerance_m: float,
) -> torch.Tensor:
    batch_size = drone_position.shape[0]
    targets = drone_position.new_zeros((batch_size, num_sectors, 3))
    rel_world = obstacle_position[None, :, :] - drone_position[:, None, :]
    rel_ego = torch.einsum("nmi,nij->nmj", rel_world, heading)
    rel_vel_world = obstacle_velocity[None, :, :] - drone_velocity[:, None, :]
    rel_vel_ego = torch.einsum("nmi,nij->nmj", rel_vel_world, heading)

    half_height = 0.5 * obstacle_size[:, 2]
    z_valid = (rel_world[..., 2].abs() - half_height[None, :]).clamp_min(0.0) <= vertical_clearance_m
    rad_xy = 0.5 * obstacle_size[:, :2].amax(dim=-1)
    center_xy = rel_ego[..., :2].norm(dim=-1)
    dist_xy = (center_xy - rad_xy[None, :]).clamp_min(0.0)
    angle = torch.atan2(rel_ego[..., 1], rel_ego[..., 0])
    visible = z_valid & (rel_ego[..., 0] > 0.0) & (angle.abs() <= half_fov_rad) & (dist_xy < max_range_m)

    if depth_map is not None:
        d_in = depth_map[:, 0] if depth_map.ndim == 4 else depth_map
        dh, dw = d_in.shape[-2:]
        fx_d = dw / (2.0 * math.tan(half_fov_rad))
        ex = rel_ego[..., 0].clamp_min(1e-3)
        u_obs = (0.5 * dw - fx_d * (rel_ego[..., 1] / ex)).floor().long().clamp(0, dw - 1)
        v_obs = (0.5 * dh - fx_d * (rel_ego[..., 2] / ex)).floor().long().clamp(0, dh - 1)
        # 3x3 valid min depth in image
        valid_d = torch.isfinite(d_in) & (d_in > 0)
        d_clean = torch.where(valid_d, d_in, torch.full_like(d_in, 1e6))
        d_min3 = -F.max_pool2d(-d_clean[:, None], kernel_size=3, stride=1, padding=1)[:, 0]
        sampled = d_min3.flatten(1).gather(1, v_obs * dw + u_obs)
        visible &= (rel_ego[..., 0] - rad_xy[None, :]) <= (sampled + los_tolerance_m)

    norm_r = rel_world.norm(dim=-1).clamp_min(1e-3)
    v_close = (-(rel_world * rel_vel_world).sum(dim=-1) / norm_r).clamp_min(0.0)
    gap = (center_xy - rad_xy[None, :]).clamp_min(0.25)
    inv_ttc = torch.where(visible, v_close / gap, 0.0)

    normalized_angle = (angle + half_fov_rad) / (2.0 * half_fov_rad)
    sector = (normalized_angle * num_sectors).floor().long().clamp(0, num_sectors - 1)

    for s in range(num_sectors):
        sec_risk = torch.where(visible & (sector == s), inv_ttc, -1.0)
        best_risk, best_idx = sec_risk.max(dim=-1)
        risk_norm = (best_risk.clamp_min(0.0) / max_inv_ttc).clamp(0.0, 1.0)
        active = risk_norm > 0.0
        gather_idx = best_idx[:, None, None].expand(-1, 1, 2)
        best_vel = rel_vel_ego[..., :2].gather(1, gather_idx)[:, 0, :]
        scaled_vel = (best_vel / max_speed_mps).clamp(-1.0, 1.0)
        targets[:, s, 0] = risk_norm
        targets[:, s, 1] = torch.where(active, scaled_vel[:, 0], 0.0)
        targets[:, s, 2] = torch.where(active, scaled_vel[:, 1], 0.0)

    return targets


@torch.no_grad()
def forward_closing_risk(
    drone_position: torch.Tensor,
    drone_velocity: torch.Tensor,
    heading: torch.Tensor,
    obstacle_position: torch.Tensor,
    obstacle_velocity: torch.Tensor,
    obstacle_size: torch.Tensor,
    *,
    half_fov_rad: float = math.radians(45.0),
    max_range_m: float = 6.0,
) -> torch.Tensor:
    """Compute scalar forward-cone closing risk R_closing[N] for temporal yielding modulation."""
    batch_size = drone_position.shape[0]
    risk = drone_position.new_zeros((batch_size,))
    if obstacle_position.numel() == 0:
        return risk

    if (
        drone_position.is_cuda
        and drone_position.dtype == torch.float32
        and obstacle_position.is_cuda
        and obstacle_position.dtype == torch.float32
    ):
        wp.init()
        wp.launch(
            _forward_closing_risk_kernel,
            dim=batch_size,
            inputs=[
                wp.from_torch(drone_position.contiguous(), dtype=wp.float32),
                wp.from_torch(drone_velocity.contiguous(), dtype=wp.float32),
                wp.from_torch(heading.contiguous(), dtype=wp.float32),
                wp.from_torch(obstacle_position.contiguous(), dtype=wp.float32),
                wp.from_torch(obstacle_velocity.contiguous(), dtype=wp.float32),
                wp.from_torch(obstacle_size.contiguous(), dtype=wp.float32),
                float(half_fov_rad),
                float(max_range_m),
            ],
            outputs=[wp.from_torch(risk, dtype=wp.float32)],
            device=wp.device_from_torch(drone_position.device),
            stream=wp.stream_from_torch(drone_position.device),
        )
        return risk

    rel_world = obstacle_position[None, :, :] - drone_position[:, None, :]
    rel_ego = torch.einsum("nmi,nij->nmj", rel_world, heading)
    rel_vel_world = obstacle_velocity[None, :, :] - drone_velocity[:, None, :]
    half_h = 0.5 * obstacle_size[:, 2]
    z_ok = (rel_world[..., 2].abs() - half_h[None, :]).clamp_min(0.0) <= 1.5
    rad_xy = 0.5 * obstacle_size[:, :2].amax(dim=-1)
    surf_gap = rel_ego[..., :2].norm(dim=-1) - rad_xy[None, :]
    angle = torch.atan2(rel_ego[..., 1], rel_ego[..., 0])
    in_cone = z_ok & (rel_ego[..., 0] > -rad_xy[None, :]) & (angle.abs() <= half_fov_rad) & (surf_gap < max_range_m)
    norm_r = rel_world.norm(dim=-1).clamp_min(1e-3)
    v_close = (-(rel_world * rel_vel_world).sum(dim=-1) / norm_r).clamp_min(0.0)
    inv_ttc = v_close / surf_gap.clamp_min(0.3)
    risk_m = (inv_ttc - 0.35).clamp_min(0.0) * (-surf_gap.clamp_min(0.0) / 3.0).exp()
    return torch.where(in_cone, risk_m, 0.0).amax(dim=-1)


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
