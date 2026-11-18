"""Warp-fused CUDA kernels for v4 ego-compensated kinematic event field."""

from __future__ import annotations

import torch
import warp as wp


@wp.func
def _valid_depth(d: float, max_depth: float) -> bool:
    return wp.isfinite(d) and d > 0.0 and d < max_depth


@wp.func
def _rotate_vec(q: wp.vec4, v: wp.vec3) -> wp.vec3:
    xyz = wp.vec3(q[0], q[1], q[2])
    t = 2.0 * wp.cross(xyz, v)
    return v + q[3] * t + wp.cross(xyz, t)


@wp.func
def _inv_rotate_vec(q: wp.vec4, v: wp.vec3) -> wp.vec3:
    xyz = wp.vec3(-q[0], -q[1], -q[2])
    t = 2.0 * wp.cross(xyz, v)
    return v + q[3] * t + wp.cross(xyz, t)


@wp.func
def _backproject_pixel(
    d: float,
    u: float,
    v: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    pose_pos: wp.vec3,
    pose_quat: wp.vec4,
) -> wp.vec3:
    pt = wp.vec3(d, -(u + 0.5 - cx) * d / fx, -(v + 0.5 - cy) * d / fy)
    return _rotate_vec(pose_quat, pt) + pose_pos


@wp.func
def _project_world(
    wpt: wp.vec3,
    pose_pos: wp.vec3,
    pose_quat: wp.vec4,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    height: int,
    width: int,
    max_depth: float,
) -> wp.vec4:
    # Returns (u_float, v_float, x_depth, visible_flag_as_float)
    pt = _inv_rotate_vec(pose_quat, wpt - pose_pos)
    x = pt[0]
    safe_x = wp.max(x, 1.0e-6)
    u = cx - fx * pt[1] / safe_x - 0.5
    v = cy - fy * pt[2] / safe_x - 0.5
    vis = 1.0 if (x > 0.0 and x < max_depth and u >= 0.0 and u <= float(width - 1) and v >= 0.0 and v <= float(height - 1)) else 0.0
    return wp.vec4(u, v, x, vis)


@wp.func
def _read_clamped(depth: wp.array3d(dtype=wp.float32), b: int, v: int, u: int, height: int, width: int, max_depth: float) -> float:
    vc = wp.clamp(v, 0, height - 1)
    uc = wp.clamp(u, 0, width - 1)
    d = depth[b, vc, uc]
    return d if _valid_depth(d, max_depth) else max_depth


@wp.func
def _grad_sq_3d(
    depth: wp.array3d(dtype=wp.float32), b: int, v: int, u: int, height: int, width: int, max_depth: float
) -> float:
    gx = 0.5 * (_read_clamped(depth, b, v, u + 1, height, width, max_depth) - _read_clamped(depth, b, v, u - 1, height, width, max_depth))
    gy = 0.5 * (_read_clamped(depth, b, v + 1, u, height, width, max_depth) - _read_clamped(depth, b, v - 1, u, height, width, max_depth))
    return gx * gx + gy * gy


@wp.func
def _grad_sq_4d(
    depth_hist: wp.array4d(dtype=wp.float32), b: int, slot: int, v: int, u: int, height: int, width: int, max_depth: float
) -> float:
    ul = wp.clamp(u - 1, 0, width - 1)
    ur = wp.clamp(u + 1, 0, width - 1)
    vu = wp.clamp(v - 1, 0, height - 1)
    vd = wp.clamp(v + 1, 0, height - 1)
    dl = depth_hist[b, slot, v, ul]
    dr = depth_hist[b, slot, v, ur]
    du = depth_hist[b, slot, vu, u]
    dd = depth_hist[b, slot, vd, u]
    dl_v = dl if _valid_depth(dl, max_depth) else max_depth
    dr_v = dr if _valid_depth(dr, max_depth) else max_depth
    du_v = du if _valid_depth(du, max_depth) else max_depth
    dd_v = dd if _valid_depth(dd, max_depth) else max_depth
    gx = 0.5 * (dr_v - dl_v)
    gy = 0.5 * (dd_v - du_v)
    return gx * gx + gy * gy


@wp.kernel(enable_backward=False)
def _kinematic_event_field_kernel(
    depth_now: wp.array3d(dtype=wp.float32),
    depth_hist: wp.array4d(dtype=wp.float32),
    pose_now: wp.array2d(dtype=wp.float32),
    pose_hist: wp.array3d(dtype=wp.float32),
    K_now: wp.array3d(dtype=wp.float32),
    K_hist: wp.array4d(dtype=wp.float32),
    next_idx: wp.array(dtype=wp.int64),
    seen: wp.array(dtype=wp.int64),
    active: wp.array(dtype=wp.uint8),
    history_frames: int,
    max_depth: float,
    threshold: float,
    relative_threshold: float,
    sigma_multiplier: float,
    c0: float,
    c2: float,
    pos_sigma: float,
    rot_sigma: float,
    decay: float,
    step_dt: float,
    looming_max_rate: float,
    out_kinematic: wp.array4d(dtype=wp.float32),
    out_world_now: wp.array4d(dtype=wp.float32),
    out_world_old: wp.array4d(dtype=wp.float32),
    out_app_mask: wp.array3d(dtype=wp.uint8),
    out_dis_mask: wp.array3d(dtype=wp.uint8),
):
    b, v, u = wp.tid()
    height = depth_now.shape[1]
    width = depth_now.shape[2]

    p_now = wp.vec3(pose_now[b, 0], pose_now[b, 1], pose_now[b, 2])
    q_now = wp.vec4(pose_now[b, 3], pose_now[b, 4], pose_now[b, 5], pose_now[b, 6])
    fx_n = K_now[b, 0, 0]
    fy_n = K_now[b, 1, 1]
    cx_n = K_now[b, 0, 2]
    cy_n = K_now[b, 1, 2]

    d_n = depth_now[b, v, u]
    valid_n = _valid_depth(d_n, max_depth)
    g2_n = _grad_sq_3d(depth_now, b, v, u, height, width, max_depth)
    w_n = _backproject_pixel(d_n if valid_n else 0.0, float(u), float(v), fx_n, fy_n, cx_n, cy_n, p_now, q_now)
    out_world_now[b, v, u, 0] = w_n[0]
    out_world_now[b, v, u, 1] = w_n[1]
    out_world_now[b, v, u, 2] = w_n[2]

    # Stereo baseline variance at current pixel.
    d_n_val = d_n if valid_n else max_depth
    sig2_st_n = c0 + c2 * d_n_val * d_n_val

    seen_b = int(seen[b])
    max_k = wp.min(seen_b, history_frames)
    oldest_slot = int(next_idx[b])
    d_o_oldest = depth_hist[b, oldest_slot, v, u]
    valid_o_oldest = _valid_depth(d_o_oldest, max_depth)
    p_old0 = wp.vec3(pose_hist[b, oldest_slot, 0], pose_hist[b, oldest_slot, 1], pose_hist[b, oldest_slot, 2])
    q_old0 = wp.vec4(pose_hist[b, oldest_slot, 3], pose_hist[b, oldest_slot, 4], pose_hist[b, oldest_slot, 5], pose_hist[b, oldest_slot, 6])
    w_o0 = _backproject_pixel(
        d_o_oldest if valid_o_oldest else 0.0,
        float(u),
        float(v),
        K_hist[b, oldest_slot, 0, 0],
        K_hist[b, oldest_slot, 1, 1],
        K_hist[b, oldest_slot, 0, 2],
        K_hist[b, oldest_slot, 1, 2],
        p_old0,
        q_old0,
    )
    out_world_old[b, v, u, 0] = w_o0[0]
    out_world_old[b, v, u, 1] = w_o0[1]
    out_world_old[b, v, u, 2] = w_o0[2]

    wake_app = wp.float32(0.0)
    looming = wp.float32(0.0)
    app_any = wp.uint8(0)
    dis_any = wp.uint8(0)

    if active[b] != wp.uint8(0) and max_k >= 1:
        gamma = wp.float32(1.0)
        for k in range(1, history_frames + 1):
            if k <= max_k:
                slot = (int(next_idx[b]) + history_frames - k) % history_frames
                p_o = wp.vec3(pose_hist[b, slot, 0], pose_hist[b, slot, 1], pose_hist[b, slot, 2])
                q_o = wp.vec4(pose_hist[b, slot, 3], pose_hist[b, slot, 4], pose_hist[b, slot, 5], pose_hist[b, slot, 6])
                fx_o = K_hist[b, slot, 0, 0]
                fy_o = K_hist[b, slot, 1, 1]
                cx_o = K_hist[b, slot, 0, 2]
                cy_o = K_hist[b, slot, 1, 2]
                dt = float(k) * step_dt

                # 1) Forward check: current point into past view -> wake_appear + radial_looming.
                if valid_n:
                    proj_p = _project_world(w_n, p_o, q_o, fx_o, fy_o, cx_o, cy_o, height, width, max_depth)
                    if proj_p[3] > 0.5:
                        pu = wp.clamp(int(wp.floor(proj_p[0] + 0.5)), 0, width - 1)
                        pv = wp.clamp(int(wp.floor(proj_p[1] + 0.5)), 0, height - 1)
                        d_r = depth_hist[b, slot, pv, pu]
                        valid_r = _valid_depth(d_r, max_depth)
                        d_r_val = d_r if valid_r else max_depth

                        # Dual-endpoint spatial depth gradient (handles multi-pixel Sigma_xi jumps across edges).
                        g2_o = _grad_sq_4d(depth_hist, b, slot, pv, pu, height, width, max_depth)
                        g2_fwd = wp.max(g2_n, g2_o)

                        sig2_st_r = c0 + c2 * d_r_val * d_r_val
                        rot_term = proj_p[2] * rot_sigma
                        sig2_z_xi = pos_sigma * pos_sigma + rot_term * rot_term
                        j_px_r = fx_o * rot_sigma
                        j_px_p = fx_o * pos_sigma / wp.max(proj_p[2], 0.3)
                        sig2_edge = g2_fwd * (j_px_r * j_px_r + j_px_p * j_px_p + 0.04)
                        sig_tot = wp.sqrt(sig2_st_n + sig2_st_r + sig2_z_xi + sig2_edge)
                        gate = threshold + sigma_multiplier * sig_tot + relative_threshold * proj_p[2]
                        innov = d_r_val - proj_p[2]
                        if innov > gate:
                            app_any = wp.uint8(1)
                            s_app = wp.clamp((innov - gate) / wp.max(2.0 * sig_tot, 0.25), 0.0, 1.0)
                            w_app = gamma * s_app
                            if w_app > wake_app:
                                wake_app = w_app
                            # Interior continuous foreground surface looming (inverse TTC).
                            if valid_r and innov <= 8.0 * dt and g2_fwd < 0.35:
                                inv_ttc = (innov - gate) / (wp.max(d_n_val, 0.3) * dt)
                                loom_k = wp.clamp(inv_ttc / looming_max_rate, 0.0, 1.0)
                                if loom_k > looming:
                                    looming = loom_k

                # 2) Backward check: past pixel into current view -> wake_disappear.
                d_o_k = depth_hist[b, slot, v, u]
                if _valid_depth(d_o_k, max_depth):
                    g2_old_pix = _grad_sq_4d(depth_hist, b, slot, v, u, height, width, max_depth)
                    w_ok = _backproject_pixel(d_o_k, float(u), float(v), fx_o, fy_o, cx_o, cy_o, p_o, q_o)
                    proj_c = _project_world(w_ok, p_now, q_now, fx_n, fy_n, cx_n, cy_n, height, width, max_depth)
                    if proj_c[3] > 0.5:
                        cu_gs = wp.clamp(int(wp.floor(proj_c[0] + 0.5)), 0, width - 1)
                        cv_gs = wp.clamp(int(wp.floor(proj_c[1] + 0.5)), 0, height - 1)
                        dc = depth_now[b, cv_gs, cu_gs]
                        dc_val = dc if _valid_depth(dc, max_depth) else max_depth
                        g2_c = _grad_sq_3d(depth_now, b, cv_gs, cu_gs, height, width, max_depth)
                        g2_bwd = wp.max(g2_old_pix, g2_c)
                        sig2_st_o = c0 + c2 * d_o_k * d_o_k
                        sig2_st_c = c0 + c2 * dc_val * dc_val
                        rot_term_c = proj_c[2] * rot_sigma
                        sig2_z_xi_c = pos_sigma * pos_sigma + rot_term_c * rot_term_c
                        j_px_r_c = fx_n * rot_sigma
                        j_px_p_c = fx_n * pos_sigma / wp.max(proj_c[2], 0.3)
                        sig2_edge_c = g2_bwd * (j_px_r_c * j_px_r_c + j_px_p_c * j_px_p_c + 0.04)
                        sig_tot_c = wp.sqrt(sig2_st_o + sig2_st_c + sig2_z_xi_c + sig2_edge_c)
                        gate_c = threshold + sigma_multiplier * sig_tot_c + relative_threshold * proj_c[2]
                        innov_c = dc_val - proj_c[2]
                        if innov_c > gate_c:
                            dis_any = wp.uint8(1)
                            cu_i = wp.clamp(int(wp.floor(proj_c[0] + 0.5)), 0, width - 1)
                            cv_i = wp.clamp(int(wp.floor(proj_c[1] + 0.5)), 0, height - 1)
                            s_dis = gamma * wp.clamp((innov_c - gate_c) / wp.max(2.0 * sig_tot_c, 0.25), 0.0, 1.0)
                            if s_dis > 0.0:
                                wp.atomic_max(out_kinematic, b, 1, cv_i, cu_i, s_dis)
                gamma = gamma * decay

    out_kinematic[b, 0, v, u] = wake_app
    out_kinematic[b, 2, v, u] = looming
    out_app_mask[b, v, u] = app_any
    out_dis_mask[b, v, u] = dis_any


@wp.kernel(enable_backward=False)
def _kinematic_despeckle_kernel(
    raw_kinematic: wp.array4d(dtype=wp.float32),
    min_cluster_pixels: int,
    out_kinematic: wp.array4d(dtype=wp.float32),
):
    b, v, u = wp.tid()
    height = raw_kinematic.shape[2]
    width = raw_kinematic.shape[3]
    v0 = wp.max(0, v - 1)
    v1 = wp.min(height - 1, v + 1)
    u0 = wp.max(0, u - 1)
    u1 = wp.min(width - 1, u + 1)
    for c in range(3):
        val = raw_kinematic[b, c, v, u]
        if val > 0.0 and min_cluster_pixels > 1:
            cnt = wp.int32(0)
            for vv in range(v0, v1 + 1):
                for uu in range(u0, u1 + 1):
                    if raw_kinematic[b, c, vv, uu] > 0.0:
                        cnt += 1
            if cnt < min_cluster_pixels:
                val = 0.0
        out_kinematic[b, c, v, u] = val


class WarpKinematicFieldWorkspace:
    """Reusable GPU buffers for zero-allocation EG-TS + Radial Looming computation."""

    def __init__(self, count: int, height: int, width: int, device: torch.device):
        wp.init()
        self.count, self.height, self.width = count, height, width
        self.device = device
        self.raw_kinematic = torch.zeros((count, 3, height, width), device=device, dtype=torch.float32)
        self.kinematic = torch.zeros((count, 3, height, width), device=device, dtype=torch.float32)
        self.world_now = torch.empty((count, height, width, 3), device=device, dtype=torch.float32)
        self.world_old = torch.empty((count, height, width, 3), device=device, dtype=torch.float32)
        self.app_mask = torch.empty((count, height, width), device=device, dtype=torch.uint8)
        self.dis_mask = torch.empty((count, height, width), device=device, dtype=torch.uint8)

    def run(
        self,
        *,
        depth: torch.Tensor,
        depth_hist: torch.Tensor,
        pose: torch.Tensor,
        pose_hist: torch.Tensor,
        intrinsic: torch.Tensor,
        intrinsic_hist: torch.Tensor,
        next_idx: torch.Tensor,
        seen: torch.Tensor,
        active_u8: torch.Tensor,
        history_frames: int,
        min_cluster_pixels: int,
        max_depth: float,
        threshold: float,
        relative_threshold: float,
        sigma_multiplier: float,
        c0: float,
        c2: float,
        pos_sigma: float,
        rot_sigma: float,
        decay: float,
        step_dt: float,
        looming_max_rate: float,
    ):
        wp_device = wp.device_from_torch(depth.device)
        stream = wp.stream_from_torch(depth.device)
        target = self.raw_kinematic if min_cluster_pixels > 1 else self.kinematic
        target[:, 1].zero_()
        wp.launch(
            _kinematic_event_field_kernel,
            dim=(self.count, self.height, self.width),
            inputs=[
                wp.from_torch(depth, dtype=wp.float32),
                wp.from_torch(depth_hist, dtype=wp.float32),
                wp.from_torch(pose.contiguous(), dtype=wp.float32),
                wp.from_torch(pose_hist, dtype=wp.float32),
                wp.from_torch(intrinsic.contiguous(), dtype=wp.float32),
                wp.from_torch(intrinsic_hist, dtype=wp.float32),
                wp.from_torch(next_idx, dtype=wp.int64),
                wp.from_torch(seen, dtype=wp.int64),
                wp.from_torch(active_u8, dtype=wp.uint8),
                history_frames,
                max_depth,
                threshold,
                relative_threshold,
                sigma_multiplier,
                c0,
                c2,
                pos_sigma,
                rot_sigma,
                decay,
                step_dt,
                looming_max_rate,
            ],
            outputs=[
                wp.from_torch(target, dtype=wp.float32),
                wp.from_torch(self.world_now, dtype=wp.float32),
                wp.from_torch(self.world_old, dtype=wp.float32),
                wp.from_torch(self.app_mask, dtype=wp.uint8),
                wp.from_torch(self.dis_mask, dtype=wp.uint8),
            ],
            device=wp_device,
            stream=stream,
        )
        if min_cluster_pixels > 1:
            wp.launch(
                _kinematic_despeckle_kernel,
                dim=(self.count, self.height, self.width),
                inputs=[wp.from_torch(self.raw_kinematic, dtype=wp.float32), min_cluster_pixels],
                outputs=[wp.from_torch(self.kinematic, dtype=wp.float32)],
                device=wp_device,
                stream=stream,
            )
        return (
            self.app_mask.bool(),
            self.dis_mask.bool(),
            self.world_now,
            self.world_old,
            self.kinematic,
        )
