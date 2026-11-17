"""Warp-fused CUDA kernels for DepthEventDetector and dual-channel splatting.

Keeps fixed-shape GPU buffers across steps so per-step event detection avoids
allocating intermediate [B, H, W, 3] PyTorch tensors or launching dozens of
unfused elementwise/pooling kernels.
"""

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
def _near_min_pixel(
    depth: wp.array3d(dtype=wp.float32),
    b: int,
    v: int,
    u: int,
    height: int,
    width: int,
    radius: int,
    max_depth: float,
) -> float:
    best = max_depth
    v0 = wp.max(0, v - radius)
    v1 = wp.min(height - 1, v + radius)
    u0 = wp.max(0, u - radius)
    u1 = wp.min(width - 1, u + radius)
    for vv in range(v0, v1 + 1):
        for uu in range(u0, u1 + 1):
            d = depth[b, vv, uu]
            val = d if _valid_depth(d, max_depth) else max_depth
            if val < best:
                best = val
    return best


@wp.kernel(enable_backward=False)
def _edge_ref_kernel(
    depth: wp.array3d(dtype=wp.float32),
    match_radius: int,
    edge_suppression: int,
    max_depth: float,
    out_ref: wp.array3d(dtype=wp.float32),
):
    b, v, u = wp.tid()
    height = depth.shape[1]
    width = depth.shape[2]
    near_1 = _near_min_pixel(depth, b, v, u, height, width, match_radius, max_depth)
    if edge_suppression == 0:
        out_ref[b, v, u] = near_1
        return

    far_1 = wp.float32(0.0)
    v0 = wp.max(0, v - 1)
    v1 = wp.min(height - 1, v + 1)
    u0 = wp.max(0, u - 1)
    u1 = wp.min(width - 1, u + 1)
    for vv in range(v0, v1 + 1):
        for uu in range(u0, u1 + 1):
            d = depth[b, vv, uu]
            val = d if _valid_depth(d, max_depth) else max_depth
            if val > far_1:
                far_1 = val

    if far_1 - near_1 > 0.75:
        wide_radius = wp.max(match_radius + 1, 2)
        out_ref[b, v, u] = _near_min_pixel(depth, b, v, u, height, width, wide_radius, max_depth)
    else:
        out_ref[b, v, u] = near_1


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
def _margin_val(
    depth_val: float,
    threshold: float,
    relative_threshold: float,
    sigma_multiplier: float,
    base_var: float,
    pose_rot_sigma: float,
) -> float:
    rot_term = depth_val * pose_rot_sigma
    sigma = wp.sqrt(base_var + rot_term * rot_term)
    return threshold + sigma_multiplier * sigma + relative_threshold * depth_val


@wp.kernel(enable_backward=False)
def _depth_event_detect_kernel(
    depth_now: wp.array3d(dtype=wp.float32),
    depth_old: wp.array3d(dtype=wp.float32),
    old_ref: wp.array3d(dtype=wp.float32),
    now_ref: wp.array3d(dtype=wp.float32),
    pose_now: wp.array2d(dtype=wp.float32),
    pose_old: wp.array2d(dtype=wp.float32),
    K_now: wp.array3d(dtype=wp.float32),
    K_old: wp.array3d(dtype=wp.float32),
    active: wp.array(dtype=wp.uint8),
    seen: wp.array(dtype=wp.int64),
    history_frames: int,
    max_depth: float,
    threshold: float,
    relative_threshold: float,
    sigma_multiplier: float,
    base_var: float,
    pose_rot_sigma: float,
    out_app_mask: wp.array3d(dtype=wp.uint8),
    out_dis_mask: wp.array3d(dtype=wp.uint8),
    out_app_str: wp.array3d(dtype=wp.float32),
    out_dis_str: wp.array3d(dtype=wp.float32),
    out_world_now: wp.array4d(dtype=wp.float32),
    out_world_old: wp.array4d(dtype=wp.float32),
    out_dis_uv: wp.array4d(dtype=wp.int32),
):
    b, v, u = wp.tid()
    height = depth_now.shape[1]
    width = depth_now.shape[2]
    eligible = active[b] != wp.uint8(0) and seen[b] >= wp.int64(history_frames)

    p_now = wp.vec3(pose_now[b, 0], pose_now[b, 1], pose_now[b, 2])
    q_now = wp.vec4(pose_now[b, 3], pose_now[b, 4], pose_now[b, 5], pose_now[b, 6])
    p_old = wp.vec3(pose_old[b, 0], pose_old[b, 1], pose_old[b, 2])
    q_old = wp.vec4(pose_old[b, 3], pose_old[b, 4], pose_old[b, 5], pose_old[b, 6])

    fx_n = K_now[b, 0, 0]
    fy_n = K_now[b, 1, 1]
    cx_n = K_now[b, 0, 2]
    cy_n = K_now[b, 1, 2]

    fx_o = K_old[b, 0, 0]
    fy_o = K_old[b, 1, 1]
    cx_o = K_old[b, 0, 2]
    cy_o = K_old[b, 1, 2]

    d_n = depth_now[b, v, u]
    valid_n = _valid_depth(d_n, max_depth)
    w_n = _backproject_pixel(d_n if valid_n else 0.0, float(u), float(v), fx_n, fy_n, cx_n, cy_n, p_now, q_now)
    out_world_now[b, v, u, 0] = w_n[0]
    out_world_now[b, v, u, 1] = w_n[1]
    out_world_now[b, v, u, 2] = w_n[2]

    d_o = depth_old[b, v, u]
    valid_o = _valid_depth(d_o, max_depth)
    w_o = _backproject_pixel(d_o if valid_o else 0.0, float(u), float(v), fx_o, fy_o, cx_o, cy_o, p_old, q_old)
    out_world_old[b, v, u, 0] = w_o[0]
    out_world_old[b, v, u, 1] = w_o[1]
    out_world_old[b, v, u, 2] = w_o[2]

    app = wp.uint8(0)
    app_s = wp.float32(0.0)
    if eligible and valid_n:
        proj_prev = _project_world(w_n, p_old, q_old, fx_o, fy_o, cx_o, cy_o, height, width, max_depth)
        if proj_prev[3] > 0.5:
            pu = wp.clamp(int(wp.floor(proj_prev[0] + 0.5)), 0, width - 1)
            pv = wp.clamp(int(wp.floor(proj_prev[1] + 0.5)), 0, height - 1)
            o_near = old_ref[b, pv, pu]
            m_n = _margin_val(proj_prev[2], threshold, relative_threshold, sigma_multiplier, base_var, pose_rot_sigma)
            if proj_prev[2] + m_n < o_near:
                app = wp.uint8(1)
                app_s = wp.clamp((o_near - proj_prev[2]) / 0.5, 0.0, 1.0)

    dis = wp.uint8(0)
    dis_s = wp.float32(0.0)
    cu_i = wp.int32(0)
    cv_i = wp.int32(0)
    c_vis = wp.int32(0)
    if eligible and valid_o:
        proj_cur = _project_world(w_o, p_now, q_now, fx_n, fy_n, cx_n, cy_n, height, width, max_depth)
        if proj_cur[3] > 0.5:
            cu_i = wp.clamp(int(wp.round(proj_cur[0])), 0, width - 1)
            cv_i = wp.clamp(int(wp.round(proj_cur[1])), 0, height - 1)
            c_vis = 1
            cu_gs = wp.clamp(int(wp.floor(proj_cur[0] + 0.5)), 0, width - 1)
            cv_gs = wp.clamp(int(wp.floor(proj_cur[1] + 0.5)), 0, height - 1)
            n_near = now_ref[b, cv_gs, cu_gs]
            m_o = _margin_val(proj_cur[2], threshold, relative_threshold, sigma_multiplier, base_var, pose_rot_sigma)
            if proj_cur[2] + m_o < n_near:
                dis = wp.uint8(1)
                dis_s = wp.clamp((n_near - proj_cur[2]) / 0.5, 0.0, 1.0)

    out_app_mask[b, v, u] = app
    out_dis_mask[b, v, u] = dis
    out_app_str[b, v, u] = app_s
    out_dis_str[b, v, u] = dis_s
    out_dis_uv[b, v, u, 0] = cu_i
    out_dis_uv[b, v, u, 1] = cv_i
    out_dis_uv[b, v, u, 2] = c_vis


@wp.kernel(enable_backward=False)
def _filter_and_splat_kernel(
    raw_app_mask: wp.array3d(dtype=wp.uint8),
    raw_dis_mask: wp.array3d(dtype=wp.uint8),
    raw_app_str: wp.array3d(dtype=wp.float32),
    raw_dis_str: wp.array3d(dtype=wp.float32),
    dis_uv: wp.array4d(dtype=wp.int32),
    min_cluster_pixels: int,
    out_app_mask: wp.array3d(dtype=wp.uint8),
    out_dis_mask: wp.array3d(dtype=wp.uint8),
    out_app_str: wp.array3d(dtype=wp.float32),
    out_dis_str: wp.array3d(dtype=wp.float32),
    out_dual: wp.array4d(dtype=wp.float32),
):
    b, v, u = wp.tid()
    height = raw_app_mask.shape[1]
    width = raw_app_mask.shape[2]
    v0 = wp.max(0, v - 1)
    v1 = wp.min(height - 1, v + 1)
    u0 = wp.max(0, u - 1)
    u1 = wp.min(width - 1, u + 1)

    app_keep = raw_app_mask[b, v, u]
    if app_keep != wp.uint8(0) and min_cluster_pixels > 1:
        cnt_a = wp.int32(0)
        for vv in range(v0, v1 + 1):
            for uu in range(u0, u1 + 1):
                if raw_app_mask[b, vv, uu] != wp.uint8(0):
                    cnt_a += 1
        if cnt_a < min_cluster_pixels:
            app_keep = wp.uint8(0)

    dis_keep = raw_dis_mask[b, v, u]
    if dis_keep != wp.uint8(0) and min_cluster_pixels > 1:
        cnt_d = wp.int32(0)
        for vv in range(v0, v1 + 1):
            for uu in range(u0, u1 + 1):
                if raw_dis_mask[b, vv, uu] != wp.uint8(0):
                    cnt_d += 1
        if cnt_d < min_cluster_pixels:
            dis_keep = wp.uint8(0)

    app_s = raw_app_str[b, v, u] if app_keep != wp.uint8(0) else 0.0
    dis_s = raw_dis_str[b, v, u] if dis_keep != wp.uint8(0) else 0.0
    out_app_mask[b, v, u] = app_keep
    out_dis_mask[b, v, u] = dis_keep
    out_app_str[b, v, u] = app_s
    out_dis_str[b, v, u] = dis_s
    out_dual[b, 0, v, u] = app_s
    if dis_keep != wp.uint8(0) and dis_uv[b, v, u, 2] != 0 and dis_s > 0.0:
        wp.atomic_max(out_dual, b, 1, dis_uv[b, v, u, 1], dis_uv[b, v, u, 0], dis_s)


class WarpDepthEventWorkspace:
    """Reusable GPU buffers for zero-allocation DepthEventDetector.update calls."""

    def __init__(self, count: int, height: int, width: int, device: torch.device):
        wp.init()
        self.count, self.height, self.width = count, height, width
        self.device = device
        self.old_ref = torch.empty((count, height, width), device=device, dtype=torch.float32)
        self.now_ref = torch.empty((count, height, width), device=device, dtype=torch.float32)
        self.raw_app_mask = torch.empty((count, height, width), device=device, dtype=torch.uint8)
        self.raw_dis_mask = torch.empty((count, height, width), device=device, dtype=torch.uint8)
        self.raw_app_str = torch.empty((count, height, width), device=device, dtype=torch.float32)
        self.raw_dis_str = torch.empty((count, height, width), device=device, dtype=torch.float32)
        self.app_mask = torch.empty((count, height, width), device=device, dtype=torch.uint8)
        self.dis_mask = torch.empty((count, height, width), device=device, dtype=torch.uint8)
        self.app_str = torch.empty((count, height, width), device=device, dtype=torch.float32)
        self.dis_str = torch.empty((count, height, width), device=device, dtype=torch.float32)
        self.world_now = torch.empty((count, height, width, 3), device=device, dtype=torch.float32)
        self.world_old = torch.empty((count, height, width, 3), device=device, dtype=torch.float32)
        self.dis_uv = torch.empty((count, height, width, 3), device=device, dtype=torch.int32)
        self.dual = torch.zeros((count, 2, height, width), device=device, dtype=torch.float32)

    def run(
        self,
        *,
        depth: torch.Tensor,
        old_depth: torch.Tensor,
        pose: torch.Tensor,
        old_pose: torch.Tensor,
        intrinsic: torch.Tensor,
        old_intrinsic: torch.Tensor,
        active_u8: torch.Tensor,
        seen: torch.Tensor,
        history_frames: int,
        match_radius: int,
        edge_suppression: bool,
        min_cluster_pixels: int,
        max_depth: float,
        threshold: float,
        relative_threshold: float,
        sigma_multiplier: float,
        base_var: float,
        pose_rot_sigma: float,
    ):
        wp_device = wp.device_from_torch(depth.device)
        stream = wp.stream_from_torch(depth.device)
        dim = (self.count, self.height, self.width)
        edge_flag = 1 if edge_suppression else 0

        wp.launch(
            _edge_ref_kernel,
            dim=dim,
            inputs=[
                wp.from_torch(old_depth, dtype=wp.float32),
                match_radius,
                edge_flag,
                max_depth,
            ],
            outputs=[wp.from_torch(self.old_ref, dtype=wp.float32)],
            device=wp_device,
            stream=stream,
        )
        wp.launch(
            _edge_ref_kernel,
            dim=dim,
            inputs=[
                wp.from_torch(depth, dtype=wp.float32),
                match_radius,
                edge_flag,
                max_depth,
            ],
            outputs=[wp.from_torch(self.now_ref, dtype=wp.float32)],
            device=wp_device,
            stream=stream,
        )
        wp.launch(
            _depth_event_detect_kernel,
            dim=dim,
            inputs=[
                wp.from_torch(depth, dtype=wp.float32),
                wp.from_torch(old_depth, dtype=wp.float32),
                wp.from_torch(self.old_ref, dtype=wp.float32),
                wp.from_torch(self.now_ref, dtype=wp.float32),
                wp.from_torch(pose.contiguous(), dtype=wp.float32),
                wp.from_torch(old_pose.contiguous(), dtype=wp.float32),
                wp.from_torch(intrinsic.contiguous(), dtype=wp.float32),
                wp.from_torch(old_intrinsic.contiguous(), dtype=wp.float32),
                wp.from_torch(active_u8, dtype=wp.uint8),
                wp.from_torch(seen, dtype=wp.int64),
                history_frames,
                max_depth,
                threshold,
                relative_threshold,
                sigma_multiplier,
                base_var,
                pose_rot_sigma,
            ],
            outputs=[
                wp.from_torch(self.raw_app_mask, dtype=wp.uint8),
                wp.from_torch(self.raw_dis_mask, dtype=wp.uint8),
                wp.from_torch(self.raw_app_str, dtype=wp.float32),
                wp.from_torch(self.raw_dis_str, dtype=wp.float32),
                wp.from_torch(self.world_now, dtype=wp.float32),
                wp.from_torch(self.world_old, dtype=wp.float32),
                wp.from_torch(self.dis_uv, dtype=wp.int32),
            ],
            device=wp_device,
            stream=stream,
        )
        self.dual[:, 1].zero_()
        wp.launch(
            _filter_and_splat_kernel,
            dim=dim,
            inputs=[
                wp.from_torch(self.raw_app_mask, dtype=wp.uint8),
                wp.from_torch(self.raw_dis_mask, dtype=wp.uint8),
                wp.from_torch(self.raw_app_str, dtype=wp.float32),
                wp.from_torch(self.raw_dis_str, dtype=wp.float32),
                wp.from_torch(self.dis_uv, dtype=wp.int32),
                min_cluster_pixels,
            ],
            outputs=[
                wp.from_torch(self.app_mask, dtype=wp.uint8),
                wp.from_torch(self.dis_mask, dtype=wp.uint8),
                wp.from_torch(self.app_str, dtype=wp.float32),
                wp.from_torch(self.dis_str, dtype=wp.float32),
                wp.from_torch(self.dual, dtype=wp.float32),
            ],
            device=wp_device,
            stream=stream,
        )
        return (
            self.app_mask.bool(),
            self.dis_mask.bool(),
            self.world_now,
            self.world_old,
            self.app_str,
            self.dis_str,
            self.dual,
        )


@wp.func
def _read_clamped(depth: wp.array3d(dtype=wp.float32), b: int, v: int, u: int, height: int, width: int, max_depth: float) -> float:
    vc = wp.clamp(v, 0, height - 1)
    uc = wp.clamp(u, 0, width - 1)
    d = depth[b, vc, uc]
    return d if _valid_depth(d, max_depth) else max_depth


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

                        # Local gradient in past reference frame (inline over slot).
                        pu_l = wp.clamp(pu - 1, 0, width - 1)
                        pu_r = wp.clamp(pu + 1, 0, width - 1)
                        pv_u = wp.clamp(pv - 1, 0, height - 1)
                        pv_d = wp.clamp(pv + 1, 0, height - 1)
                        dl = depth_hist[b, slot, pv, pu_l]
                        dr = depth_hist[b, slot, pv, pu_r]
                        du = depth_hist[b, slot, pv_u, pu]
                        dd = depth_hist[b, slot, pv_d, pu]
                        dl_v = dl if _valid_depth(dl, max_depth) else max_depth
                        dr_v = dr if _valid_depth(dr, max_depth) else max_depth
                        du_v = du if _valid_depth(du, max_depth) else max_depth
                        dd_v = dd if _valid_depth(dd, max_depth) else max_depth
                        gx_o = 0.5 * (dr_v - dl_v)
                        gy_o = 0.5 * (dd_v - du_v)
                        g2_o = gx_o * gx_o + gy_o * gy_o

                        sig2_st_r = c0 + c2 * d_r_val * d_r_val
                        rot_term = proj_p[2] * rot_sigma
                        sig2_z_xi = pos_sigma * pos_sigma + rot_term * rot_term
                        j_px_r = fx_o * rot_sigma
                        j_px_p = fx_o * pos_sigma / wp.max(proj_p[2], 0.3)
                        sig2_edge = g2_o * (j_px_r * j_px_r + j_px_p * j_px_p + 0.25)
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
                            if valid_r and innov <= 8.0 * dt and g2_o < 1.0:
                                inv_ttc = (innov - gate) / (wp.max(d_n_val, 0.3) * dt)
                                loom_k = wp.clamp(inv_ttc / looming_max_rate, 0.0, 1.0)
                                if loom_k > looming:
                                    looming = loom_k

                # 2) Backward check: past pixel into current view -> wake_disappear.
                d_o_k = depth_hist[b, slot, v, u]
                if _valid_depth(d_o_k, max_depth):
                    w_ok = _backproject_pixel(d_o_k, float(u), float(v), fx_o, fy_o, cx_o, cy_o, p_o, q_o)
                    proj_c = _project_world(w_ok, p_now, q_now, fx_n, fy_n, cx_n, cy_n, height, width, max_depth)
                    if proj_c[3] > 0.5:
                        cu_gs = wp.clamp(int(wp.floor(proj_c[0] + 0.5)), 0, width - 1)
                        cv_gs = wp.clamp(int(wp.floor(proj_c[1] + 0.5)), 0, height - 1)
                        dc = depth_now[b, cv_gs, cu_gs]
                        dc_val = dc if _valid_depth(dc, max_depth) else max_depth
                        gx_c = 0.5 * (_read_clamped(depth_now, b, cv_gs, cu_gs + 1, height, width, max_depth) - _read_clamped(depth_now, b, cv_gs, cu_gs - 1, height, width, max_depth))
                        gy_c = 0.5 * (_read_clamped(depth_now, b, cv_gs + 1, cu_gs, height, width, max_depth) - _read_clamped(depth_now, b, cv_gs - 1, cu_gs, height, width, max_depth))
                        g2_c = gx_c * gx_c + gy_c * gy_c
                        sig2_st_o = c0 + c2 * d_o_k * d_o_k
                        sig2_st_c = c0 + c2 * dc_val * dc_val
                        rot_term_c = proj_c[2] * rot_sigma
                        sig2_z_xi_c = pos_sigma * pos_sigma + rot_term_c * rot_term_c
                        j_px_r_c = fx_n * rot_sigma
                        j_px_p_c = fx_n * pos_sigma / wp.max(proj_c[2], 0.3)
                        sig2_edge_c = g2_c * (j_px_r_c * j_px_r_c + j_px_p_c * j_px_p_c + 0.25)
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


class WarpKinematicFieldWorkspace:
    """Reusable GPU buffers for zero-allocation EG-TS + Radial Looming computation."""

    def __init__(self, count: int, height: int, width: int, device: torch.device):
        wp.init()
        self.count, self.height, self.width = count, height, width
        self.device = device
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
        self.kinematic[:, 1].zero_()
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
                wp.from_torch(self.kinematic, dtype=wp.float32),
                wp.from_torch(self.world_now, dtype=wp.float32),
                wp.from_torch(self.world_old, dtype=wp.float32),
                wp.from_torch(self.app_mask, dtype=wp.uint8),
                wp.from_torch(self.dis_mask, dtype=wp.uint8),
            ],
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
