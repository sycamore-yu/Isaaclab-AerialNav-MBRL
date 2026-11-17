"""Pose-compensated depth events: sparse geometric appearances and disappearances.

Adapted from the isaaclab-IPC depth-occlusion-events experiment (8d74f4d).

Unlike a DVS emulator these events report depth/occlusion changes, not intensity
changes. Coordinates use the Isaac Lab robotics camera frame: +X forward,
+Y left, +Z up; depth is distance along +X (distance_to_image_plane).
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def _rotate(q, v):
    """Rotate 3-vectors with scalar-last (xyzw) quaternions."""
    xyz = q[..., :3]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + q[..., 3:] * t + torch.cross(xyz, t, dim=-1)


def _backproject(depth, intrinsic, pose, pixels):
    """Depth [B,H,W] -> world points [B,H,W,3]. Invalid rays are masked by caller."""
    fx = intrinsic[:, 0, 0, None, None]
    fy = intrinsic[:, 1, 1, None, None]
    cx = intrinsic[:, 0, 2, None, None]
    cy = intrinsic[:, 1, 2, None, None]
    u, v = pixels
    points = torch.stack((depth, -(u - cx) * depth / fx, -(v - cy) * depth / fy), dim=-1)
    return _rotate(pose[:, None, None, 3:], points) + pose[:, None, None, :3]


def _project(world, pose, intrinsic, height, width, max_depth):
    """World points -> image center indices, predicted camera-plane depth, overlap mask."""
    q = pose[:, None, None, 3:]
    inv_q = torch.cat((-q[..., :3], q[..., 3:]), dim=-1)
    points = _rotate(inv_q, world - pose[:, None, None, :3])
    x = points[..., 0]
    safe_x = x.clamp_min(1e-6)
    u = intrinsic[:, 0, 2, None, None] - intrinsic[:, 0, 0, None, None] * points[..., 1] / safe_x - 0.5
    v = intrinsic[:, 1, 2, None, None] - intrinsic[:, 1, 1, None, None] * points[..., 2] / safe_x - 0.5
    visible = (x > 0) & (x < max_depth) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    # grid_sample align_corners=False maps integer pixel centers to (index+.5)/extent.
    grid = torch.stack((2 * (u + 0.5) / width - 1, 2 * (v + 0.5) / height - 1), dim=-1)
    return grid, x, visible


@dataclass
class DepthEventBatch:
    """GPU masks and positions; sparse conversion is optional and synchronizes on CUDA.

    appeared: current-image pixels showing newly closer surfaces.
    disappeared: *previous-image* pixels whose old surface is now farther/absent.
    Positive polarity (+1) means appearance, negative (-1) disappearance.
    """

    appeared: torch.Tensor
    disappeared: torch.Tensor
    appeared_world: torch.Tensor
    disappeared_world: torch.Tensor
    timestamp: torch.Tensor
    appearance_strength: torch.Tensor
    disappearance_strength: torch.Tensor
    _precomputed_dual: torch.Tensor | None = None
    _precomputed_kinematic: torch.Tensor | None = None
    _pose_ref: torch.Tensor | None = None
    _intrinsic_ref: torch.Tensor | None = None

    def _matches_capture_view(self, current_pose, intrinsic) -> bool:
        if self._pose_ref is None or self._intrinsic_ref is None or current_pose.device != self._pose_ref.device:
            return False
        same_pose = (
            current_pose is self._pose_ref
            or current_pose.data_ptr() == self._pose_ref.data_ptr()
            or torch.equal(current_pose, self._pose_ref)
        )
        same_k = (
            intrinsic is self._intrinsic_ref
            or intrinsic.data_ptr() == self._intrinsic_ref.data_ptr()
            or torch.equal(intrinsic, self._intrinsic_ref)
        )
        return same_pose and same_k

    def kinematic_image(self, current_pose, intrinsic, max_depth):
        """3-channel kinematic field [B,3,H,W]: (wake_appear, wake_disappear, radial_looming)."""
        if self._precomputed_kinematic is not None and self._matches_capture_view(current_pose, intrinsic):
            return self._precomputed_kinematic
        dual = self.dual_image(current_pose, intrinsic, max_depth)
        return torch.cat((dual, torch.zeros_like(dual[:, :1])), dim=1)

    def dual_image(self, current_pose, intrinsic, max_depth):
        """Dual [0,1] channels in the CURRENT camera frame [B,2,H,W]: (appeared, disappeared)."""
        if self._precomputed_dual is not None and self._matches_capture_view(current_pose, intrinsic):
            return self._precomputed_dual
        count, height, width = self.appeared.shape
        grid, _, within = _project(self.disappeared_world, current_pose, intrinsic, height, width, max_depth)
        u = ((grid[..., 0] + 1) * width / 2 - 0.5).round().long().clamp(0, width - 1)
        v = ((grid[..., 1] + 1) * height / 2 - 0.5).round().long().clamp(0, height - 1)
        negative = self.appeared_world.new_zeros((count, height * width))
        negative.scatter_reduce_(
            1,
            (v * width + u).reshape(count, -1),
            torch.where(self.disappeared & within, self.disappearance_strength, 0).reshape(count, -1),
            reduce="amax",
            include_self=True,
        )
        positive = torch.where(self.appeared, self.appearance_strength, 0.0)
        return torch.stack((positive, negative.reshape(count, height, width)), dim=1)

    def signed_image(self, current_pose, intrinsic, max_depth):
        """Signed [-1,1] confidence in the CURRENT camera frame [B,1,H,W]."""
        dual = self.dual_image(current_pose, intrinsic, max_depth)
        # A nearer appearing surface wins if two events overlap in a pixel.
        output = torch.where(self.appeared, dual[:, 0], -dual[:, 1])
        return output[:, None]

    def sparse(self):
        """[N,8]: env_id, u, v, time_s, polarity, world_x, world_y, world_z."""
        pieces = []
        for polarity, mask, points in (
            (1, self.appeared, self.appeared_world),
            (-1, self.disappeared, self.disappeared_world),
        ):
            ids = mask.nonzero(as_tuple=False)
            if ids.numel():
                e, v, u = ids.unbind(-1)
                pieces.append(
                    torch.stack(
                        (
                            e.to(points.dtype), u.to(points.dtype), v.to(points.dtype),
                            self.timestamp[e].to(points.dtype), torch.full_like(u, polarity, dtype=points.dtype),
                            *points[e, v, u].unbind(-1),
                        ),
                        dim=-1,
                    )
                )
        return torch.cat(pieces) if pieces else self.appeared_world.new_empty((0, 8))


class DepthEventDetector:
    """Depth-consistency events in overlapping views, with known simulation poses.

    The test is conservative: the candidate must be closer than *all* pixels in
    a local window of the reference image. This rejects reprojected silhouette
    jitter; it also sets a minimum detectable object footprint.
    """

    def __init__(
        self, height, width, *, max_depth=20.0, threshold=0.25, relative_threshold=0.01,
        match_radius=1, history_frames=1, depth_sigma=0.0, pose_position_sigma=0.0,
        pose_rotation_sigma=0.0, sigma_multiplier=0.0, edge_suppression=False,
        edge_suppression_scale=0.0, min_cluster_pixels=1, kinematic_field=False,
        eg_ts_decay=0.7, stereo_c2=4e-6, step_dt=0.05, looming_max_rate=2.0,
    ):
        if height <= 0 or width <= 0 or max_depth <= 0 or threshold < 0 or relative_threshold < 0:
            raise ValueError("Invalid depth event configuration")
        if not isinstance(match_radius, int) or match_radius < 0:
            raise ValueError("match_radius must be a nonnegative integer")
        if not isinstance(history_frames, int) or history_frames < 1:
            raise ValueError("history_frames must be a positive integer")
        if not isinstance(min_cluster_pixels, int) or min_cluster_pixels < 1:
            raise ValueError("min_cluster_pixels must be a positive integer")
        if min(depth_sigma, pose_position_sigma, pose_rotation_sigma, sigma_multiplier, edge_suppression_scale) < 0:
            raise ValueError("Uncertainty parameters must be nonnegative")
        if not 0.0 < eg_ts_decay <= 1.0 or stereo_c2 < 0 or step_dt <= 0 or looming_max_rate <= 0:
            raise ValueError("Invalid kinematic field parameters")
        self.height, self.width = height, width
        self.max_depth = float(max_depth)
        self.threshold = float(threshold)
        self.relative_threshold = float(relative_threshold)
        self.match_radius = match_radius
        self.history_frames = history_frames
        self.depth_sigma = depth_sigma
        self.pose_position_sigma = pose_position_sigma
        self.pose_rotation_sigma = pose_rotation_sigma
        self.sigma_multiplier = sigma_multiplier
        self.edge_suppression = bool(edge_suppression or edge_suppression_scale > 0)
        self.min_cluster_pixels = 3 if (self.edge_suppression and min_cluster_pixels == 1) else min_cluster_pixels
        self.kinematic_field = bool(kinematic_field)
        self.eg_ts_decay = float(eg_ts_decay)
        self.stereo_c2 = float(stereo_c2)
        self.step_dt = float(step_dt)
        self.looming_max_rate = float(looming_max_rate)
        self._depth = self._pose = self._intrinsic = self._seen = self._next = None
        self._pixels = None
        self._warp_workspace = None
        self._kinematic_workspace = None

    def reset(self, env_ids=None):
        if self._seen is not None:
            if env_ids is None:
                self._seen.zero_()
            else:
                self._seen[env_ids] = 0

    def _margin(self, depth):
        # First-order uncertainty of differencing two depth measurements after
        # pose compensation. Legacy settings (all sigmas zero) are unchanged.
        sigma = torch.sqrt(
            2 * self.depth_sigma**2 + self.pose_position_sigma**2
            + (depth * self.pose_rotation_sigma)**2
        )
        return self.threshold + self.sigma_multiplier * sigma + self.relative_threshold * depth

    def _near_min(self, depth, radius=None):
        # Infinity means no surface was hit within max_depth (free-space ray).
        valid = torch.isfinite(depth) & (depth > 0) & (depth < self.max_depth)
        values = torch.where(valid, depth, torch.full_like(depth, self.max_depth))
        radius = self.match_radius if radius is None else radius
        if radius:
            return -F.max_pool2d(-values[:, None], kernel_size=2 * radius + 1, stride=1, padding=radius)[:, 0]
        return values

    def _edge_aware_reference(self, depth):
        """Suppress 1-2 px reprojection jitter across sharp static depth silhouettes."""
        near_1 = self._near_min(depth, radius=self.match_radius)
        if not self.edge_suppression:
            return near_1
        valid = torch.isfinite(depth) & (depth > 0) & (depth < self.max_depth)
        values = torch.where(valid, depth, torch.full_like(depth, self.max_depth))
        far_1 = F.max_pool2d(values[:, None], kernel_size=3, stride=1, padding=1)[:, 0]
        span = (far_1 - near_1).clamp_min(0.0)
        wide_radius = max(self.match_radius + 1, 2)
        near_wide = self._near_min(depth, radius=wide_radius)
        return torch.where(span > 0.75, near_wide, near_1)

    def _filter_isolated(self, mask):
        if self.min_cluster_pixels <= 1:
            return mask
        count = F.avg_pool2d(mask[:, None].float(), kernel_size=3, stride=1, padding=1)[:, 0] * 9.0
        return mask & (count >= self.min_cluster_pixels - 0.1)

    def _depth_grad_sq(self, depth):
        valid = torch.isfinite(depth) & (depth > 0) & (depth < self.max_depth)
        val = torch.where(valid, depth, torch.full_like(depth, self.max_depth))
        padded = F.pad(val[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
        gx = 0.5 * (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2])
        gy = 0.5 * (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1])
        return val, gx * gx + gy * gy

    def _update_kinematic_cpu(self, depth, pose, intrinsic, time, active, rows):
        count = depth.shape[0]
        if self._pixels is None:
            v, u = torch.meshgrid(
                torch.arange(self.height, device=depth.device, dtype=depth.dtype) + 0.5,
                torch.arange(self.width, device=depth.device, dtype=depth.dtype) + 0.5,
                indexing="ij",
            )
            self._pixels = u, v
        valid_now = torch.isfinite(depth) & (depth > 0) & (depth < self.max_depth)
        d_now_val, g2_now = self._depth_grad_sq(depth)
        world_now = _backproject(torch.where(valid_now, depth, 0), intrinsic, pose, self._pixels)

        old_depth0 = self._depth[rows, self._next]
        old_pose0 = self._pose[rows, self._next]
        old_int0 = self._intrinsic[rows, self._next]
        valid_old0 = torch.isfinite(old_depth0) & (old_depth0 > 0) & (old_depth0 < self.max_depth)
        world_old0 = _backproject(torch.where(valid_old0, old_depth0, 0), old_int0, old_pose0, self._pixels)

        kinematic = depth.new_zeros((count, 3, self.height, self.width))
        app_any = torch.zeros((count, self.height, self.width), device=depth.device, dtype=torch.bool)
        dis_any = torch.zeros_like(app_any)
        c0 = self.depth_sigma**2
        sig2_st_n = c0 + self.stereo_c2 * d_now_val * d_now_val
        fx_n = intrinsic[:, 0, 0, None, None]

        gamma = 1.0
        for k in range(1, self.history_frames + 1):
            has_k = active & (self._seen >= k)
            if bool(has_k.any()):
                slot = (self._next + self.history_frames - k) % self.history_frames
                old_d = self._depth[rows, slot]
                old_p = self._pose[rows, slot]
                old_k = self._intrinsic[rows, slot]
                dt = k * self.step_dt
                valid_r = torch.isfinite(old_d) & (old_d > 0) & (old_d < self.max_depth)
                d_r_raw, g2_o = self._depth_grad_sq(old_d)

                puv, d_proj, in_prev = _project(world_now, old_p, old_k, self.height, self.width, self.max_depth)
                d_r_gs = F.grid_sample(d_r_raw[:, None], puv, mode="nearest", align_corners=False)[:, 0]
                g2_o_gs = F.grid_sample(g2_o[:, None], puv, mode="nearest", align_corners=False)[:, 0]
                vr_gs = F.grid_sample(valid_r[:, None].to(depth.dtype), puv, mode="nearest", align_corners=False)[:, 0] > 0.5
                fx_o = old_k[:, 0, 0, None, None]
                sig2_st_r = c0 + self.stereo_c2 * d_r_gs * d_r_gs
                sig2_z = self.pose_position_sigma**2 + (d_proj * self.pose_rotation_sigma) ** 2
                j2 = (fx_o * self.pose_rotation_sigma) ** 2 + (fx_o * self.pose_position_sigma / d_proj.clamp_min(0.3)) ** 2 + 0.25
                sig_tot = torch.sqrt(sig2_st_n + sig2_st_r + sig2_z + g2_o_gs * j2)
                gate = self.threshold + self.sigma_multiplier * sig_tot + self.relative_threshold * d_proj
                innov = d_r_gs - d_proj
                hit = has_k[:, None, None] & valid_now & in_prev & (innov > gate)
                app_any |= hit
                s_app = ((innov - gate) / sig_tot.mul(2.0).clamp_min(0.25)).clamp(0.0, 1.0)
                kinematic[:, 0] = torch.maximum(kinematic[:, 0], torch.where(hit, gamma * s_app, 0.0))
                loom_hit = hit & vr_gs & (innov <= 8.0 * dt) & (g2_o_gs < 1.0)
                inv_ttc = (innov - gate) / (d_now_val.clamp_min(0.3) * dt)
                kinematic[:, 2] = torch.maximum(
                    kinematic[:, 2], torch.where(loom_hit, (inv_ttc / self.looming_max_rate).clamp(0.0, 1.0), 0.0)
                )

                w_ok = _backproject(torch.where(valid_r, old_d, 0), old_k, old_p, self._pixels)
                cuv, d_c_proj, in_cur = _project(w_ok, pose, intrinsic, self.height, self.width, self.max_depth)
                dc_gs = F.grid_sample(d_now_val[:, None], cuv, mode="nearest", align_corners=False)[:, 0]
                g2_c_gs = F.grid_sample(g2_now[:, None], cuv, mode="nearest", align_corners=False)[:, 0]
                sig2_st_o = c0 + self.stereo_c2 * torch.where(valid_r, old_d, 0.0) ** 2
                sig2_st_c = c0 + self.stereo_c2 * dc_gs * dc_gs
                sig2_zc = self.pose_position_sigma**2 + (d_c_proj * self.pose_rotation_sigma) ** 2
                j2c = (fx_n * self.pose_rotation_sigma) ** 2 + (fx_n * self.pose_position_sigma / d_c_proj.clamp_min(0.3)) ** 2 + 0.25
                sig_tot_c = torch.sqrt(sig2_st_o + sig2_st_c + sig2_zc + g2_c_gs * j2c)
                gate_c = self.threshold + self.sigma_multiplier * sig_tot_c + self.relative_threshold * d_c_proj
                innov_c = dc_gs - d_c_proj
                hit_d = has_k[:, None, None] & valid_r & in_cur & (innov_c > gate_c)
                dis_any |= hit_d
                s_dis = gamma * ((innov_c - gate_c) / sig_tot_c.mul(2.0).clamp_min(0.25)).clamp(0.0, 1.0)
                cu = ((cuv[..., 0] + 1) * self.width / 2).floor().long().clamp(0, self.width - 1)
                cv = ((cuv[..., 1] + 1) * self.height / 2).floor().long().clamp(0, self.height - 1)
                neg = depth.new_zeros((count, self.height * self.width))
                neg.scatter_reduce_(
                    1, (cv * self.width + cu).reshape(count, -1),
                    torch.where(hit_d, s_dis, 0.0).reshape(count, -1), reduce="amax", include_self=True,
                )
                kinematic[:, 1] = torch.maximum(kinematic[:, 1], neg.reshape(count, self.height, self.width))
            gamma *= self.eg_ts_decay

        return DepthEventBatch(
            app_any, dis_any, world_now, world_old0, time.clone(),
            kinematic[:, 0], kinematic[:, 1],
            _precomputed_dual=kinematic[:, :2], _precomputed_kinematic=kinematic,
            _pose_ref=pose, _intrinsic_ref=intrinsic,
        )

    @torch.no_grad()
    def update(self, depth, pose, intrinsic, timestamp, *, active=None):
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 3 or tuple(depth.shape[-2:]) != (self.height, self.width):
            raise ValueError(f"depth must have shape [B,{self.height},{self.width}]")
        count = depth.shape[0]
        if pose.shape != (count, 7) or intrinsic.shape != (count, 3, 3):
            raise ValueError("pose must be [B,7] (xyz,xyzw), intrinsic [B,3,3]")
        if active is None:
            active = torch.ones(count, device=depth.device, dtype=torch.bool)
        else:
            active = torch.as_tensor(active, device=depth.device, dtype=torch.bool)
        if active.shape != (count,):
            raise ValueError("active must be [B]")
        time = torch.as_tensor(timestamp, device=depth.device, dtype=depth.dtype)
        if time.ndim == 0:
            time = time.expand(count)
        if time.shape != (count,):
            raise ValueError("timestamp must be scalar or [B]")
        depth = depth.detach().contiguous()
        pose = pose.detach()
        intrinsic = intrinsic.detach()
        if self._seen is None:
            self._seen = torch.zeros(count, dtype=torch.long, device=depth.device)
            self._next = torch.zeros_like(self._seen)
            self._depth = depth.new_zeros((count, self.history_frames, self.height, self.width))
            self._pose = pose.new_zeros((count, self.history_frames, 7))
            self._intrinsic = intrinsic.new_zeros((count, self.history_frames, 3, 3))
        elif self._depth.shape != (count, self.history_frames, self.height, self.width) or self._depth.device != depth.device:
            raise ValueError("Batch shape/device changed; create a new detector")

        rows = torch.arange(count, device=depth.device)
        old_depth = self._depth[rows, self._next]
        old_pose = self._pose[rows, self._next]
        old_intrinsic = self._intrinsic[rows, self._next]

        if self.kinematic_field:
            if depth.is_cuda and depth.dtype == torch.float32:
                if (
                    self._kinematic_workspace is None
                    or self._kinematic_workspace.count != count
                    or self._kinematic_workspace.device != depth.device
                ):
                    from .geometric_depth_events_warp import WarpKinematicFieldWorkspace

                    self._kinematic_workspace = WarpKinematicFieldWorkspace(count, self.height, self.width, depth.device)
                app_mask, dis_mask, world_now, world_old, kinematic = self._kinematic_workspace.run(
                    depth=depth,
                    depth_hist=self._depth,
                    pose=pose,
                    pose_hist=self._pose,
                    intrinsic=intrinsic,
                    intrinsic_hist=self._intrinsic,
                    next_idx=self._next,
                    seen=self._seen,
                    active_u8=active.to(torch.uint8),
                    history_frames=self.history_frames,
                    max_depth=self.max_depth,
                    threshold=self.threshold,
                    relative_threshold=self.relative_threshold,
                    sigma_multiplier=self.sigma_multiplier,
                    c0=self.depth_sigma**2,
                    c2=self.stereo_c2,
                    pos_sigma=self.pose_position_sigma,
                    rot_sigma=self.pose_rotation_sigma,
                    decay=self.eg_ts_decay,
                    step_dt=self.step_dt,
                    looming_max_rate=self.looming_max_rate,
                )
                result = DepthEventBatch(
                    app_mask, dis_mask, world_now, world_old, time.clone(),
                    kinematic[:, 0], kinematic[:, 1],
                    _precomputed_dual=kinematic[:, :2], _precomputed_kinematic=kinematic,
                    _pose_ref=pose, _intrinsic_ref=intrinsic,
                )
            else:
                result = self._update_kinematic_cpu(depth, pose, intrinsic, time, active, rows)
            self._depth[rows, self._next] = torch.where(active[:, None, None], depth, old_depth)
            self._pose[rows, self._next] = torch.where(active[:, None], pose, old_pose)
            self._intrinsic[rows, self._next] = torch.where(active[:, None, None], intrinsic, old_intrinsic)
            self._next = (self._next + active.long()) % self.history_frames
            self._seen = (self._seen + active.long()).clamp_max(self.history_frames)
            return result

        if depth.is_cuda and depth.dtype == torch.float32:
            if (
                self._warp_workspace is None
                or self._warp_workspace.count != count
                or self._warp_workspace.device != depth.device
            ):
                from .geometric_depth_events_warp import WarpDepthEventWorkspace

                self._warp_workspace = WarpDepthEventWorkspace(count, self.height, self.width, depth.device)
            base_var = 2.0 * self.depth_sigma**2 + self.pose_position_sigma**2
            app_mask, dis_mask, world_now, world_old, app_str, dis_str, dual = self._warp_workspace.run(
                depth=depth,
                old_depth=old_depth,
                pose=pose,
                old_pose=old_pose,
                intrinsic=intrinsic,
                old_intrinsic=old_intrinsic,
                active_u8=active.to(torch.uint8),
                seen=self._seen,
                history_frames=self.history_frames,
                match_radius=self.match_radius,
                edge_suppression=self.edge_suppression,
                min_cluster_pixels=self.min_cluster_pixels,
                max_depth=self.max_depth,
                threshold=self.threshold,
                relative_threshold=self.relative_threshold,
                sigma_multiplier=self.sigma_multiplier,
                base_var=base_var,
                pose_rot_sigma=self.pose_rotation_sigma,
            )
            result = DepthEventBatch(
                app_mask, dis_mask, world_now, world_old, time.clone(),
                app_str, dis_str, _precomputed_dual=dual, _pose_ref=pose, _intrinsic_ref=intrinsic,
            )
            self._depth[rows, self._next] = torch.where(active[:, None, None], depth, old_depth)
            self._pose[rows, self._next] = torch.where(active[:, None], pose, old_pose)
            self._intrinsic[rows, self._next] = torch.where(active[:, None, None], intrinsic, old_intrinsic)
            self._next = (self._next + active.long()) % self.history_frames
            self._seen = (self._seen + active.long()).clamp_max(self.history_frames)
            return result

        if self._pixels is None:
            v, u = torch.meshgrid(
                torch.arange(self.height, device=depth.device, dtype=depth.dtype) + 0.5,
                torch.arange(self.width, device=depth.device, dtype=depth.dtype) + 0.5,
                indexing="ij",
            )
            self._pixels = u, v

        valid_now = torch.isfinite(depth) & (depth > 0) & (depth < self.max_depth)
        valid_old = torch.isfinite(old_depth) & (old_depth > 0) & (old_depth < self.max_depth)
        world_now = _backproject(torch.where(valid_now, depth, 0), intrinsic, pose, self._pixels)
        world_old = _backproject(torch.where(valid_old, old_depth, 0), old_intrinsic, old_pose, self._pixels)

        # Newly occupied rays: current points must have been visible and free in
        # the previous view. A previous nearer hit means prior occlusion, not motion.
        previous_uv, current_in_previous, in_previous = _project(
            world_now, old_pose, old_intrinsic, self.height, self.width, self.max_depth
        )
        old_ref = self._edge_aware_reference(old_depth)
        old_near = F.grid_sample(old_ref[:, None], previous_uv, mode="nearest", align_corners=False)[:, 0]
        appeared = self._filter_isolated(
            valid_now
            & in_previous
            & (current_in_previous + self._margin(current_in_previous) < old_near)
        )

        # Vacated rays: only mark an old surface if the *current* observation
        # sees farther. If a new surface occludes it, do not claim it vanished.
        current_uv, previous_in_current, in_current = _project(
            world_old, pose, intrinsic, self.height, self.width, self.max_depth
        )
        now_ref = self._edge_aware_reference(depth)
        now_near = F.grid_sample(now_ref[:, None], current_uv, mode="nearest", align_corners=False)[:, 0]
        disappeared = self._filter_isolated(
            valid_old
            & in_current
            & (previous_in_current + self._margin(previous_in_current) < now_near)
        )

        eligible = (active & (self._seen >= self.history_frames))[:, None, None]
        appeared &= eligible
        disappeared &= eligible
        result = DepthEventBatch(
            appeared, disappeared, world_now, world_old, time.clone(),
            torch.where(appeared, ((old_near - current_in_previous) / 0.5).clamp(0, 1), 0),
            torch.where(disappeared, ((now_near - previous_in_current) / 0.5).clamp(0, 1), 0),
        )
        self._depth[rows, self._next] = torch.where(active[:, None, None], depth, old_depth)
        self._pose[rows, self._next] = torch.where(active[:, None], pose, old_pose)
        self._intrinsic[rows, self._next] = torch.where(active[:, None, None], intrinsic, old_intrinsic)
        self._next = (self._next + active.long()) % self.history_frames
        self._seen = (self._seen + active.long()).clamp_max(self.history_frames)
        return result
