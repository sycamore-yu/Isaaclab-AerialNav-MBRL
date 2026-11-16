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

    def dual_image(self, current_pose, intrinsic, max_depth):
        """Dual [0,1] channels in the CURRENT camera frame [B,2,H,W]: (appeared, disappeared)."""
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
        edge_suppression_scale=0.0, min_cluster_pixels=1,
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
        self._depth = self._pose = self._intrinsic = self._seen = self._next = None
        self._pixels = None

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
