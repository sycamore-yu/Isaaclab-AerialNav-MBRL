"""Full-resolution Isaac Lab depth rays with bounded image working memory.

Uses each backend's official multi-mesh sensor and unchanged local triangle geometry.
One top-level BVH conservatively selects candidate meshes, then the original
local-frame mesh query computes depth. Every configured pixel is traced before
the DiffNav depth transform; only image working buffers are batched.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import warp as wp
from isaaclab.sensors.ray_caster import kernels as camera_kernels
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.utils import math as math_utils
from isaaclab.utils.warp import ProxyArray

from aerialnav.warp_utils import torch_warp_stream

from .depth_processing import depth_features
from .lidar_kernels import rotate_directions_with_yaw


def _slice(array: wp.array, begin: int, end: int) -> wp.array:
    return array[begin:end]


@wp.kernel(enable_backward=False)
def _update_mesh_bounds(
    local_lower: wp.array(dtype=wp.vec3f),
    local_upper: wp.array(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    orientations: wp.array2d(dtype=wp.quatf),
    world_lower: wp.array(dtype=wp.vec3f),
    world_upper: wp.array(dtype=wp.vec3f),
):
    mesh = wp.tid()
    pose = wp.transform(positions[0, mesh], orientations[0, mesh])
    lower = wp.vec3f(1.0e30)
    upper = wp.vec3f(-1.0e30)
    for corner in range(8):
        point = wp.vec3f(
            wp.where((corner & 1) == 0, local_lower[mesh][0], local_upper[mesh][0]),
            wp.where((corner & 2) == 0, local_lower[mesh][1], local_upper[mesh][1]),
            wp.where((corner & 4) == 0, local_lower[mesh][2], local_upper[mesh][2]),
        )
        point = wp.transform_point(pose, point)
        lower = wp.vec3f(wp.min(lower[0], point[0]), wp.min(lower[1], point[1]), wp.min(lower[2], point[2]))
        upper = wp.vec3f(wp.max(upper[0], point[0]), wp.max(upper[1], point[1]), wp.max(upper[2], point[2]))
    magnitude = wp.max(
        wp.max(wp.abs(lower[0]), wp.abs(upper[0])),
        wp.max(wp.max(wp.abs(lower[1]), wp.abs(upper[1])), wp.max(wp.abs(lower[2]), wp.abs(upper[2]))),
    )
    # Broad-phase only: outward padding covers float32 transform/slab rounding.
    # It never changes a triangle or the final mesh-query hit distance.
    padding = wp.vec3f(0.001 + 4.0e-6 * magnitude)
    world_lower[mesh] = lower - padding
    world_upper[mesh] = upper + padding


@wp.kernel(enable_backward=False)
def _raycast_shared_meshes(
    bvh: wp.uint64,
    env_mask: wp.array(dtype=wp.bool),
    mesh_ids: wp.array2d(dtype=wp.uint64),
    positions: wp.array2d(dtype=wp.vec3f),
    orientations: wp.array2d(dtype=wp.quatf),
    ray_starts: wp.array2d(dtype=wp.vec3f),
    ray_directions: wp.array2d(dtype=wp.vec3f),
    max_dist: wp.float32,
    ray_hits: wp.array2d(dtype=wp.vec3f),
    ray_distance: wp.array2d(dtype=wp.float32),
):
    env, ray = wp.tid()
    if not env_mask[env]:
        return
    query = wp.bvh_query_ray(bvh, ray_starts[env, ray], ray_directions[env, ray])
    column = int(0)  # noqa: UP018 -- Warp mutable variables in a dynamic loop.
    closest = float(1.0e30)  # noqa: UP018 -- Warp mutable variables in a dynamic loop.
    closest_point = wp.vec3f()
    found = bool(False)  # noqa: UP018 -- Warp mutable variables in a dynamic loop.
    while wp.bvh_query_next(query, column):
        # Preserve Isaac Lab raycast_dynamic_meshes_kernel's local-frame
        # arithmetic, including its inverse-transform order and max distance.
        pose = wp.transform(positions[0, column], orientations[0, column])
        inverse = wp.transform_inverse(pose)
        direction = wp.transform_vector(inverse, ray_directions[env, ray])
        start = wp.transform_point(inverse, ray_starts[env, ray])
        hit = wp.mesh_query_ray(mesh_ids[0, column], start, direction, max_dist)
        if hit.result and hit.t < closest:
            closest = hit.t
            closest_point = wp.transform_point(pose, start + hit.t * direction)
            found = True
    if found:
        ray_hits[env, ray] = closest_point
        ray_distance[env, ray] = closest
    else:
        # Match the native fill kernel without a separate launch or an
        # initial write for hits. Inactive environments returned above.
        ray_hits[env, ray] = wp.vec3f(wp.inf, wp.inf, wp.inf)
        ray_distance[env, ray] = wp.inf


class _BatchedDepthCameraMixin:
    """An official sensor extension retaining pooled features for all drones.

    ``image_shape`` and intrinsic matrices describe the configured raw sensor.
    ``data.output['policy_depth']`` contains its transformed 12 x 16 result.
    ``last_depth_batch`` exposes the most recently computed raw depth batch
    for numerical checks without retaining all full-resolution images.
    """

    def _initialize_warp_meshes(self):
        """Build a shared-world broad phase without modifying native meshes."""
        self._clear_shared_mesh_tracking()
        super()._initialize_warp_meshes()
        stream = torch_warp_stream(self._device)
        with wp.ScopedStream(stream) if stream is not None else nullcontext():
            # Preserve the official per-target initialization before validating
            # and replacing only the subsequent transform-refresh path.
            super()._update_mesh_transforms()
            mesh_ids = self._mesh_ids_wp.numpy()
            if not (mesh_ids == mesh_ids[:1]).all():
                raise ValueError("The batched navigation camera requires one shared obstacle world.")
            for poses in (self._mesh_positions_w, self._mesh_orientations_w):
                values = poses.numpy()
                if not (values == values[:1]).all():
                    raise ValueError("Shared obstacle poses must match across camera environments.")
            self._initialize_shared_mesh_tracking()
            meshes_by_id = {int(mesh.id): mesh for mesh in self.meshes.values()}
            lower, upper = [], []
            vertex_count, triangle_count = 0, 0
            for mesh_id in mesh_ids[0]:
                mesh = meshes_by_id[int(mesh_id)]
                points = wp.to_torch(mesh.points)
                lower.append(points.amin(dim=0))
                upper.append(points.amax(dim=0))
                vertex_count += len(points)
                triangle_count += mesh.indices.size // 3
            self._local_lower_torch = torch.stack(lower).contiguous()
            self._local_upper_torch = torch.stack(upper).contiguous()
            self._local_lower = wp.from_torch(self._local_lower_torch, dtype=wp.vec3f)
            self._local_upper = wp.from_torch(self._local_upper_torch, dtype=wp.vec3f)
            self._world_lower = wp.empty(len(lower), dtype=wp.vec3f, device=self._device)
            self._world_upper = wp.empty(len(upper), dtype=wp.vec3f, device=self._device)
            self._update_world_bounds()
            self._world_bvh = wp.Bvh(self._world_lower, self._world_upper)
            self.world_mesh_info = {
                "acceleration_structure": "native_local_meshes_with_top_level_aabb_bvh",
                "source_meshes": len(mesh_ids[0]),
                "vertices": vertex_count,
                "triangles": triangle_count,
                "dynamic_meshes": sum(
                    self._num_meshes_per_env[target.prim_expr]
                    for target in self._raycast_targets_cfg
                    if target.track_mesh_transforms
                ),
                "broad_phase_padding_min_m": 0.001,
                "broad_phase_padding_relative": 4.0e-6,
                "mesh_transform_tracking": self.mesh_transform_info,
            }

    def _clear_shared_mesh_tracking(self):
        self._shared_mesh_tracking_ready = False
        self._shared_mesh_view = None
        self._shared_mesh_rows = None
        self._shared_mesh_count = 0
        self._shared_mesh_offset = 0
        self._shared_mesh_transform_source = None
        self._shared_mesh_transform_reordered = None

    def _invalidate_initialize_callback(self, event):
        self._clear_shared_mesh_tracking()
        super()._invalidate_initialize_callback(event)

    def _update_world_bounds(self):
        wp.launch(
            _update_mesh_bounds,
            dim=self._world_lower.size,
            inputs=[self._local_lower, self._local_upper, self._mesh_positions_w, self._mesh_orientations_w],
            outputs=[self._world_lower, self._world_upper],
            device=self._device,
        )

    def _initialize_rays_impl(self):
        if self.cfg.data_types != ["distance_to_image_plane"] or self.cfg.update_mesh_ids:
            raise ValueError("The batched policy camera requires only image-plane depth, without mesh IDs.")
        self.num_rays = self.image_shape[0] * self.image_shape[1]
        self._batch_size = min(self._num_envs, self.cfg.batch_size)
        self._data.create_buffers(self._num_envs, self._device)
        self._data.intrinsic_matrices.torch[:, 2, 2] = 1.0
        self._data.image_shape = self.image_shape
        self._data._output = {
            "policy_depth": ProxyArray(wp.zeros((self._num_envs, 12, 16, 1), dtype=wp.float32, device=self._device)),
            "distance_to_image_plane": ProxyArray(
                wp.zeros((self._num_envs, *self.image_shape, 1), dtype=wp.float32, device=self._device)
            ),
        }
        self._data.info = {"policy_depth": {"source_image_shape": self.image_shape}}
        self._compute_intrinsic_matrices()
        self._ray_inverse_intrinsics = torch.inverse(self._data.intrinsic_matrices.torch)
        self._shared_local_ray_cache = {}
        self._shared_ray_intrinsics = None
        self._refresh_shared_local_rays()
        self._create_offset_buffers(self._num_envs)
        self._frame_wp = wp.zeros(self._num_envs, dtype=wp.int64, device=self._device)
        self._frame = wp.to_torch(self._frame_wp)
        self._noise_rng = torch.Generator(device=self._device).manual_seed(self.cfg.seed)
        shape = (self._batch_size, self.num_rays)
        self._batch_starts_w = wp.empty(shape, dtype=wp.vec3f, device=self._device)
        self._batch_directions_w = wp.empty(shape, dtype=wp.vec3f, device=self._device)
        self.ray_hits_w = ProxyArray(wp.empty(shape, dtype=wp.vec3f, device=self._device))
        self._batch_distance = wp.empty(shape, dtype=wp.float32, device=self._device)
        self._batch_depth = wp.empty((self._batch_size, *self.image_shape, 1), dtype=wp.float32, device=self._device)
        self._pose_ray = wp.zeros((self._num_envs, 1), dtype=wp.vec3f, device=self._device)
        self._pose_start = wp.empty_like(self._pose_ray)
        self._pose_direction = wp.empty_like(self._pose_ray)
        self.last_depth_batch = None

    def set_intrinsic_matrices(self, matrices, focal_length=1.0, env_ids=None):
        indices = slice(None) if env_ids is None else env_ids
        matrices = matrices.to(device=self._device, dtype=self._data.intrinsic_matrices.torch.dtype)
        if torch.equal(self._data.intrinsic_matrices.torch[indices], matrices):
            # Episode resets can reapply the same calibration without changing rays.
            self._focal_length = focal_length
            return
        # Keep native singular-matrix error checking outside the frame hot path.
        inverse = torch.inverse(matrices)
        self._data.intrinsic_matrices.torch[indices] = matrices
        self._ray_inverse_intrinsics[indices] = inverse
        self._focal_length = focal_length
        self._refresh_shared_local_rays()

    def _refresh_shared_local_rays(self):
        # Only calibration setters synchronize to decide whether rays are shared.
        inverse = self._ray_inverse_intrinsics
        self._use_shared_local_rays = torch.equal(inverse, inverse[:1].expand_as(inverse))
        if not self._use_shared_local_rays:
            self._shared_local_ray_cache.clear()
            self._shared_ray_intrinsics = None
        elif self._shared_ray_intrinsics is None or not torch.equal(inverse[:1], self._shared_ray_intrinsics):
            self._shared_local_ray_cache.clear()
            self._shared_ray_intrinsics = inverse[:1].clone()

    def _local_rays(self, begin, end):
        if not self._use_shared_local_rays:
            return self._compute_local_rays(begin, end)
        count = end - begin
        if count not in self._shared_local_ray_cache:
            # Keep the original matmul batch shape, including a separate tail batch.
            starts, directions = self._compute_local_rays(begin, end)
            shared = []
            for values in (starts, directions):
                if not torch.equal(values, values[:1].expand_as(values)):
                    raise ValueError("Identical intrinsics produced nonidentical shared camera rays")
                # Clone one row so the expanded view owns no N-camera allocation.
                shared.append(values[:1].clone().expand_as(values))
            self._shared_local_ray_cache[count] = tuple(shared)
        return self._shared_local_ray_cache[count]

    def _compute_local_rays(self, begin, end):
        # Use the official pattern, including its pixel centers and frame convention.
        starts, directions = self.cfg.pattern_cfg.func(
            self.cfg.pattern_cfg, self._data.intrinsic_matrices.torch[begin:end], self._device
        )
        return starts.contiguous(), directions.contiguous()

    def set_mount_pitch(self, pitch, env_ids):
        """Apply per-episode source camera pitch around the fixed Navigation mount."""
        zero = torch.zeros_like(pitch)
        perturbation = math_utils.quat_from_euler_xyz(zero, -pitch, zero)
        base = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=self._device),
            origin=self.cfg.offset.convention,
            target="world",
        ).expand(len(env_ids), -1)
        wp.to_torch(self._offset_quat_wp)[env_ids] = math_utils.quat_mul(base, perturbation)

    def reset(self, env_ids=None, env_mask=None):
        mask = self._resolve_indices_and_mask(env_ids, env_mask)
        SensorBase.reset(self, env_mask=mask)
        self._update_frame(mask, frame_op=2)

    def _update_ray_infos(self, env_mask):
        # Pose-only update for the inherited camera pose setters.
        wp.launch(
            camera_kernels.update_ray_caster_kernel,
            dim=(self._num_envs, 1),
            inputs=[
                self._get_view_transforms_wp(),
                env_mask,
                self._offset_pos_wp,
                self._offset_quat_wp,
                self.drift.warp,
                self.ray_cast_drift.warp,
                self._pose_ray,
                self._pose_ray,
                int(camera_kernels.ALIGNMENT_BASE),
            ],
            outputs=[self._data.pos_w.warp, self._data.quat_w_world.warp, self._pose_start, self._pose_direction],
            device=self._device,
        )

    def _update_buffers_impl(self, env_mask):
        stream = torch_warp_stream(self._device)
        with wp.ScopedStream(stream) if stream is not None else nullcontext():
            self._update_frame(env_mask, frame_op=1)
            self._update_mesh_transforms()
            self._update_world_bounds()
            self._world_bvh.refit()
            transforms = self._get_view_transforms_wp()
            for begin in range(0, self._num_envs, self._batch_size):
                end = min(begin + self._batch_size, self._num_envs)
                count = end - begin
                mask = _slice(env_mask, begin, end)
                local_starts, local_directions = self._local_rays(begin, end)
                wp.launch(
                    camera_kernels.update_ray_caster_kernel,
                    dim=(count, self.num_rays),
                    inputs=[
                        _slice(transforms, begin, end),
                        mask,
                        _slice(self._offset_pos_wp, begin, end),
                        _slice(self._offset_quat_wp, begin, end),
                        _slice(self.drift.warp, begin, end),
                        _slice(self.ray_cast_drift.warp, begin, end),
                        wp.from_torch(local_starts, dtype=wp.vec3f),
                        wp.from_torch(local_directions, dtype=wp.vec3f),
                        int(camera_kernels.ALIGNMENT_BASE),
                    ],
                    outputs=[
                        _slice(self._data.pos_w.warp, begin, end),
                        _slice(self._data.quat_w_world.warp, begin, end),
                        self._batch_starts_w,
                        self._batch_directions_w,
                    ],
                    device=self._device,
                )
                wp.launch(
                    _raycast_shared_meshes,
                    dim=(count, self.num_rays),
                    inputs=[
                        self._world_bvh.id,
                        mask,
                        self._mesh_ids_wp,
                        self._mesh_positions_w,
                        self._mesh_orientations_w,
                        self._batch_starts_w,
                        self._batch_directions_w,
                        float(camera_kernels.CAMERA_RAYCAST_MAX_DIST),
                    ],
                    outputs=[self.ray_hits_w.warp, self._batch_distance],
                    device=self._device,
                )
                wp.launch(
                    camera_kernels.compute_distance_to_image_plane_to_image_masked_kernel,
                    dim=(count, self.num_rays),
                    inputs=[
                        mask,
                        _slice(self._data.quat_w_world.warp, begin, end),
                        self._batch_distance,
                        self._batch_directions_w,
                        self.image_shape[1],
                        True,
                        float(self.cfg.max_distance),
                        float(self.cfg.max_distance),
                    ],
                    outputs=[self._batch_depth],
                    device=self._device,
                )
                depth = wp.to_torch(self._batch_depth)[:count].permute(0, 3, 1, 2)
                noise = None
                if self.cfg.pixel_noise_std:
                    noise = (
                        torch.randn(depth.shape, generator=self._noise_rng, device=self._device)
                        * self.cfg.pixel_noise_std
                    )
                features = depth_features(depth, noise).permute(0, 2, 3, 1)
                current = self._data.output["policy_depth"].torch[begin:end]
                current.copy_(torch.where(wp.to_torch(mask)[:, None, None, None], features, current))
                # The history receives a depth measurement, including configured pixel noise.
                measured = depth if noise is None else 3 / (3 / depth.clamp(0.3, 24) + noise).clamp_min(1e-6)
                raw = self._data.output["distance_to_image_plane"].torch[begin:end]
                raw.copy_(torch.where(wp.to_torch(mask)[:, None, None, None], measured.permute(0, 2, 3, 1), raw))
                self.last_depth_batch = (begin, end, depth)


class _NavigationRayCasterMixin:
    """Preserve the customized Orbit sensor without modifying Isaac Lab."""

    def _update_ray_infos(self, env_mask):
        # The official yaw path already rotates starts and supplies the tracked
        # orientation. Navigation also rotates directions by that same yaw.
        super()._update_ray_infos(env_mask)
        if self.cfg.ray_alignment == "yaw":
            wp.launch(
                rotate_directions_with_yaw,
                dim=(self._num_envs, self.num_rays),
                inputs=[self._data._quat_w, env_mask, self.ray_directions.warp],
                outputs=[self._ray_directions_w],
                device=self._device,
            )
