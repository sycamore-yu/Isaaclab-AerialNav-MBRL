"""Full-resolution Isaac Lab depth rays with bounded image working memory.

Uses the official PhysX multi-mesh sensor and Warp ray-casting kernels. Only
the image buffers are batched; every configured pixel is traced before the
Zhang inverse-depth/noise/max-pooling transform.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import warp as wp
from isaaclab.sensors.ray_caster import kernels as camera_kernels
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.utils import math as math_utils
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp import kernels as warp_kernels
from isaaclab_physx.sensors.ray_caster.multi_mesh_ray_caster_camera import MultiMeshRayCasterCamera

from .depth_processing import depth_features


def _slice(array: wp.array, begin: int, end: int) -> wp.array:
    return array[begin:end]


class BatchedDepthCamera(MultiMeshRayCasterCamera):
    """An official sensor extension retaining pooled features for all drones.

    ``image_shape`` and intrinsic matrices describe the full 480 x 640 sensor.
    ``data.output['policy_depth']`` contains its transformed 12 x 16 result.
    ``last_depth_batch`` exposes the most recently computed raw depth batch
    for numerical checks without retaining all full-resolution images.
    """

    def _initialize_rays_impl(self):
        if self.cfg.data_types != ["distance_to_image_plane"] or self.cfg.update_mesh_ids:
            raise ValueError("The batched policy camera requires only image-plane depth, without mesh IDs.")
        self.num_rays = self.image_shape[0] * self.image_shape[1]
        self._batch_size = min(self._num_envs, self.cfg.batch_size)
        self._data.create_buffers(self._num_envs, self._device)
        self._data.intrinsic_matrices.torch[:, 2, 2] = 1.0
        self._data.image_shape = self.image_shape
        self._data._output = {
            "policy_depth": ProxyArray(wp.zeros((self._num_envs, 12, 16, 1), dtype=wp.float32, device=self._device))
        }
        self._data.info = {"policy_depth": {"source_image_shape": self.image_shape}}
        self._compute_intrinsic_matrices()
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
        self._dummy_normal = wp.empty((1, 1), dtype=wp.vec3f, device=self._device)
        self._dummy_face = wp.empty((1, 1), dtype=wp.int32, device=self._device)
        self._dummy_mesh = wp.empty((1, 1), dtype=wp.int16, device=self._device)
        self._pose_ray = wp.zeros((self._num_envs, 1), dtype=wp.vec3f, device=self._device)
        self._pose_start = wp.empty_like(self._pose_ray)
        self._pose_direction = wp.empty_like(self._pose_ray)
        self.last_depth_batch = None

    def set_intrinsic_matrices(self, matrices, focal_length=1.0, env_ids=None):
        indices = slice(None) if env_ids is None else env_ids
        self._data.intrinsic_matrices.torch[indices] = matrices.to(self._device)
        self._focal_length = focal_length

    def set_mount_pitch(self, pitch, env_ids):
        """Apply per-episode source camera pitch around the fixed NavRL mount."""
        zero = torch.zeros_like(pitch)
        perturbation = math_utils.quat_from_euler_xyz(zero, -pitch, zero)
        base = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=self._device),
            origin=self.cfg.offset.convention, target="world",
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
            inputs=[self._get_view_transforms_wp(), env_mask, self._offset_pos_wp, self._offset_quat_wp,
                    self.drift.warp, self.ray_cast_drift.warp, self._pose_ray, self._pose_ray,
                    int(camera_kernels.ALIGNMENT_BASE)],
            outputs=[self._data.pos_w.warp, self._data.quat_w_world.warp, self._pose_start, self._pose_direction],
            device=self._device,
        )

    def _update_buffers_impl(self, env_mask):
        stream = wp.stream_from_torch(torch.cuda.current_stream(self._device)) if self._device.startswith("cuda") else None
        with wp.ScopedStream(stream) if stream is not None else nullcontext():
            self._update_frame(env_mask, frame_op=1)
            self._update_mesh_transforms()
            transforms = self._get_view_transforms_wp()
            for begin in range(0, self._num_envs, self._batch_size):
                end = min(begin + self._batch_size, self._num_envs)
                count = end - begin
                mask = _slice(env_mask, begin, end)
                local_starts, local_directions = self.cfg.pattern_cfg.func(
                    self.cfg.pattern_cfg, self._data.intrinsic_matrices.torch[begin:end], self._device
                )
                wp.launch(
                    camera_kernels.update_ray_caster_kernel,
                    dim=(count, self.num_rays),
                    inputs=[_slice(transforms, begin, end), mask,
                            _slice(self._offset_pos_wp, begin, end), _slice(self._offset_quat_wp, begin, end),
                            _slice(self.drift.warp, begin, end), _slice(self.ray_cast_drift.warp, begin, end),
                            wp.from_torch(local_starts.contiguous(), dtype=wp.vec3f),
                            wp.from_torch(local_directions.contiguous(), dtype=wp.vec3f),
                            int(camera_kernels.ALIGNMENT_BASE)],
                    outputs=[_slice(self._data.pos_w.warp, begin, end),
                             _slice(self._data.quat_w_world.warp, begin, end),
                             self._batch_starts_w, self._batch_directions_w],
                    device=self._device,
                )
                wp.launch(
                    camera_kernels.fill_ray_hits_distance_inf_kernel,
                    dim=(count, self.num_rays), inputs=[mask, False],
                    outputs=[self.ray_hits_w.warp, self._batch_distance, self._dummy_normal], device=self._device,
                )
                wp.launch(
                    warp_kernels.raycast_dynamic_meshes_kernel,
                    dim=(self._mesh_ids_wp.shape[1], count, self.num_rays),
                    inputs=[mask, _slice(self._mesh_ids_wp, begin, end), self._batch_starts_w,
                            self._batch_directions_w, self.ray_hits_w.warp, self._batch_distance,
                            self._dummy_normal, self._dummy_face, self._dummy_mesh,
                            _slice(self._mesh_positions_w, begin, end), _slice(self._mesh_orientations_w, begin, end),
                            float(camera_kernels.CAMERA_RAYCAST_MAX_DIST), 0, 0, 0],
                    device=self._device,
                )
                wp.launch(
                    camera_kernels.compute_distance_to_image_plane_to_image_masked_kernel,
                    dim=(count, self.num_rays),
                    inputs=[mask, _slice(self._data.quat_w_world.warp, begin, end), self._batch_distance,
                            self._batch_directions_w, self.image_shape[1], True,
                            float(self.cfg.max_distance), float(self.cfg.max_distance)],
                    outputs=[self._batch_depth], device=self._device,
                )
                depth = wp.to_torch(self._batch_depth)[:count].permute(0, 3, 1, 2)
                noise = None
                if self.cfg.pixel_noise_std:
                    noise = torch.randn(depth.shape, generator=self._noise_rng, device=self._device) * self.cfg.pixel_noise_std
                features = depth_features(depth, noise).permute(0, 2, 3, 1)
                current = self._data.output["policy_depth"].torch[begin:end]
                current.copy_(torch.where(wp.to_torch(mask)[:, None, None, None], features, current))
                self.last_depth_batch = (begin, end, depth)
