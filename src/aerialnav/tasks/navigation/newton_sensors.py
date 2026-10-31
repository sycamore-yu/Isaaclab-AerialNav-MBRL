"""Newton navigation sensors without Isaac Sim or PhysX imports."""

import warp as wp
from isaaclab_newton.sensors.ray_caster.legacy_ray_caster import LegacyMultiMeshRayCasterCamera, LegacyRayCaster

from .sensor_mixins import _BatchedDepthCameraMixin, _NavigationRayCasterMixin


@wp.kernel(enable_backward=False)
def _set_shared_obstacle_mesh_poses(
    positions: wp.array(dtype=wp.vec3f),
    active_indices: wp.array(dtype=wp.int32),
    offset: int,
    mesh_positions: wp.array2d(dtype=wp.vec3f),
    mesh_orientations: wp.array2d(dtype=wp.quatf),
):
    env, column = wp.tid()
    active = active_indices[column]
    if active >= 0:
        mesh_positions[env, offset + column] = positions[active]
    else:
        mesh_positions[env, offset + column] = wp.vec3f(0.0, 0.0, -1000.0)
    mesh_orientations[env, offset + column] = wp.quat_identity()


class NewtonNavigationRayCaster(_NavigationRayCasterMixin, LegacyRayCaster):
    """Newton LiDAR with the same Warp mesh and yaw-only ray alignment."""


class NewtonBatchedDepthCamera(_BatchedDepthCameraMixin, LegacyMultiMeshRayCasterCamera):
    """Newton depth camera using global non-physics USD mesh geometry.

    The physics model contains no obstacle bodies. Warp's existing shared-mesh
    raycaster receives obstacle transforms directly from DynamicObstacleMotion,
    without USD writes, site registrations or per-world rigid-body poses.
    """

    def set_motion(self, motion, active_ids):
        """Bind authoritative GPU state; called after scene initialization/reset."""
        import torch

        self._shared_motion = motion
        columns = len(self._raycast_targets_cfg) - 2
        if len(active_ids) != len(motion.position) or (len(active_ids) and int(active_ids.max()) >= columns):
            raise ValueError("Shared obstacle geometry and motion indices disagree")
        index = torch.full((columns,), -1, device=motion.position.device, dtype=torch.int32)
        index[active_ids] = torch.arange(len(active_ids), device=index.device, dtype=torch.int32)
        self._shared_active_indices = wp.from_torch(index)
        self._shared_active_index_storage = index
        self._shared_motion_positions = wp.from_torch(motion.position.contiguous(), dtype=wp.vec3f)
        self.mesh_transform_info["mode"] = "shared_motion_gpu_poses_without_rigid_bodies"
        self.mesh_transform_info["active_obstacles"] = len(motion.position)
        self.world_mesh_info["dynamic_meshes"] = columns

    def _clear_shared_mesh_tracking(self):
        super()._clear_shared_mesh_tracking()
        self._shared_motion = None
        self._shared_active_indices = None
        self._shared_motion_positions = None
        self._shared_active_index_storage = None

    def _initialize_shared_mesh_tracking(self):
        """All global targets are static USD geometry, not tracked physics sites."""
        self._clear_shared_mesh_tracking()
        for target in self._raycast_targets_cfg:
            if target.track_mesh_transforms:
                raise ValueError("Newton shared dynamic geometry must not register physics sites")
            if self._num_meshes_per_env[target.prim_expr] != 1:
                raise ValueError(f"Expected one shared mesh per target: {target.prim_expr}")
        # All queries use row zero. Own one row so the original N-world tables
        # are released, instead of retaining their allocation through a view.
        self._mesh_ids_wp = wp.clone(self._mesh_ids_wp[:1])
        self._mesh_positions_w = wp.clone(self._mesh_positions_w[:1])
        self._mesh_orientations_w = wp.clone(self._mesh_orientations_w[:1])
        self._shared_mesh_count, self._shared_mesh_offset = 0, 2
        self.mesh_transform_info = {
            "mode": "shared_motion_gpu_poses_without_rigid_bodies",
            "body_count": 0,
            "mesh_order_body_paths": [],
            "site_indices": [],
            "identity_row_order": True,
            "mapping": [],
            "mesh_pose_rows": 1,
        }
        self._shared_mesh_tracking_ready = True

    def _update_mesh_transforms(self):
        if not getattr(self, "_shared_mesh_tracking_ready", False):
            raise RuntimeError("Shared camera transform tracking is not initialized.")
        if self._shared_motion_positions is None or self._shared_active_indices.shape[0] == 0:
            return
        wp.launch(
            _set_shared_obstacle_mesh_poses,
            dim=(1, self._shared_active_indices.shape[0]),
            inputs=[
                self._shared_motion_positions,
                self._shared_active_indices,
                2,
                self._mesh_positions_w,
                self._mesh_orientations_w,
            ],
            device=self._device,
        )

    def _get_view_transforms_wp(self):
        self._update_newton_site_transforms(
            self._sensor_site_indices, self._newton_pose_w, self._newton_pos_w.warp, self._newton_quat_w.warp
        )
        return self._newton_pose_w
