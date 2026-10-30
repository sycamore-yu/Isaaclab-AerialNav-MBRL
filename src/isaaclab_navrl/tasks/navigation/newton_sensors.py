"""Newton navigation sensors without Isaac Sim or PhysX imports."""

import warp as wp
from isaaclab.sensors.ray_caster import kernels as camera_kernels
from isaaclab_newton.sensors.ray_caster.legacy_ray_caster import LegacyMultiMeshRayCasterCamera, LegacyRayCaster

from .sensor_mixins import _BatchedDepthCameraMixin, _NavRLRayCasterMixin


class NewtonNavRLRayCaster(_NavRLRayCasterMixin, LegacyRayCaster):
    """Newton LiDAR with the same Warp mesh and yaw-only ray alignment."""


class NewtonBatchedDepthCamera(_BatchedDepthCameraMixin, LegacyMultiMeshRayCasterCamera):
    """Newton depth camera with one aggregate site-pose read per update."""

    def _clear_shared_mesh_tracking(self):
        super()._clear_shared_mesh_tracking()
        self._shared_mesh_site_indices = None
        self._shared_mesh_pose = None
        self._shared_mesh_pos = None
        self._shared_mesh_quat = None

    def _initialize_shared_mesh_tracking(self):
        """Track the first synchronized body copy in the configured mesh order."""
        self._clear_shared_mesh_tracking()
        sites, mesh_columns, target_paths = [], [], []
        mesh_column = 0
        for view, target in zip(self._mesh_views, self._raycast_targets_cfg, strict=True):
            width = self._num_meshes_per_env[target.prim_expr]
            if target.track_mesh_transforms:
                if view is None or width != 1:
                    raise ValueError(
                        f"Shared camera target {target.prim_expr!r} must have one body and one mesh column."
                    )
                # Newton may resolve a global site or one body copy per world.
                # Scene updates broadcast the same obstacle pose to every copy;
                # the shared-camera initialization verifies those poses match.
                values = view.numpy().reshape(-1)
                if len(values) not in (1, self._num_envs):
                    raise ValueError(
                        f"Newton camera target {target.prim_expr!r} must resolve to one site or one site per environment."
                    )
                sites.append(int(values[0]))
                mesh_columns.append(mesh_column)
                target_paths.append(target.prim_expr)
            mesh_column += width
        count = len(sites)
        if len(set(sites)) != count:
            raise ValueError("Shared camera targets must resolve to distinct Newton sites.")
        offset = mesh_columns[0] if count else 0
        if mesh_columns != list(range(offset, offset + count)):
            raise ValueError("Shared tracked camera targets must occupy contiguous mesh columns.")
        self._shared_mesh_count, self._shared_mesh_offset = count, offset
        self.mesh_transform_info = {
            "mode": "aggregate_newton_sites",
            "body_count": count,
            "mesh_order_body_paths": target_paths,
            "site_indices": sites,
            "identity_row_order": True,
            "mapping": [
                {"body_path": path, "mesh_column": column, "site_index": site}
                for path, column, site in zip(target_paths, mesh_columns, sites, strict=True)
            ],
        }
        if count:
            self._shared_mesh_site_indices = wp.array(sites, dtype=wp.int32, device=self._device)
            self._shared_mesh_pose = wp.empty(count, dtype=wp.transformf, device=self._device)
            self._shared_mesh_pos = wp.empty(count, dtype=wp.vec3f, device=self._device)
            self._shared_mesh_quat = wp.empty(count, dtype=wp.quatf, device=self._device)
        self._shared_mesh_tracking_ready = True

    def _update_mesh_transforms(self):
        if not getattr(self, "_shared_mesh_tracking_ready", False):
            raise RuntimeError("Shared camera transform tracking is not initialized.")
        if self._shared_mesh_site_indices is None:
            return
        self._update_newton_site_transforms(
            self._shared_mesh_site_indices, self._shared_mesh_pose, self._shared_mesh_pos, self._shared_mesh_quat
        )
        wp.launch(
            camera_kernels.copy_mesh_poses_to_table_kernel,
            dim=(self._num_envs, self._shared_mesh_count),
            inputs=[
                self._shared_mesh_pos,
                self._shared_mesh_quat,
                self._shared_mesh_count,
                self._shared_mesh_offset,
                True,
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
