"""PhysX navigation depth camera with shared Warp ray processing."""

from contextlib import nullcontext

import torch
import warp as wp
from isaaclab.sensors.ray_caster import kernels as camera_kernels
from isaaclab_physx.sensors.ray_caster.multi_mesh_ray_caster_camera import MultiMeshRayCasterCamera

from aerialnav.warp_utils import torch_warp_stream

from .sensor_mixins import _BatchedDepthCameraMixin


class BatchedDepthCamera(_BatchedDepthCameraMixin, MultiMeshRayCasterCamera):
    """PhysX depth camera with one aggregate body-pose read per update."""

    def _initialize_shared_mesh_tracking(self):
        """Resolve body owners and validate the aggregate view's actual row order."""
        self._clear_shared_mesh_tracking()
        body_paths, mesh_columns = [], []
        mesh_column = 0
        for view, target in zip(self._mesh_views, self._raycast_targets_cfg, strict=True):
            width = self._num_meshes_per_env[target.prim_expr]
            if target.track_mesh_transforms:
                if view is None or view.count != 1 or width != 1:
                    raise ValueError(
                        f"Shared camera target {target.prim_expr!r} must have one body and one mesh column."
                    )
                paths = [str(path) for path in view.prim_paths]
                if len(paths) != 1:
                    raise ValueError(f"Expected one resolved body path for camera target {target.prim_expr!r}.")
                body_paths.append(paths[0])
                mesh_columns.append(mesh_column)
            mesh_column += width

        count = len(body_paths)
        self.mesh_transform_info = {
            "mode": "aggregate_physx_view",
            "body_count": count,
            "mesh_order_body_paths": body_paths,
            "aggregate_body_paths": [],
            "identity_row_order": True,
            "mapping": [],
        }
        if count == 0:
            self._shared_mesh_tracking_ready = True
            return
        if len(set(body_paths)) != count:
            raise ValueError("Shared camera targets must resolve to distinct rigid-body owner paths.")
        offset = mesh_columns[0]
        if mesh_columns != list(range(offset, offset + count)):
            raise ValueError("Shared tracked camera targets must occupy contiguous mesh columns.")

        view = self._create_tracked_target_view(body_paths)
        actual_paths = [str(path) for path in view.prim_paths]
        if view.count != count or len(actual_paths) != count or len(set(actual_paths)) != count:
            raise ValueError("Aggregate PhysX camera view has an unexpected body count or duplicate body paths.")
        if set(actual_paths) != set(body_paths):
            raise ValueError("Aggregate PhysX camera view body paths do not match the resolved target owners.")
        row_by_path = {path: row for row, path in enumerate(actual_paths)}
        rows = [row_by_path[path] for path in body_paths]
        identity = rows == list(range(count))
        self._shared_mesh_view = view
        self._shared_mesh_rows = None if identity else torch.tensor(rows, device=self._device, dtype=torch.long)
        self._shared_mesh_count, self._shared_mesh_offset = count, offset
        self.mesh_transform_info.update(
            {
                "aggregate_body_paths": actual_paths,
                "identity_row_order": identity,
                "mapping": [
                    {"body_path": path, "mesh_column": column, "aggregate_row": row}
                    for path, column, row in zip(body_paths, mesh_columns, rows, strict=True)
                ],
            }
        )
        self._shared_mesh_tracking_ready = True

    def _update_mesh_transforms(self):
        if not getattr(self, "_shared_mesh_tracking_ready", False):
            raise RuntimeError("Shared camera transform tracking is not initialized.")
        if self._shared_mesh_view is None:
            return
        stream = torch_warp_stream(self._device)
        with wp.ScopedStream(stream) if stream is not None else nullcontext():
            transforms = self._shared_mesh_view.get_transforms()
            # Retain backing allocations through the asynchronous official copy.
            self._shared_mesh_transform_source = (
                transforms if isinstance(transforms, wp.array) else transforms.contiguous()
            )
            source = self._shared_mesh_transform_source
            transforms_wp = (
                source.view(wp.transformf)
                if isinstance(source, wp.array)
                else wp.from_torch(source).view(wp.transformf)
            )
            if self._shared_mesh_rows is not None:
                self._shared_mesh_transform_reordered = (
                    wp.to_torch(transforms_wp).reshape(-1, 7).index_select(0, self._shared_mesh_rows)
                )
                transforms_wp = wp.from_torch(self._shared_mesh_transform_reordered).view(wp.transformf)
            wp.launch(
                camera_kernels.copy_mesh_transforms_to_table_kernel,
                dim=(self._num_envs, self._shared_mesh_count),
                inputs=[
                    transforms_wp,
                    self._shared_mesh_count,
                    self._shared_mesh_offset,
                    True,
                    self._mesh_positions_w,
                    self._mesh_orientations_w,
                ],
                device=self._device,
            )
