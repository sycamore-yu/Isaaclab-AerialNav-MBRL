"""Nearest-surface targets for Zhang's differentiable collision losses."""

import numpy as np
import torch
import warp as wp


@wp.kernel
def _closest_mesh_point(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), output: wp.array(dtype=wp.vec3)):
    index = wp.tid()
    query = wp.mesh_query_point_no_sign(mesh, points[index], 1.0e6)
    if query.result:
        output[index] = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
    else:
        output[index] = wp.vec3(1.0e6, 1.0e6, 1.0e6)


class TerrainDistance:
    """Query the actual triangulated terrain, including overwritten height cells."""

    def __init__(self, vertices, faces, device):
        self.device = device
        self.mesh = wp.Mesh(
            points=wp.array(np.asarray(vertices, dtype=np.float32), dtype=wp.vec3, device=device),
            indices=wp.array(np.asarray(faces, dtype=np.int32).reshape(-1), dtype=wp.int32, device=device),
        )

    @classmethod
    def from_stage(cls, stage, device):
        from pxr import Usd, UsdGeom

        points, faces, offset = [], [], 0
        transforms = UsdGeom.XformCache()
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/ground")):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            if not np.all(counts == 3):
                raise ValueError("The NavRL height-field terrain must be triangulated.")
            transform = np.asarray(transforms.GetLocalToWorldTransform(prim))
            points.append(vertices @ transform[:3, :3] + transform[3, :3])
            faces.append(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3) + offset)
            offset += len(vertices)
        if not points:
            raise RuntimeError("No terrain mesh found under /World/ground.")
        return cls(np.concatenate(points), np.concatenate(faces), device)

    def closest_points(self, query):
        points = query.detach().reshape(-1, 3).contiguous()
        output = torch.empty_like(points)
        stream = wp.stream_from_torch(torch.cuda.current_stream(points.device)) if points.is_cuda else None
        wp.launch(
            _closest_mesh_point, dim=len(points),
            inputs=[self.mesh.id, wp.from_torch(points, dtype=wp.vec3)],
            outputs=[wp.from_torch(output, dtype=wp.vec3)], device=self.device, stream=stream,
        )
        return output.reshape_as(query)


def nearest_obstacle_vector(query, static_nearest, dynamic_position, dynamic_size):
    """Return the vector from each query to the closest obstacle surface.

    Geometric selection is detached, as in the source simulator. The returned
    vector retains the query-position derivative used by the avoidance loss.
    """
    shape = query.shape
    point = query.reshape(-1, 3)
    with torch.no_grad():
        measured = point.detach()
        nearest = measured.clone()
        nearest[:, 2] = 0.01  # NavRL's shared ground plane.
        best_distance = (nearest - measured).square().sum(-1)
        closest = static_nearest.reshape(-1, 3)
        distance = (closest - measured).square().sum(-1)
        nearest = torch.where((distance < best_distance)[:, None], closest, nearest)
        best_distance = torch.minimum(best_distance, distance)
        if len(dynamic_position):
            delta = measured[:, None] - dynamic_position[None]
            extent = dynamic_size[None] * 0.5
            box = delta.clamp(-extent, extent)
            radius = extent[..., 0:1]
            radial = delta[..., :2]
            radial = radial * (radius / radial.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_max(1)
            cylinder = torch.cat((radial, delta[..., 2:3].clamp(-extent[..., 2:3], extent[..., 2:3])), dim=-1)
            local = torch.where((dynamic_size[:, 2] > 1)[None, :, None], cylinder, box)
            candidates = local + dynamic_position[None]
            distance, indices = (candidates - measured[:, None]).square().sum(-1).min(-1)
            closest = candidates[torch.arange(len(point), device=point.device), indices]
            nearest = torch.where((distance < best_distance)[:, None], closest, nearest)
    return (nearest - point).reshape(shape)
