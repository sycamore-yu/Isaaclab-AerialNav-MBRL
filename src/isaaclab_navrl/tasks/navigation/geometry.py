"""Signed terrain and primitive distances for spherical-drone clearance."""

import numpy as np
import torch
import warp as wp


@wp.kernel
def _signed_mesh_distance(mesh: wp.uint64, points: wp.array(dtype=wp.vec3),
                          distances: wp.array(dtype=float), normals: wp.array(dtype=wp.vec3)):
    index = wp.tid()
    query = wp.mesh_query_point_no_sign(mesh, points[index], 1.0e6)
    if query.result:
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        delta = points[index] - closest
        distance = wp.length(delta)
        # Height fields are open surfaces: winding-number signs do not apply.
        # An upward ray exits the terrain solid iff the query is underneath it.
        ray = wp.mesh_query_ray(mesh, points[index], wp.vec3(0.0, 0.0, 1.0), 1.0e6)
        sign = 1.0
        if ray.result and ray.normal[2] > 0.0:
            sign = -1.0
        a = wp.mesh_eval_position(mesh, query.face, 1.0, 0.0)
        b = wp.mesh_eval_position(mesh, query.face, 0.0, 1.0)
        c = wp.mesh_eval_position(mesh, query.face, 0.0, 0.0)
        normal = wp.normalize(wp.cross(b - a, c - a))
        if distance > 1.0e-7:
            normal = sign * delta / distance
        distances[index] = sign * distance
        normals[index] = normal
    else:
        distances[index] = 1.0e6
        normals[index] = wp.vec3(0.0, 0.0, 0.0)


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

    def signed_distance(self, query):
        points = query.detach().reshape(-1, 3).contiguous()
        output = torch.empty(len(points), device=points.device, dtype=points.dtype)
        normals = torch.empty_like(points)
        stream = wp.stream_from_torch(torch.cuda.current_stream(points.device)) if points.is_cuda else None
        wp.launch(
            _signed_mesh_distance, dim=len(points),
            inputs=[self.mesh.id, wp.from_torch(points, dtype=wp.vec3)],
            outputs=[wp.from_torch(output), wp.from_torch(normals, dtype=wp.vec3)], device=self.device, stream=stream,
        )
        # Freeze the selected surface, retaining its outward first derivative.
        correction = ((query.reshape(-1, 3) - points) * normals).sum(-1)
        return (output + correction).reshape(query.shape[:-1])


def escape_abs(value):
    """Choose the positive one-sided derivative at a symmetry plane."""
    return torch.where(value >= 0, value, -value)


def escape_norm_xy(value):
    radius = value[..., :2].norm(dim=-1)
    return torch.where(radius > 0, radius, value[..., 0])


def primitive_signed_distance(query, positions, sizes):
    """Exact point SDF; size-Z > 1 denotes NavRL's finite Z cylinders.

    At centre/axis ties select one escape branch, never an averaged zero.
    """
    delta = query - positions
    box = escape_abs(delta) - sizes * 0.5
    cylinder = torch.stack((escape_norm_xy(delta) - sizes[..., 0] * 0.5,
                            escape_abs(delta[..., 2]) - sizes[..., 2] * 0.5), dim=-1)

    def sdf(offset):
        return offset.clamp_min(0).norm(dim=-1) + offset.max(dim=-1).values.clamp_max(0)

    return torch.where(sizes[..., 2] > 1, sdf(cylinder), sdf(box))


def dynamic_signed_distance(query, positions, sizes):
    """Check every obstacle; retain only the selected body's backward graph."""
    if len(positions) == 0:
        return query.sum(-1) * 0 + torch.inf
    point = query.reshape(-1, 3)
    with torch.no_grad():
        best = torch.full_like(point[:, 0], torch.inf)
        winner = torch.zeros(len(point), dtype=torch.long, device=query.device)
        for start in range(0, len(positions), 32):
            distance = primitive_signed_distance(
                point[:, None], positions[None, start:start + 32], sizes[None, start:start + 32])
            value, index = distance.min(-1)
            winner = torch.where(value < best, index + start, winner)
            best = torch.minimum(value, best)
        selected_position, selected_size = positions[winner], sizes[winner]
    return primitive_signed_distance(point, selected_position, selected_size).reshape(query.shape[:-1])


def obstacle_signed_distance(query, static_distance, positions, sizes, ground_z=0.01):
    """Minimum signed surface distance; subtract the drone radius once."""
    return torch.minimum(torch.minimum(static_distance, query[..., 2] - ground_z),
                         dynamic_signed_distance(query, positions, sizes))
