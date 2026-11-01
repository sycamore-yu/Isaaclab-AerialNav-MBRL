"""Signed training clearance with verified exits on the unchanged terrain mesh.

Isaac Lab's steep-slope conversion can leave folded/intersecting triangles and
zero-thickness fins.  Training occupancy is conservatively defined below the
highest projected triangle; it is not claimed to equal a watertight physical
solid.  Outside that occupancy the original mesh distance is retained.  Inside,
a negative distance and frozen-anchor escape derivative use a verified exit on
an actual triangle.  This is a signed clearance proxy, not an exact mesh SDF.
The BVH-selected triangle is projected in float64 and stored in float32 to
avoid tangential drift on tall, narrow faces; this refines both query APIs.
"""

from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

from aerialnav.warp_utils import torch_warp_stream


@wp.kernel
def _closest_mesh_point(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    triangles: wp.array(dtype=wp.vec3),
    output: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    query = wp.mesh_query_point_no_sign(mesh, points[index], 1.0e6)
    if query.result:
        surface, _selected_face = _refine_nearby_surface(mesh, points[index], triangles, query.face)
        output[index] = surface
    else:
        output[index] = wp.vec3(1.0e6, 1.0e6, 1.0e6)


@wp.func
def _height_hit(mesh: wp.uint64, point: wp.vec3, ray_top: float, ray_length: float):
    return wp.mesh_query_ray(mesh, wp.vec3(point[0], point[1], ray_top), wp.vec3(0.0, 0.0, -1.0), ray_length)


@wp.func
def _inside_height_field(mesh: wp.uint64, point: wp.vec3, ray_top: float, ray_length: float):
    hit = _height_hit(mesh, point, ray_top, ray_length)
    return hit.result and point[2] < ray_top - hit.t


@wp.func
def _exit_direction_valid(
    mesh: wp.uint64,
    surface: wp.vec3,
    direction: wp.vec3,
    ray_top: float,
    ray_length: float,
    probe: float,
):
    """Require a numerically separated crossing at this point, not its whole face.

    A grazing candidate can change occupancy after float32 ray/probe rounding.
    Keep occupancy itself strict, but reject ambiguous candidate exits so the
    search chooses a better-separated original-mesh point. This is a numerical
    guard, not a proof of a universal error bound for arbitrary triangle shapes.
    """
    before = surface - probe * direction
    after = surface + probe * direction
    before_hit = _height_hit(mesh, before, ray_top, ray_length)
    after_hit = _height_hit(mesh, after, ray_top, ray_length)
    height_scale = wp.max(1.0, wp.max(wp.abs(ray_top), wp.abs(ray_top - ray_length)))
    guard = wp.max(1.0e-6, 9.5367431640625e-7 * height_scale)  # Eight float32 epsilons.
    before_clear = before_hit.result and ray_top - before_hit.t - before[2] > guard
    after_clear = not after_hit.result or after[2] - (ray_top - after_hit.t) > guard
    return before_clear and after_clear


@wp.func
def _surface_escape_normal(
    mesh: wp.uint64,
    surface: wp.vec3,
    normal: wp.vec3,
    ray_top: float,
    ray_length: float,
    probe: float,
):
    result = wp.vec3(0.0, 0.0, 0.0)
    if _exit_direction_valid(mesh, surface, normal, ray_top, ray_length, probe):
        result = normal
    elif _exit_direction_valid(mesh, surface, -normal, ray_top, ray_length, probe):
        result = -normal
    return result


@wp.func
def _as_vec3d(value: wp.vec3):
    return wp.vec3d(wp.float64(value[0]), wp.float64(value[1]), wp.float64(value[2]))


@wp.func
def _as_vec3f(value: wp.vec3d):
    return wp.vec3(wp.float32(value[0]), wp.float32(value[1]), wp.float32(value[2]))


@wp.func
def _closest_edge_point_double(point: wp.vec3d, a: wp.vec3d, b: wp.vec3d):
    edge = b - a
    denominator = wp.dot(edge, edge)
    if denominator <= wp.float64(0.0):
        return a
    fraction = wp.clamp(wp.dot(point - a, edge) / denominator, wp.float64(0.0), wp.float64(1.0))
    return a + fraction * edge


@wp.func
def _closest_triangle_point_double(point: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
    """Refine the selected face by plane projection and dominant-axis barycentrics."""
    ab = b - a
    ac = c - a
    normal = wp.cross(ab, ac)
    normal_squared = wp.dot(normal, normal)
    if normal_squared > wp.float64(0.0):
        projected = point - normal * (wp.dot(point - a, normal) / normal_squared)
        relative = projected - a
        i = int(0)  # noqa: UP018 - explicitly typed Warp index
        j = int(1)  # noqa: UP018 - explicitly typed Warp index
        if wp.abs(normal[0]) >= wp.abs(normal[1]) and wp.abs(normal[0]) >= wp.abs(normal[2]):
            i = 1
            j = 2
        elif wp.abs(normal[1]) >= wp.abs(normal[2]):
            i = 0
            j = 2
        determinant = ab[i] * ac[j] - ab[j] * ac[i]
        beta = (relative[i] * ac[j] - relative[j] * ac[i]) / determinant
        gamma = (ab[i] * relative[j] - ab[j] * relative[i]) / determinant
        if beta >= wp.float64(0.0) and gamma >= wp.float64(0.0) and beta + gamma <= wp.float64(1.0):
            return projected
    best = _closest_edge_point_double(point, a, b)
    best_squared = wp.dot(point - best, point - best)
    candidate = _closest_edge_point_double(point, b, c)
    candidate_squared = wp.dot(point - candidate, point - candidate)
    if candidate_squared < best_squared:
        best = candidate
        best_squared = candidate_squared
    candidate = _closest_edge_point_double(point, c, a)
    candidate_squared = wp.dot(point - candidate, point - candidate)
    if candidate_squared < best_squared:
        best = candidate
    return best


@wp.func
def _closest_triangle_point(point: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    return _as_vec3f(_closest_triangle_point_double(_as_vec3d(point), _as_vec3d(a), _as_vec3d(b), _as_vec3d(c)))


@wp.func
def _closest_edge_point(point: wp.vec3, a: wp.vec3, b: wp.vec3):
    return _as_vec3f(_closest_edge_point_double(_as_vec3d(point), _as_vec3d(a), _as_vec3d(b)))


@wp.func
def _refine_nearby_surface(mesh: wp.uint64, point: wp.vec3, triangles: wp.array(dtype=wp.vec3), seed_face: int):
    """Refine the seed and correct face selection within a bounded near-surface neighborhood."""
    point64 = _as_vec3d(point)
    best = _closest_triangle_point_double(
        point64,
        _as_vec3d(triangles[3 * seed_face]),
        _as_vec3d(triangles[3 * seed_face + 1]),
        _as_vec3d(triangles[3 * seed_face + 2]),
    )
    best_squared = wp.dot(point64 - best, point64 - best)
    winner = seed_face
    radius = wp.float64(0.002)
    if best_squared <= radius * radius:
        # Ten micrometres of AABB padding covers float32 bound rounding.
        extent = wp.vec3(0.00201, 0.00201, 0.00201)
        candidates = wp.mesh_query_aabb(mesh, point - extent, point + extent)
        face = int(0)  # noqa: UP018 - mutable Warp query index
        while wp.mesh_query_aabb_next(candidates, face):
            candidate = _closest_triangle_point_double(
                point64,
                _as_vec3d(triangles[3 * face]),
                _as_vec3d(triangles[3 * face + 1]),
                _as_vec3d(triangles[3 * face + 2]),
            )
            squared = wp.dot(point64 - candidate, point64 - candidate)
            if squared < best_squared:
                best = candidate
                best_squared = squared
                winner = face
    return _as_vec3f(best), winner


@wp.kernel
def _signed_mesh_query(
    mesh: wp.uint64,
    height_mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    triangle_points: wp.array(dtype=wp.vec3),
    face_normals: wp.array(dtype=wp.vec3),
    face_centers: wp.array(dtype=wp.vec3),
    ray_top: float,
    ray_length: float,
    normal_probe: float,
    exit_search_limit: float,
    closest: wp.array(dtype=wp.vec3),
    mesh_closest: wp.array(dtype=wp.vec3),
    outward: wp.array(dtype=wp.vec3),
    inside: wp.array(dtype=wp.int32),
    supported: wp.array(dtype=wp.int32),
    height: wp.array(dtype=float),
    exit_valid: wp.array(dtype=wp.int32),
    used_exit_search: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    point = points[index]
    nearest = wp.mesh_query_point_no_sign(mesh, point, 1.0e6)
    hit = _height_hit(height_mesh, point, ray_top, ray_length)
    supported[index] = int(hit.result)
    height[index] = ray_top - hit.t
    is_inside = hit.result and point[2] < height[index]
    inside[index] = int(is_inside)
    exit_valid[index] = 0
    used_exit_search[index] = 0
    if nearest.result:
        surface, surface_face = _refine_nearby_surface(mesh, point, triangle_points, nearest.face)
        mesh_closest[index] = surface
        closest[index] = surface
        delta = point - surface
        distance = wp.length(delta)
        normal = face_normals[surface_face]
        if is_inside:
            direction = wp.vec3(0.0, 0.0, 0.0)
            legal = False
            if distance > 0.0:
                direction = -delta / distance
                legal = _exit_direction_valid(height_mesh, surface, direction, ray_top, ray_length, normal_probe)
            else:
                direction = _surface_escape_normal(height_mesh, surface, normal, ray_top, ray_length, normal_probe)
                legal = wp.length_sq(direction) > 0.0
            if not legal:
                used_exit_search[index] = 1
                # An actual highest ray hit is a conservative, original-mesh
                # roof exit.  Never invent a vertical curtain at a projection
                # boundary or accept an internal zero-distance sheet as exit.
                surface = wp.vec3(point[0], point[1], height[index])
                direction = wp.vec3(0.0, 0.0, 1.0)
                best_distance = wp.length(surface - point)
                legal = _exit_direction_valid(height_mesh, surface, direction, ray_top, ray_length, normal_probe)
                roof_distance = best_distance
                search_limit = wp.min(roof_distance, exit_search_limit)
                radius = wp.min(search_limit, wp.max(0.2, 4.0 * distance))
                searching = bool(True)  # noqa: UP018 - mutable Warp loop state, not a compile-time constant
                # Only anomalous inside points pay for this local BVH search.
                # Partial face coverage is not solved by deleting whole faces:
                # every candidate point is tested.  Missing the closest legal
                # fragment may select a farther exit, hence the explicit proxy.
                while searching:
                    extent = wp.vec3(radius, radius, radius)
                    candidates = wp.mesh_query_aabb(mesh, point - extent, point + extent)
                    face = int(0)  # noqa: UP018 - mesh query writes this Warp loop variable
                    found = bool(False)  # noqa: UP018 - mutable Warp loop state
                    while wp.mesh_query_aabb_next(candidates, face):
                        a = triangle_points[3 * face]
                        b = triangle_points[3 * face + 1]
                        c = triangle_points[3 * face + 2]
                        projection = _closest_triangle_point(point, a, b, c)
                        for choice in range(5):
                            candidate = projection
                            if choice == 1:
                                candidate = face_centers[face]
                            elif choice == 2:
                                candidate = _closest_edge_point(point, a, b)
                            elif choice == 3:
                                candidate = _closest_edge_point(point, b, c)
                            elif choice == 4:
                                candidate = _closest_edge_point(point, c, a)
                            candidate_distance = wp.length(candidate - point)
                            if candidate_distance < best_distance and candidate_distance <= radius:
                                candidate_direction = wp.vec3(0.0, 0.0, 0.0)
                                candidate_valid = False
                                if candidate_distance > 0.0:
                                    candidate_direction = (candidate - point) / candidate_distance
                                    candidate_valid = _exit_direction_valid(
                                        height_mesh, candidate, candidate_direction, ray_top, ray_length, normal_probe
                                    )
                                else:
                                    candidate_direction = _surface_escape_normal(
                                        height_mesh, candidate, face_normals[face], ray_top, ray_length, normal_probe
                                    )
                                    candidate_valid = wp.length_sq(candidate_direction) > 0.0
                                if candidate_valid:
                                    surface = candidate
                                    direction = candidate_direction
                                    best_distance = candidate_distance
                                    legal = True
                                    found = True
                    if found or radius >= search_limit:
                        searching = False
                    else:
                        radius = wp.min(2.0 * radius, search_limit)
                closest[index] = surface
            exit_valid[index] = int(legal)
            outward[index] = direction
        elif distance > 0.0:
            outward[index] = delta / distance
        else:
            direction = _surface_escape_normal(height_mesh, surface, normal, ray_top, ray_length, normal_probe)
            if wp.length_sq(direction) == 0.0:
                # A genuine zero-thickness fin has free space on both sides.
                # Retain its original unsigned surface with one finite normal.
                direction = normal
            outward[index] = direction
    else:
        closest[index] = wp.vec3(1.0e6, 1.0e6, 1.0e6)
        mesh_closest[index] = closest[index]
        outward[index] = wp.vec3(0.0, 0.0, 1.0)


@dataclass(frozen=True)
class TerrainSignedQuery:
    """Detached geometric choices plus a signed clearance/escape derivative.

    ``distance`` has units metres and is negative inside.  ``surface_height``
    is NaN where ``has_surface`` is false; no nearest-height extrapolation fills
    holes or points outside the mesh projection. ``closest_point`` is the
    verified selected exit for inside queries and the original mesh nearest
    point outside. It need not be the mathematically closest valid exit.
    ``outward_normal`` is a frozen-anchor escape direction, not a claim that
    every fallback differentiates the re-queried proxy value exactly.
    """

    distance: torch.Tensor
    closest_point: torch.Tensor
    inside: torch.Tensor
    surface_height: torch.Tensor
    has_surface: torch.Tensor
    outward_normal: torch.Tensor
    mesh_closest_point: torch.Tensor
    exit_valid: torch.Tensor
    used_exit_search: torch.Tensor


class TerrainDistance:
    """Query unchanged mesh surfaces with conservative occupancy/exit semantics."""

    geometry_mode = "signed_heightfield_occupancy_exit_proxy"
    projection_precision = "float64_selected_face_refinement_float32_storage"
    near_surface_refinement_radius_m = 0.002
    exit_search_radius_limit_m = 2.0

    def __init__(self, vertices, faces, device):
        self.device = device
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not len(faces):
            raise ValueError("TerrainDistance requires nonempty vertices (V, 3) and triangular faces (F, 3).")
        if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("Terrain vertices must be finite and face indices must reference those vertices.")
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        norm = np.linalg.norm(normals, axis=-1)
        nondegenerate = norm > 1e-12
        # Height-field cliff conversion can create zero-area triangles.  They
        # have no surface and must not win nearest-point/normal queries.
        faces, triangles, normals, norm = (
            faces[nondegenerate],
            triangles[nondegenerate],
            normals[nondegenerate],
            norm[nondegenerate],
        )
        if not len(faces):
            raise ValueError("TerrainDistance requires at least one nondegenerate triangle.")
        normals = normals / norm[:, None]
        projected = np.abs(normals[:, 2]) > 1e-8
        if not projected.any():
            raise ValueError("A height-field terrain must have triangles with nonzero XY projection.")
        points = wp.array(vertices, dtype=wp.vec3, device=device)
        self.mesh = wp.Mesh(
            points=points,
            indices=wp.array(faces.reshape(-1), dtype=wp.int32, device=device),
        )
        # A vertical face has zero projected area and is parallel to the ray.
        # Omitting it from *only* the occupancy BVH avoids coplanar cliff-hit
        # ambiguity; the complete mesh remains the nearest-surface geometry.
        self.height_mesh = wp.Mesh(
            points=points,
            indices=wp.array(faces[projected].reshape(-1), dtype=wp.int32, device=device),
        )
        self._face_normals = wp.array(normals, dtype=wp.vec3, device=device)
        self._face_centers = wp.array(triangles.mean(axis=1), dtype=wp.vec3, device=device)
        self._triangle_points = wp.array(triangles.reshape(-1, 3), dtype=wp.vec3, device=device)
        self.bounds = (vertices.min(axis=0), vertices.max(axis=0))
        self._ray_top = float(self.bounds[1][2]) + 1.0
        self._ray_length = float(self.bounds[1][2] - self.bounds[0][2]) + 2.0
        self._normal_probe = max(1e-4, float(np.abs(vertices).max()) * 8 * np.finfo(np.float32).eps)
        self.exit_height_guard_m = max(
            1e-6,
            8 * float(np.finfo(np.float32).eps) * max(1.0, abs(self._ray_top), abs(self._ray_top - self._ray_length)),
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
                raise ValueError("The Navigation height-field terrain must be triangulated.")
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
        stream = torch_warp_stream(points.device)
        wp.launch(
            _closest_mesh_point,
            dim=len(points),
            inputs=[self.mesh.id, wp.from_torch(points, dtype=wp.vec3), self._triangle_points],
            outputs=[wp.from_torch(output, dtype=wp.vec3)],
            device=self.device,
            stream=stream,
        )
        return output.reshape_as(query)

    def signed_query(self, query):
        """Signed clearance proxy with selected original-mesh exits.

        Query positions may have any leading shape and float32/float64 dtype.
        BVH geometry/inside selection is detached and performed in one Warp
        launch. Ordinary nearest points are kept when they are legal exits;
        anomalous inside points search original triangle candidates instead.
        The returned norm keeps the live query graph with a frozen anchor. In
        particular a vertical roof fallback has an upward escape derivative,
        which need not equal the derivative of its re-queried sloped-roof value.
        This is not an exact SDF or a physical watertight-solid reconstruction.
        """
        if query.ndim < 1 or query.shape[-1] != 3 or query.dtype not in (torch.float32, torch.float64):
            raise ValueError("Signed terrain queries require floating positions with shape (..., 3).")
        shape = query.shape[:-1]
        points = query.detach().reshape(-1, 3).to(dtype=torch.float32).contiguous()
        closest, outward, mesh_closest = torch.empty_like(points), torch.empty_like(points), torch.empty_like(points)
        inside = torch.empty(len(points), device=points.device, dtype=torch.int32)
        supported = torch.empty_like(inside)
        height = torch.empty(len(points), device=points.device, dtype=torch.float32)
        exit_valid, used_exit_search = torch.empty_like(inside), torch.empty_like(inside)
        if len(points):
            stream = torch_warp_stream(points.device)
            wp.launch(
                _signed_mesh_query,
                dim=len(points),
                inputs=[
                    self.mesh.id,
                    self.height_mesh.id,
                    wp.from_torch(points, dtype=wp.vec3),
                    self._triangle_points,
                    self._face_normals,
                    self._face_centers,
                    self._ray_top,
                    self._ray_length,
                    self._normal_probe,
                    self.exit_search_radius_limit_m,
                ],
                outputs=[
                    wp.from_torch(closest, dtype=wp.vec3),
                    wp.from_torch(mesh_closest, dtype=wp.vec3),
                    wp.from_torch(outward, dtype=wp.vec3),
                    wp.from_torch(inside, dtype=wp.int32),
                    wp.from_torch(supported, dtype=wp.int32),
                    wp.from_torch(height, dtype=wp.float32),
                    wp.from_torch(exit_valid, dtype=wp.int32),
                    wp.from_torch(used_exit_search, dtype=wp.int32),
                ],
                device=self.device,
                stream=stream,
            )
        closest, outward = (
            closest.to(dtype=query.dtype).reshape_as(query),
            outward.to(dtype=query.dtype).reshape_as(query),
        )
        inside, supported = inside.bool().reshape(shape), supported.bool().reshape(shape)
        length = (query - closest).norm(dim=-1)
        distance = torch.where(inside, -length, length)
        # A zero-valued linear correction supplies a face normal only at an
        # exact zero distance.  Nonzero distances retain precisely sign*norm
        # values/gradients; adding this near-but-not-at-zero would double them.
        surface_gradient = ((query - query.detach()) * outward).sum(-1)
        distance = distance + torch.where(length.detach() == 0, surface_gradient, torch.zeros_like(length))
        height = height.to(dtype=query.dtype).reshape(shape).masked_fill(~supported, torch.nan)
        exit_valid = exit_valid.bool().reshape(shape)
        # A numerically unverified roof must never silently become a negative
        # "exit" distance. Propagate failure to existing finiteness gates without
        # adding a host/GPU synchronization to every policy geometry query.
        distance = distance.masked_fill(inside & ~exit_valid, torch.nan)
        return TerrainSignedQuery(
            distance,
            closest,
            inside,
            height,
            supported,
            outward,
            mesh_closest.to(dtype=query.dtype).reshape_as(query),
            exit_valid,
            used_exit_search.bool().reshape(shape),
        )

    def signed_distance(self, query):
        """Return the signed metre proxy; use signed_query for exit diagnostics."""
        return self.signed_query(query).distance


def _closest_dynamic_surface(point, dynamic_position, dynamic_size):
    """The original Navigation dynamic surface projection, with detached selection."""
    delta = point[:, None] - dynamic_position[None]
    extent = dynamic_size[None] * 0.5
    box = delta.clamp(-extent, extent)
    radius = extent[..., 0:1]
    radial = delta[..., :2]
    radial = radial * (radius / radial.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_max(1)
    cylinder = torch.cat((radial, delta[..., 2:3].clamp(-extent[..., 2:3], extent[..., 2:3])), dim=-1)
    local = torch.where((dynamic_size[:, 2] > 1)[None, :, None], cylinder, box)
    candidates = local + dynamic_position[None]
    distance, indices = (candidates - point[:, None]).square().sum(-1).min(-1)
    return candidates[torch.arange(len(point), device=point.device), indices], distance


def escape_abs(value):
    """Choose the positive one-sided derivative at a symmetry plane."""
    return torch.where(value >= 0, value, -value)


def escape_norm_xy(value):
    radius = value[..., :2].norm(dim=-1)
    return torch.where(radius > 0, radius, value[..., 0])


def primitive_signed_distance(query, positions, sizes):
    """Exact point SDF; size-Z > 1 denotes Navigation's finite Z cylinders.

    At centre/axis ties select one escape branch, never an averaged zero.
    """
    delta = query - positions
    box = escape_abs(delta) - sizes * 0.5
    cylinder = torch.stack(
        (escape_norm_xy(delta) - sizes[..., 0] * 0.5, escape_abs(delta[..., 2]) - sizes[..., 2] * 0.5), dim=-1
    )

    def sdf(offset):
        return offset.clamp_min(0).norm(dim=-1) + offset.max(dim=-1).values.clamp_max(0)

    return torch.where(sizes[..., 2] > 1, sdf(cylinder), sdf(box))


def dynamic_signed_distance(query, positions, sizes, *, return_index=False):
    """Check shared or per-query poses; retain only the selected body's graph."""
    shape = query.shape[:-1]
    if positions.shape[-2:] != sizes.shape or sizes.ndim != 2 or sizes.shape[-1] != 3:
        raise ValueError("Expected positions (..., obstacles, 3) and sizes (obstacles, 3)")
    positions = torch.broadcast_to(positions.detach(), (*shape, len(sizes), 3))
    if len(sizes) == 0:
        distance = query.sum(-1) * 0 + torch.inf
        return (distance, None) if return_index else distance
    point = query.reshape(-1, 3)
    with torch.no_grad():
        best = torch.full_like(point[:, 0], torch.inf)
        winner = torch.zeros(len(point), dtype=torch.long, device=query.device)
        for start in range(0, len(sizes), 32):
            stop = min(start + 32, len(sizes))
            distance = primitive_signed_distance(
                point[:, None], positions[..., start:stop, :].reshape(-1, stop - start, 3), sizes[None, start:stop]
            )
            value, index = distance.min(-1)
            winner = torch.where(value < best, index + start, winner)
            best = torch.minimum(value, best)
        index = winner.reshape(*shape, 1, 1).expand(*shape, 1, 3)
        selected_position = positions.gather(-2, index).squeeze(-2).reshape(-1, 3)
        selected_size = sizes[winner]
    distance = primitive_signed_distance(point, selected_position, selected_size).reshape(shape)
    return (distance, winner.reshape(shape)) if return_index else distance


def nearest_obstacle_signed_distance(
    query,
    static_signed_distance,
    dynamic_position,
    dynamic_size,
    ground_z=0.01,
    *,
    dynamic_velocity=None,
    dynamic_backend=None,
):
    """Minimum signed static/ground/full dynamic surface distance, before radius."""
    if static_signed_distance.shape != query.shape[:-1]:
        raise ValueError("static_signed_distance must have shape query.shape[:-1], not closest-point vectors.")
    fixed = torch.minimum(static_signed_distance, query[..., 2] - ground_z)
    moving, winner = (dynamic_signed_distance if dynamic_backend is None else dynamic_backend)(
        query, dynamic_position, dynamic_size, return_index=True
    )
    distance = torch.minimum(fixed, moving)
    if dynamic_velocity is None:
        return distance
    # The fixed terrain and ground have zero velocity, including minimum ties.
    velocity = torch.zeros_like(query)
    if winner is not None:
        velocity = torch.where((moving < fixed)[..., None], dynamic_velocity[winner], velocity)
    return distance, velocity.detach()


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
        nearest[:, 2] = 0.01  # Navigation's shared ground plane.
        best_distance = (nearest - measured).square().sum(-1)
        closest = static_nearest.reshape(-1, 3)
        distance = (closest - measured).square().sum(-1)
        nearest = torch.where((distance < best_distance)[:, None], closest, nearest)
        best_distance = torch.minimum(best_distance, distance)
        if len(dynamic_position):
            closest, distance = _closest_dynamic_surface(measured, dynamic_position, dynamic_size)
            nearest = torch.where((distance < best_distance)[:, None], closest, nearest)
    return (nearest - point).reshape(shape)
