"""Optional cuRoboV2 ESDF and full-obstacle query experiment.

Known height-field occupancy is retained; cuRobo PBA computes its metric ESDF.
This does not fuse depth images or change the policy's sensor information.
"""

import math
import time

import torch
import warp as wp
from curobo._src.geom.collision.wp_collision_kernel import compute_local_sdf_with_grad
from curobo._src.geom.data.data_cuboid import CuboidData, CuboidDataWarp
from curobo._src.geom.data.data_mesh import MeshData, MeshDataWarp
from curobo._src.geom.data.data_voxel import sample_voxel_sdf_with_grad
from curobo._src.geom.types import Mesh, SceneCfg
from curobo._src.perception.mapper.esdf.edt_parallel_banding import ParallelBandingEDT
from curobo._src.types.device_cfg import DeviceCfg

from aerialnav.tasks.navigation.geometry import _height_hit
from aerialnav.warp_utils import torch_warp_stream


@wp.kernel(enable_backward=False)
def _column_heights(
    mesh: wp.uint64,
    shape: wp.vec3i,
    origin: wp.vec3,
    vs: float,
    ray_top: float,
    ray_length: float,
    heights: wp.array2d(dtype=float),
):
    x, y = wp.tid()
    p = origin + wp.vec3(
        (float(x) - float(shape[0]) * 0.5 + 0.5) * vs, (float(y) - float(shape[1]) * 0.5 + 0.5) * vs, 0.0
    )
    hit = _height_hit(mesh, p, ray_top, ray_length)
    height = -1.0e6
    if hit.result:
        height = ray_top - hit.t
    heights[x, y] = wp.max(height, 0.01)


@wp.func
def _occupied(heights: wp.array2d(dtype=float), x: int, y: int, z: int, shape: wp.vec3i, origin: wp.vec3, vs: float):
    x = wp.clamp(x, 0, shape[0] - 1)
    y = wp.clamp(y, 0, shape[1] - 1)
    pz = origin[2] + (float(z) - float(shape[2]) * 0.5 + 0.5) * vs
    return pz < heights[x, y]


@wp.kernel(enable_backward=False)
def _seed_sites(
    heights: wp.array2d(dtype=float), shape: wp.vec3i, origin: wp.vec3, vs: float, sites: wp.array(dtype=wp.int32)
):
    i = wp.tid()
    z = i % shape[2]
    y = (i // shape[2]) % shape[1]
    x = i // (shape[2] * shape[1])
    here = _occupied(heights, x, y, z, shape, origin, vs)
    boundary = (
        _occupied(heights, x - 1, y, z, shape, origin, vs) != here
        or _occupied(heights, x + 1, y, z, shape, origin, vs) != here
        or _occupied(heights, x, y - 1, z, shape, origin, vs) != here
        or _occupied(heights, x, y + 1, z, shape, origin, vs) != here
        or _occupied(heights, x, y, z - 1, shape, origin, vs) != here
        or _occupied(heights, x, y, z + 1, shape, origin, vs) != here
    )
    value = wp.int32(-1)
    if boundary:
        value = wp.int32(x) | (wp.int32(y) << 10) | (wp.int32(z) << 20)
    sites[i] = value


@wp.kernel(enable_backward=False)
def _finish_esdf(
    heights: wp.array2d(dtype=float),
    shape: wp.vec3i,
    origin: wp.vec3,
    vs: float,
    sites: wp.array(dtype=wp.int32),
    field: wp.array(dtype=wp.float16),
):
    i = wp.tid()
    z = i % shape[2]
    y = (i // shape[2]) % shape[1]
    x = i // (shape[2] * shape[1])
    site = sites[i]
    sx, sy, sz = site & 1023, (site >> 10) & 1023, (site >> 20) & 1023
    distance = (wp.length(wp.vec3(float(x - sx), float(y - sy), float(z - sz))) + 0.5) * vs
    if _occupied(heights, x, y, z, shape, origin, vs):
        distance = -distance
    field[i] = wp.float16(distance)


@wp.kernel(enable_backward=False)
def _query_static(
    points: wp.array(dtype=wp.vec3),
    field: wp.array(dtype=wp.float16),
    shape: wp.vec3i,
    origin: wp.vec3,
    vs: float,
    distance: wp.array(dtype=float),
    gradient: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    p = points[i] - origin
    # Outside the padded map only the task's infinite ground plane remains.
    result = wp.vec4(points[i][2] - 0.01, 0.0, 0.0, 1.0)
    if (
        wp.abs(p[0]) < (float(shape[0]) * 0.5 - 1.0) * vs
        and wp.abs(p[1]) < (float(shape[1]) * 0.5 - 1.0) * vs
        and wp.abs(p[2]) < (float(shape[2]) * 0.5 - 1.0) * vs
    ):
        result = sample_voxel_sdf_with_grad(field, 0, p, shape, vs, 10000.0)
        normal = wp.vec3(result[1], result[2], result[3])
        if result[0] < 0.0 and wp.length(normal) < 1.0e-5:
            # At medial-axis interpolation plateaus, choose a one-sided escape as in the native geometry.
            best = wp.float32(result[0])
            for axis in range(3):
                for sign in range(2):
                    delta = wp.vec3(0.0)
                    direction = 1.0
                    if sign == 1:
                        direction = -1.0
                    delta[axis] = direction * vs
                    probe = sample_voxel_sdf_with_grad(field, 0, p + delta, shape, vs, 10000.0)
                    if probe[0] > best:
                        best = probe[0]
                        normal = delta / vs
            result = wp.vec4(result[0], normal[0], normal[1], normal[2])
    distance[i] = result[0]
    gradient[i] = wp.vec3(result[1], result[2], result[3])


@wp.kernel(enable_backward=False)
def _query_dynamic(
    points: wp.array(dtype=wp.vec3),
    poses: wp.array3d(dtype=wp.vec3),
    kinds: wp.array(dtype=wp.int32),
    cuboids: CuboidDataWarp,
    meshes: MeshDataWarp,
    count: int,
    query_count: int,
    distance: wp.array(dtype=float),
    gradient: wp.array(dtype=wp.vec3),
    winner: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    batch, query = i // query_count, i % query_count
    best = wp.float32(1.0e6)
    normal = wp.vec3(0.0)
    index = wp.int32(0)
    for obstacle in range(count):
        local = points[i] - poses[batch, query, obstacle]
        result = wp.vec4(1.0e6, 0.0, 0.0, 0.0)
        if kinds[obstacle] == 0:
            result = compute_local_sdf_with_grad(cuboids, 0, obstacle, local, 100.0)
        else:
            result = compute_local_sdf_with_grad(meshes, 0, obstacle, local, 100.0)
        if result[0] < best:
            best = result[0]
            index = obstacle
            # cuRobo's primitive/mesh helper returns the collision-cost direction (-dSDF/dp).
            normal = -wp.vec3(result[1], result[2], result[3])
    distance[i] = best
    gradient[i] = normal
    winner[i] = wp.int64(index)


class _Query(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, backend, positions):
        points = query.detach().reshape(-1, 3).contiguous()
        distance = torch.empty(points.shape[0], device=query.device, dtype=query.dtype)
        gradient = torch.empty_like(points)
        stream = torch_warp_stream(query.device)
        if positions is None:
            wp.launch(
                _query_static,
                dim=len(points),
                inputs=[
                    wp.from_torch(points, dtype=wp.vec3),
                    wp.from_torch(backend.field, dtype=wp.float16),
                    wp.vec3i(*backend.shape),
                    wp.vec3(*backend.origin),
                    backend.voxel_size,
                    wp.from_torch(distance),
                    wp.from_torch(gradient, dtype=wp.vec3),
                ],
                device=backend.device,
                stream=stream,
            )
            winner = torch.zeros(query.shape[:-1], dtype=torch.long, device=query.device)
        else:
            shape = query.shape[:-1]
            count = len(backend.sizes)
            positions = torch.broadcast_to(positions.detach(), (*shape, count, 3)).contiguous()
            queries = math.prod(shape[1:])
            winner = torch.empty(len(points), dtype=torch.long, device=query.device)
            wp.launch(
                _query_dynamic,
                dim=len(points),
                inputs=[
                    wp.from_torch(points, dtype=wp.vec3),
                    wp.from_torch(positions.reshape(shape[0], queries, count, 3), dtype=wp.vec3),
                    wp.from_torch(backend.kinds),
                    backend.cuboids.to_warp(),
                    backend.meshes.to_warp(),
                    count,
                    queries,
                    wp.from_torch(distance),
                    wp.from_torch(gradient, dtype=wp.vec3),
                    wp.from_torch(winner),
                ],
                device=backend.device,
                stream=stream,
            )
        ctx.save_for_backward(gradient.reshape_as(query))
        ctx.mark_non_differentiable(winner)
        return distance.reshape(query.shape[:-1]), winner.reshape(query.shape[:-1])

    @staticmethod
    def backward(ctx, grad_output, _winner_gradient):
        (gradient,) = ctx.saved_tensors
        return grad_output[..., None] * gradient, None, None


class CuroboGeometry:
    """One shared static ESDF and full shared dynamic world; query batches are drones."""

    geometry_mode = "curobo_v2_known_occupancy_pba_esdf"
    projection_precision = "fp16_esdf_trilinear_fp32_gradient"

    def __init__(self, terrain, sizes, voxel_size=0.0625):
        import trimesh

        if not str(terrain.device).startswith("cuda"):
            raise ValueError("cuRoboV2 experiment requires CUDA")
        self.native = terrain
        self.device, self.sizes = terrain.device, sizes
        self.origin = [float((terrain.bounds[0][i] + terrain.bounds[1][i]) * 0.5) for i in range(2)] + [2.0]
        extents = [float(terrain.bounds[1][i] - terrain.bounds[0][i] + 8.0) for i in range(2)] + [8.0]
        # PBA packs each site coordinate in 10 bits; large maps need a coarser shared grid.
        while max(math.ceil(extent / voxel_size) for extent in extents) > 1024:
            voxel_size *= 2
        self.voxel_size = voxel_size
        self.shape = tuple(math.ceil(extent / voxel_size) for extent in extents)
        stream = torch_warp_stream(self.device)
        heights = torch.empty(self.shape[:2], device=self.device)
        n = math.prod(self.shape)
        sites = torch.empty(n, dtype=torch.int32, device=self.device)
        self.field = torch.empty(n, dtype=torch.float16, device=self.device)
        started = time.perf_counter()
        arguments = [wp.from_torch(heights), wp.vec3i(*self.shape), wp.vec3(*self.origin), voxel_size]
        wp.launch(
            _column_heights,
            dim=self.shape[:2],
            inputs=[
                terrain.height_mesh.id,
                wp.vec3i(*self.shape),
                wp.vec3(*self.origin),
                voxel_size,
                terrain._ray_top,
                terrain._ray_length,
                wp.from_torch(heights),
            ],
            device=self.device,
            stream=stream,
        )
        wp.launch(_seed_sites, dim=n, inputs=[*arguments, wp.from_torch(sites)], device=self.device, stream=stream)
        edt = ParallelBandingEDT(self.shape, voxel_size, torch.device(self.device))
        edt.propagate(sites)
        wp.launch(
            _finish_esdf,
            dim=n,
            inputs=[*arguments, wp.from_torch(sites), wp.from_torch(self.field, dtype=wp.float16)],
            device=self.device,
            stream=stream,
        )
        torch.cuda.synchronize(self.device)
        self.mapping_seconds = time.perf_counter() - started
        cfg = DeviceCfg(device=torch.device(self.device))
        self.kinds = (sizes[:, 2] > 1).to(torch.int32)
        self.cuboids = CuboidData.create_cache(max(len(sizes), 1), 1, cfg)
        self.cuboids.dims[0, : len(sizes), :3] = sizes
        mesh_list = []
        for index, size in enumerate(sizes.cpu().numpy()):
            # shortcut: 128-sided cylinders approximate the task's analytic cylinders; max radial error is <0.16 mm at r<=0.5m.
            mesh = trimesh.creation.cylinder(radius=float(size[0] / 2), height=float(size[2]), sections=128)
            mesh_list.append(
                Mesh(
                    name=f"dynamic_{index}",
                    pose=[0, 0, 0, 1, 0, 0, 0],
                    vertices=mesh.vertices.tolist(),
                    faces=mesh.faces.reshape(-1).tolist(),
                )
            )
        self.meshes = MeshData.from_scene_cfg(SceneCfg(mesh=mesh_list), cfg, max_n=max(len(sizes), 1), max_dist=100.0)

    def __getattr__(self, name):
        return getattr(self.native, name)

    def signed_distance(self, query):
        return _Query.apply(query, self, None)[0]

    def dynamic_signed_distance(self, query, positions, sizes, *, return_index=False):
        if sizes.shape != self.sizes.shape or sizes.data_ptr() != self.sizes.data_ptr():
            raise ValueError("Dynamic obstacle geometry changed after cuRobo cache construction")
        if len(sizes) == 0:
            value = query.sum(-1) * 0 + torch.inf
            return (value, None) if return_index else value
        value, winner = _Query.apply(query, self, positions)
        return (value, winner) if return_index else value

    def metadata(self):
        return dict(
            backend=self.geometry_mode,
            voxel_size=self.voxel_size,
            shape=list(self.shape),
            origin=self.origin,
            mapping_seconds=self.mapping_seconds,
            map_storage_mib=self.field.numel() * self.field.element_size() / 2**20,
            knowledge="known shared terrain occupancy; not depth fusion",
            dynamic_geometry="cuRobo cuboids and 128-sided watertight cylinder meshes",
            dynamic_obstacles=len(self.sizes),
            boundary="all returned query gradients are per-call snapshots",
        )
