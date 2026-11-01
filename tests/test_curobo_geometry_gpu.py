"""Optional real cuRoboV2 GPU geometry acceptance, not a flight-success test."""

import pytest
import torch

pytest.importorskip("curobo")
if not torch.cuda.is_available():
    pytest.skip("cuRoboV2 requires CUDA", allow_module_level=True)

import trimesh

from aerialnav.reference.curobo_geometry import CuroboGeometry
from aerialnav.tasks.navigation.geometry import TerrainDistance, primitive_signed_distance


@pytest.fixture(scope="module")
def backend():
    mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    mesh.apply_translation([0, 0, 1.0])
    terrain = TerrainDistance(mesh.vertices, mesh.faces, "cuda:0")
    sizes = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 5.0]], device="cuda")
    return CuroboGeometry(terrain, sizes, voxel_size=0.05)


def test_static_esdf_sign_distance_finite_difference_and_medial_escape(backend):
    p = torch.tensor(
        [[1.3, 0.1, 1.0], [0.5, 0.4, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 1.0]], device="cuda", requires_grad=True
    )
    value = backend.signed_distance(p)
    gradient = torch.autograd.grad(value.sum(), p)[0]
    torch.testing.assert_close(value, torch.tensor([0.3, -0.5, 1.0, -1.0], device="cuda"), atol=0.08, rtol=0)
    assert torch.isfinite(gradient).all() and (gradient.norm(dim=-1) > 0.1).all()
    for axis in range(3):
        delta = torch.zeros_like(p[:3])
        delta[:, axis] = 0.002
        finite = (
            backend.signed_distance(p[:3].detach() + delta) - backend.signed_distance(p[:3].detach() - delta)
        ) / 0.004
        torch.testing.assert_close(finite, gradient[:3, axis], atol=0.02, rtol=0.02)
    improved = backend.signed_distance(p.detach() + 0.08 * gradient)
    assert (improved > value.detach()).all()


def test_full_dynamic_box_cylinder_caps_deep_penetration_and_snapshots(backend):
    positions = torch.tensor([[10.0, 0.0, 2.0], [20.0, 0.0, 2.5]], device="cuda")
    p = torch.tensor(
        [
            [10.0, 0.0, 2.0],
            [10.1, 0.1, 2.1],
            [10.55, 0.6, 2.6],
            [10.8, 0.0, 2.0],
            [20.0, 0.0, 2.5],
            [20.1, 0.0, 2.5],
            [20.0, 0.0, 5.2],
            [20.6, 0.0, 5.2],
        ],
        device="cuda",
        requires_grad=True,
    )
    value = backend.dynamic_signed_distance(p, positions, backend.sizes)
    gradient = torch.autograd.grad(value.sum(), p, retain_graph=True)[0]
    native = primitive_signed_distance(p[:, None], positions, backend.sizes).min(-1).values
    torch.testing.assert_close(value, native, atol=0.0002, rtol=0.0001)
    assert torch.isfinite(gradient).all() and (gradient.norm(dim=-1) > 0.9).all()
    shifted = positions[None, None] + torch.tensor([0.0, 2.0, 0.0], device="cuda")
    q = p.detach()[:, None].clone().requires_grad_()
    values = backend.dynamic_signed_distance(q, shifted, backend.sizes)
    shifted.add_(100.0)
    assert torch.isfinite(torch.autograd.grad(values.sum(), q)[0]).all()
    torch.testing.assert_close(torch.autograd.grad(value.sum(), p)[0], gradient)


def test_batched_256_queries_and_collision_radius(backend):
    x = torch.linspace(-2, 2, 256, device="cuda")
    p = torch.stack([x, x * 0, x * 0 + 1.0], dim=-1).requires_grad_()
    gap = backend.signed_distance(p) - 0.3
    expected = p.detach()[:, 0].abs() - 1.3
    certain = expected.abs() > 0.1
    assert torch.equal((gap.detach() < 0)[certain], (expected < 0)[certain])
    assert torch.isfinite(torch.autograd.grad(gap.sum(), p)[0]).all()


def test_dynamic_query_keeps_obstacles_beyond_nearest_five_centres(backend):
    sizes = torch.tensor([[0.1, 0.1, 1.0]] * 7 + [[10.0, 2.0, 1.0]], device="cuda")
    full = CuroboGeometry(backend.native, sizes, voxel_size=0.1)
    positions = torch.tensor([[1.0 + i * 0.01, 0.0, 2.0] for i in range(7)] + [[5.0, 0.0, 2.0]], device="cuda")
    p = torch.tensor([[0.25, 0.0, 2.0]], device="cuda", requires_grad=True)
    distance, winner = full.dynamic_signed_distance(p, positions, sizes, return_index=True)
    assert winner.item() == 7 and distance.item() < 0
    assert torch.autograd.grad(distance.sum(), p)[0].norm() > 0.9


def test_static_only_stage_accepts_empty_dynamic_world(backend):
    sizes = torch.empty(0, 3, device="cuda")
    static = CuroboGeometry(backend.native, sizes, voxel_size=0.1)
    p = torch.tensor([[1.3, 0.1, 1.0]], device="cuda", requires_grad=True)
    distance, winner = static.dynamic_signed_distance(p, sizes, sizes, return_index=True)
    assert torch.isposinf(distance).all() and winner is None
    torch.testing.assert_close(torch.autograd.grad(distance.sum(), p)[0], torch.zeros_like(p))
    assert torch.isfinite(static.signed_distance(p)).all()
