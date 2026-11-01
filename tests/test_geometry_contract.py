"""Signed values, escape gradients and complete sphere/obstacle coverage."""

import pytest
import torch
import trimesh
from test_diffnav_contract import loss_history

from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.diffnav.losses import diffnav_transition_costs
from aerialnav.reference.dcol import dynamic_min_alpha, hybrid_collision_gap
from aerialnav.tasks.navigation.geometry import (
    TerrainDistance,
    dynamic_signed_distance,
    nearest_obstacle_signed_distance,
    primitive_signed_distance,
)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("open_bottom", [False, True])
def test_static_sign_escape_sides_corners_deep_surface_and_ground(device, open_bottom):
    mesh = trimesh.creation.box(extents=(2, 2, 4))
    mesh.apply_translation((0, 0, 2))
    if open_bottom:
        mesh.update_faces(mesh.face_normals[:, 2] > -0.5)
    terrain = TerrainDistance(mesh.vertices, mesh.faces, device)
    points = [[1.2, 0, 2], [1.2, 1.2, 2], [0.9, 0, 2], [0.9, 0.9, 2], [0, 0, 2], [1, 0, 2], [0, 0, 4.2], [0, 0, 3.9]]
    q = torch.tensor(points, device=device, requires_grad=True)
    sdf = terrain.signed_distance(q)
    expected = q.new_tensor([0.2, 0.2 * 2**0.5, -0.1, -0.1, -1, 0, 0.2, -0.1])
    torch.testing.assert_close(sdf, expected, atol=2e-6, rtol=2e-6)
    gradient = torch.autograd.grad(sdf.sum(), q)[0]
    torch.testing.assert_close(gradient.norm(dim=-1), torch.ones(8, device=device))
    assert torch.isfinite(gradient).all()
    assert (terrain.signed_distance(q.detach() + 0.02 * gradient) > sdf.detach()).all()
    # Central differences only away from nonsmooth medial axes and edges.
    for axis in range(3):
        delta = torch.zeros_like(q)
        delta[:, axis] = 1e-3
        fd = (terrain.signed_distance(q.detach() + delta) - terrain.signed_distance(q.detach() - delta)) / 2e-3
        torch.testing.assert_close(gradient[[0, 1, 2, 6, 7], axis], fd[[0, 1, 2, 6, 7]], atol=2e-4, rtol=2e-4)
    empty = q.new_empty(0, 3)
    ground = q.new_tensor([[3, 3, -2.0]]).requires_grad_()
    gap = nearest_obstacle_signed_distance(ground, terrain.signed_distance(ground), empty, empty) - 0.3
    assert gap.item() < -2
    torch.testing.assert_close(torch.autograd.grad(gap.sum(), ground)[0], ground.new_tensor([[0, 0, 1]]))
    # The DCOL static branch must preserve the negative interior distance.
    hybrid = hybrid_collision_gap(q, 0.3, terrain.signed_distance(q), empty, empty)
    torch.testing.assert_close(hybrid, sdf - 0.3)


@pytest.mark.parametrize("size", [[2.0, 1.0, 1.0], [1.0, 1.0, 5.0]])
def test_dynamic_interior_centre_axis_and_loss_gradients(size):
    center = torch.tensor([[0.0, 0.0, 2.5]], dtype=torch.float64)
    sizes = torch.tensor([size], dtype=torch.float64)
    q = (
        center + torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.0, 0.0, 0.1]], dtype=torch.float64)
    ).requires_grad_()
    sdf = dynamic_signed_distance(q, center, sizes)
    assert (sdf < 0).all()
    grad = torch.autograd.grad(sdf.sum(), q, retain_graph=True)[0]
    torch.testing.assert_close(grad.norm(dim=-1), torch.ones(3, dtype=q.dtype))
    assert (dynamic_signed_distance(q.detach() + grad * 0.01, center, sizes) > sdf.detach()).all()
    alpha = dynamic_min_alpha(q, 0.3, center, sizes)
    dcol_grad = torch.autograd.grad(alpha.sum(), q, retain_graph=True)[0]
    assert (dcol_grad.norm(dim=-1) > 0).all()
    assert (dynamic_min_alpha(q.detach() + dcol_grad * 0.01, 0.3, center, sizes) > alpha.detach()).all()
    history = loss_history(torch.zeros(3, 3, 3, dtype=q.dtype))
    history["obstacle_signed_distance"] = [sdf[:, None].expand(-1, 10)] * 3
    losses, _, _ = diffnav_transition_costs(
        history, DiffNavTrainCfg(physics_dt=0.01, policy_dt=0.1, variant="step", velocity_loss_mode="variant_default")
    )
    for name in ["obstacle_avoidance", "collision"]:
        descent = -torch.autograd.grad(losses[name], q, retain_graph=True)[0]
        assert (descent * grad).sum(-1).gt(0).all()


@pytest.mark.parametrize("size", [[2.0, 1.0, 1.0], [1.0, 1.0, 5.0]])
def test_dynamic_finite_differences_and_finite_cap_corners(size):
    sizes = torch.tensor(size, dtype=torch.float64)
    center = torch.zeros(3, dtype=torch.float64)
    q = torch.tensor([[1.3, 0.7, 0.8], [0.17, 0.11, 0.12], [0.1, 0.1, 3.0]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: primitive_signed_distance(x, center, sizes), (q,))
    if size[2] > 1:
        cap = primitive_signed_distance(torch.tensor([[0.0, 0.0, 2.81]], dtype=q.dtype), center, sizes)
        torch.testing.assert_close(cap, torch.tensor([0.31], dtype=q.dtype))
        assert (cap - 0.3 > 0).all()
    else:
        corner = primitive_signed_distance(torch.tensor([[1.2, 0.7, 0.7]], dtype=q.dtype), center, sizes)
        torch.testing.assert_close(corner, torch.tensor([0.2 * 3**0.5], dtype=q.dtype))
        assert (corner - 0.3 > 0).all()


def test_sixth_center_collision_and_snapshot_ownership():
    q = torch.tensor([[0.0, 0.0, 2.0]], requires_grad=True)
    positions = torch.tensor([[0.0, 0.0, 3.0]] * 5 + [[2.0, 0.0, 2.0]])
    sizes = torch.tensor([[0.1, 0.1, 0.1]] * 5 + [[4.0, 1.0, 1.0]])
    distance = dynamic_signed_distance(q, positions, sizes)
    assert (distance - 0.3 <= 0).all()
    positions.add_(100)
    sizes.mul_(0.01)
    torch.testing.assert_close(torch.autograd.grad(distance.sum(), q)[0], torch.tensor([[-1.0, 0.0, 0.0]]))
