"""Geometric/value/gradient contracts for the DCOL comparison branch."""

import math

import pytest
import torch

from aerialnav.reference.dcol import (
    dynamic_min_alpha,
    sphere_box_alpha,
    sphere_cylinder_alpha,
    sphere_sphere_alpha,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _bisect_reference(query, radius, center, extent, cylinder=False):
    """Independent bracketing of the original shared-scale intersection."""
    with torch.no_grad():
        delta = query - center
        low = torch.zeros_like(radius)
        high = delta.norm(dim=-1) / radius
        for _ in range(70):
            alpha = (low + high) * 0.5
            if cylinder:
                radial = (delta[..., :2].norm(dim=-1) - alpha * extent[..., 0]).clamp_min(0)
                vertical = (delta[..., 2].abs() - alpha * extent[..., 1]).clamp_min(0)
                distance = torch.sqrt(radial.square() + vertical.square())
            else:
                distance = (delta.abs() - alpha[..., None] * extent).clamp_min(0).norm(dim=-1)
            separated = distance > alpha * radius
            low = torch.where(separated, alpha, low)
            high = torch.where(separated, high, alpha)
        return (low + high) * 0.5


def test_spheres_match_analytic_scale_contact_and_penetration():
    point = torch.tensor([[2.0, 0, 0], [1.0, 0, 0], [0.5, 0, 0], [0, 0, 0]], dtype=torch.float64, requires_grad=True)
    value = sphere_sphere_alpha(point, 0.3, torch.zeros(3), 0.7)
    torch.testing.assert_close(value, torch.tensor([2.0, 1.0, 0.5, 0.0], dtype=torch.float64))
    value.sum().backward()
    torch.testing.assert_close(
        point.grad, torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0], [0, 0, 0]], dtype=torch.float64)
    )


@pytest.mark.parametrize("cylinder", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_analytic_root_matches_independent_common_scaling_bisection(cylinder, dtype):
    generator = torch.Generator().manual_seed(413)
    point = 3 * torch.randn(256, 3, generator=generator, dtype=dtype)
    radius = 0.15 + 0.5 * torch.rand(256, generator=generator, dtype=dtype)
    center = torch.randn(256, 3, generator=generator, dtype=dtype)
    extent = 0.15 + 1.5 * torch.rand(256, 2 if cylinder else 3, generator=generator, dtype=dtype)
    if cylinder:
        actual = sphere_cylinder_alpha(point, radius, center, extent[:, 0], extent[:, 1])
    else:
        actual = sphere_box_alpha(point, radius, center, extent)
    expected = _bisect_reference(point, radius, center, extent, cylinder)
    tolerance = 1.0e-5 if dtype == torch.float32 else 1.0e-11
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_box_faces_corners_and_planar_limit_have_correct_scale_and_gradient():
    point = torch.tensor([[0.8, 0, 0], [0.7, 0, 0], [1.2, 0, 0]], dtype=torch.float64, requires_grad=True)
    value = sphere_box_alpha(point, 0.3, torch.zeros(3), torch.tensor([0.5, 0.6, 0.7]))
    torch.testing.assert_close(value, torch.tensor([1.0, 0.875, 1.5], dtype=torch.float64), atol=3e-8, rtol=3e-8)
    value.sum().backward()
    torch.testing.assert_close(point.grad, torch.tensor([[1.25, 0, 0]] * 3, dtype=torch.float64), atol=3e-8, rtol=3e-8)
    corner = torch.ones(1, 3, dtype=torch.float64)
    expected_corner = math.sqrt(3) / (math.sqrt(3) * 0.5 + 0.3)
    torch.testing.assert_close(
        sphere_box_alpha(corner, 0.3, torch.zeros(3), torch.full((3,), 0.5)),
        torch.tensor([expected_corner], dtype=torch.float64),
    )
    # A zero-thickness rectangle is a plane locally, away from its edges.
    above_plane = torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float64, requires_grad=True)
    alpha = sphere_box_alpha(above_plane, 0.3, torch.zeros(3), torch.tensor([2.0, 2.0, 0.0]))
    torch.testing.assert_close(alpha, torch.ones(1, dtype=torch.float64))
    alpha.sum().backward()
    torch.testing.assert_close(above_plane.grad, torch.tensor([[0.0, 0.0, 1 / 0.3]], dtype=torch.float64))


@pytest.mark.parametrize("cylinder", [False, True])
def test_implicit_query_radius_and_shape_gradients_match_finite_differences(cylinder):
    center = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64, requires_grad=True)
    offsets = (
        [[1.1, 0.3, 0.2], [0.2, 0.1, 1.7], [0.8, 0.3, 1.5], [0.2, 0.1, 0.3]]
        if cylinder
        else [[1.4, 0.1, 0.1], [1.1, 0.9, 0.1], [1.1, 0.9, 1.3], [0.4, 0.12, 0.11]]
    )
    point = (center.detach() + torch.tensor(offsets, dtype=torch.float64)).requires_grad_()
    radius = torch.tensor([0.3] * 4, dtype=torch.float64, requires_grad=True)
    extent = torch.tensor([0.5, 1.2] if cylinder else [0.6, 0.4, 0.8], dtype=torch.float64, requires_grad=True)
    if cylinder:

        def function(p, r, c, e):
            return sphere_cylinder_alpha(p, r, c, e[0], e[1])
    else:
        function = sphere_box_alpha
    assert torch.autograd.gradcheck(function, (point, radius, center, extent), eps=1e-6, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("cylinder", [False, True])
def test_coincident_centres_and_symmetry_axes_are_finite(cylinder):
    point = torch.tensor([[0.0, 0, 0], [0, 0, 0.8], [0.8, 0, 0]], dtype=torch.float64, requires_grad=True)
    if cylinder:
        value = sphere_cylinder_alpha(point, 0.3, torch.zeros(3), 0.5, 1.2)
    else:
        value = sphere_box_alpha(point, 0.3, torch.zeros(3), torch.tensor([0.5, 0.5, 1.2]))
    value.sum().backward()
    assert value[0] == 0
    assert torch.isfinite(value).all() and torch.isfinite(point.grad).all()
    torch.testing.assert_close(point.grad[0], torch.tensor([1.25, 0, 0], dtype=torch.float64))


def test_dynamic_minimum_handles_both_primitives_and_owns_scene_snapshots():
    point = torch.tensor([[[1.2, 0, 0], [3.6, 0, 0], [3.0, 0, 1.5]]], dtype=torch.float64, requires_grad=True)
    positions = torch.tensor([[0.0, 0, 0], [3.0, 0, 0]], dtype=torch.float64)
    sizes = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 2.4]], dtype=torch.float64)
    value = dynamic_min_alpha(
        point, torch.tensor([[0.3]], dtype=torch.float64), positions, sizes, obstacle_chunk_size=1
    )
    torch.testing.assert_close(value, torch.tensor([[1.5, 0.75, 1.0]], dtype=torch.float64))
    # Subsequent scene updates/reset must not change the saved loss graph.
    positions.add_(1000)
    sizes.mul_(0.1)
    value.sum().backward()
    expected = torch.tensor([[[1.25, 0, 0], [1.25, 0, 0], [0, 0, 1 / 1.5]]], dtype=torch.float64)
    torch.testing.assert_close(point.grad, expected)


def test_saved_dynamic_graph_size_does_not_grow_with_obstacle_count():
    saved_counts, gradients = [], []
    for count in (2, 130):
        point = torch.tensor([[1.2, 0.1, 0.1], [-0.9, 0.1, 0.1]], requires_grad=True)
        positions = torch.zeros(count, 3)
        positions[1:, 0] = torch.arange(1, count) * 10.0
        sizes = torch.ones(count, 3)
        saved = []

        def pack(tensor, saved=saved):
            saved.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            value = dynamic_min_alpha(point, 0.3, positions, sizes, obstacle_chunk_size=16)
        value.sum().backward()
        saved_counts.append(sum(saved))
        gradients.append(point.grad)
    assert saved_counts[0] == saved_counts[1]
    torch.testing.assert_close(gradients[0], gradients[1])


def test_no_dynamic_obstacles_leave_ground_gradient_unchanged():
    point = torch.tensor([[0.0, 0.0, 0.6]], requires_grad=True)
    alpha = dynamic_min_alpha(point, 0.3, torch.empty(0, 3), torch.empty(0, 3))
    assert torch.isposinf(alpha).all()
    clearance = torch.minimum(point[:, 2] - 0.31, 0.3 * (alpha - 1))
    clearance.sum().backward()
    torch.testing.assert_close(point.grad, torch.tensor([[0.0, 0.0, 1.0]]))
