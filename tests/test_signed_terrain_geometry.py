"""Actual triangle-query contracts; synthetic geometry, not training evidence."""

import math

import numpy as np
import pytest
import torch

from aerialnav.diffnav.dcol import dynamic_min_alpha, hybrid_collision_gap
from aerialnav.tasks.navigation.geometry import (
    TerrainDistance,
    nearest_obstacle_signed_distance,
    nearest_obstacle_vector,
)


def stepped_heightfield(*, reverse_winding=False):
    """An open, nonconvex height field with unequal adjacent/overwritten tops.

    These are the actual exposed triangles, not a union of old obstacle boxes.
    A top replaced by height 3 has no hidden surface remaining at height 2.
    """
    edges = np.arange(-2.0, 3.0, dtype=np.float32)
    levels = np.zeros((4, 4), dtype=np.float32)
    levels[1, 1], levels[1, 2], levels[2, 1] = 2.0, 3.0, 1.5
    vertices, faces = [], []

    def quad(a, b, c, d):
        offset = len(vertices)
        vertices.extend((a, b, c, d))
        faces.extend(((offset, offset + 1, offset + 2), (offset, offset + 2, offset + 3)))

    for i in range(4):
        for j in range(4):
            x0, x1, y0, y1, h = edges[i], edges[i + 1], edges[j], edges[j + 1], levels[i, j]
            quad((x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h))
            if i < 3 and h != levels[i + 1, j]:
                lo, hi = sorted((h, levels[i + 1, j]))
                quad((x1, y0, lo), (x1, y1, lo), (x1, y1, hi), (x1, y0, hi))
            if j < 3 and h != levels[i, j + 1]:
                lo, hi = sorted((h, levels[i, j + 1]))
                quad((x0, y1, lo), (x1, y1, lo), (x1, y1, hi), (x0, y1, hi))
    faces = np.asarray(faces, dtype=np.int32)
    if reverse_winding:
        faces = faces[:, ::-1].copy()
    return TerrainDistance(np.asarray(vertices, dtype=np.float32), faces, "cpu")


@pytest.fixture
def terrain():
    return stepped_heightfield()


@pytest.mark.parametrize("reverse_winding", [False, True])
def test_signed_top_side_and_height_overlap_escape_direction(reverse_winding):
    terrain = stepped_heightfield(reverse_winding=reverse_winding)
    points = torch.tensor(
        [
            [-1.2, -0.4, 1.0],
            [-0.9, -0.4, 1.0],
            [-0.5, -0.4, 1.0],
            [-0.5, 0.5, 3.2],
            [-0.5, 0.5, 2.9],
            [-0.5, 0.1, 2.5],
            [-0.5, -0.1, 2.5],
        ],
        requires_grad=True,
    )
    result = terrain.signed_query(points)
    expected = points.new_tensor([0.2, -0.1, -0.5, 0.2, -0.1, -0.1, 0.1])
    torch.testing.assert_close(result.distance, expected, rtol=0, atol=2e-6)
    assert result.inside.tolist() == [False, True, True, False, True, True, False]
    assert bool(result.has_surface.all())
    gradient = torch.autograd.grad(result.distance.sum(), points)[0]
    expected_gradient = points.new_tensor(
        [
            [-1, 0, 0],
            [-1, 0, 0],
            [-1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, -1, 0],
            [0, -1, 0],
        ]
    )
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=3e-5)
    # No old height-2 top is left underneath the overwritten height-3 cell.
    overwritten = terrain.signed_query(points.new_tensor([[-0.5, 0.5, 2.1]]))
    assert bool(overwritten.inside[0]) and float(overwritten.distance[0]) < -0.49
    torch.testing.assert_close(overwritten.surface_height, points.new_tensor([3.0]), rtol=0, atol=1e-6)


@pytest.mark.parametrize("reverse_winding", [False, True])
def test_exact_surface_has_finite_outward_gradient_without_trusting_winding(reverse_winding):
    terrain = stepped_heightfield(reverse_winding=reverse_winding)
    points = torch.tensor([[-1.0, -0.5, 0.75], [-0.5, 0.5, 3.0], [1.5, 1.5, 0.0]], requires_grad=True)
    result = terrain.signed_query(points)
    torch.testing.assert_close(result.distance, torch.zeros(3), rtol=0, atol=2e-7)
    gradient = torch.autograd.grad(result.distance.sum(), points)[0]
    assert bool(torch.isfinite(gradient).all())
    torch.testing.assert_close(gradient, points.new_tensor([[-1, 0, 0], [0, 0, 1], [0, 0, 1]]), rtol=0, atol=2e-5)


def test_outside_projection_is_not_filled_and_ground_stays_signed(terrain):
    points = torch.tensor([[2.2, 0.0, -0.4], [2.2, 0.0, 0.4], [1.5, 1.5, -0.4]], requires_grad=True)
    result = terrain.signed_query(points)
    assert result.has_surface.tolist() == [False, False, True]
    assert result.inside.tolist() == [False, False, True]
    assert bool(torch.isnan(result.surface_height[:2]).all())
    assert bool((result.distance[:2] > 0).all())
    combined = nearest_obstacle_signed_distance(points, result.distance, torch.empty(0, 3), torch.empty(0, 3))
    torch.testing.assert_close(combined, points[:, 2] - 0.01, rtol=0, atol=2e-6)
    gradient = torch.autograd.grad(combined.sum(), points)[0]
    torch.testing.assert_close(gradient, points.new_tensor([[0, 0, 1]] * 3), rtol=0, atol=1e-6)
    # A point can be within the XY bounding box but outside actual projected
    # triangles.  A bounding-box or nearest-height rule would classify it wrong.
    triangle = TerrainDistance([[0, 0, 0], [2, 0, 0], [0, 2, 0]], [[0, 1, 2]], "cpu")
    hole = triangle.signed_query(torch.tensor([[1.5, 1.5, -0.3]]))
    assert not bool(hole.has_surface[0]) and not bool(hole.inside[0]) and float(hole.distance[0]) > 0


def test_actual_sloped_triangle_distance_and_derivative():
    terrain = TerrainDistance(
        [[-1, -1, 0.75], [1, -1, 1.25], [1, 1, 1.25], [-1, 1, 0.75]], [[0, 1, 2], [0, 2, 3]], "cpu"
    )
    points = torch.tensor([[0.0, 0.0, 0.9], [0.0, 0.0, 1.1]], requires_grad=True)
    result = terrain.signed_query(points)
    normal = points.new_tensor([-0.25, 0, 1]) / math.sqrt(1.0625)
    torch.testing.assert_close(result.distance, points.new_tensor([-0.1, 0.1]) / math.sqrt(1.0625), rtol=0, atol=2e-6)
    gradient = torch.autograd.grad(result.distance.sum(), points)[0]
    torch.testing.assert_close(gradient, normal.expand_as(points), rtol=0, atol=3e-5)


def test_nonzero_signed_gradient_matches_requeried_finite_difference(terrain):
    points = torch.tensor([[-0.9, -0.4, 1.0], [-0.5, 0.5, 2.9], [-0.5, -0.1, 2.5]], requires_grad=True)
    distance = terrain.signed_distance(points)
    gradient = torch.autograd.grad(distance.sum(), points)[0]
    finite_difference = torch.empty_like(points)
    epsilon = 2e-3  # Warp stores geometry/query positions in float32.
    for axis in range(3):
        offset = torch.zeros_like(points)
        offset[:, axis] = epsilon
        finite_difference[:, axis] = (
            terrain.signed_distance(points.detach() + offset) - terrain.signed_distance(points.detach() - offset)
        ) / (2 * epsilon)
    torch.testing.assert_close(gradient, finite_difference, rtol=0, atol=2e-4)


def test_exterior_legacy_value_and_gradient_are_preserved(terrain):
    points = torch.tensor([[[-1.2, -0.4, 1.0], [-0.5, 0.5, 3.2], [1.5, 1.5, 0.7]]], requires_grad=True)
    signed = terrain.signed_distance(points)
    old = (points - terrain.closest_points(points)).norm(dim=-1)
    torch.testing.assert_close(signed, old, rtol=0, atol=0)
    old_gradient = torch.autograd.grad(old.sum(), points, retain_graph=True)[0]
    new_gradient = torch.autograd.grad(signed.sum(), points)[0]
    torch.testing.assert_close(new_gradient, old_gradient, rtol=0, atol=0)
    empty = terrain.signed_query(torch.empty(0, 2, 3, requires_grad=True))
    assert empty.distance.shape == (0, 2) and empty.closest_point.shape == (0, 2, 3)


def test_static_penetration_loss_descends_toward_side_exit(terrain):
    points = torch.tensor([[-0.9, -0.4, 1.0], [-0.5, -0.4, 1.0]], requires_grad=True)
    distance = terrain.signed_distance(points)
    empty = torch.empty(0, 3)
    combined = nearest_obstacle_signed_distance(points, distance, empty, empty)
    gap = hybrid_collision_gap(points, 0.3, distance, empty, empty)
    torch.testing.assert_close(gap, points.new_tensor([-0.4, -0.8]), rtol=0, atol=2e-6)
    losses = [torch.relu(1 - (combined - 0.3)).square(), torch.nn.functional.softplus(-32 * gap)]
    for loss in losses:
        gradient = torch.autograd.grad(loss.sum(), points, retain_graph=True)[0]
        # Outward is -X here, so descent -gradient must have negative X.
        assert bool((gradient[:, 0] > 0).all())
        torch.testing.assert_close(gradient[:, 1:], torch.zeros_like(gradient[:, 1:]), rtol=0, atol=1e-4)


def test_exterior_dynamic_avoidance_and_dcol_values_are_preserved(terrain):
    points = torch.tensor([[6.35, 0.0, 1.0], [-6.4, 0.0, 1.0]], requires_grad=True)
    positions = torch.tensor([[6.0, 0, 1.0], [-6.0, 0, 1.0]])
    sizes = torch.tensor([[0.4, 0.4, 0.4], [0.6, 0.6, 2.0]])
    static = terrain.signed_distance(points)
    old_distance = nearest_obstacle_vector(points, terrain.closest_points(points), positions, sizes).norm(dim=-1)
    new_distance = nearest_obstacle_signed_distance(points, static, positions, sizes)
    torch.testing.assert_close(new_distance, old_distance, rtol=0, atol=3e-7)
    old_gradient = torch.autograd.grad(old_distance.sum(), points, retain_graph=True)[0]
    new_gradient = torch.autograd.grad(new_distance.sum(), points, retain_graph=True)[0]
    torch.testing.assert_close(new_gradient, old_gradient, rtol=0, atol=0)
    expected = 0.3 * (dynamic_min_alpha(points, 0.3, positions, sizes) - 1)
    actual = hybrid_collision_gap(points, 0.3, static, positions, sizes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(actual.sum(), points, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), points)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_old_closest_point_api_cannot_silently_restore_unsigned_distance():
    points = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)
    empty = torch.empty(0, 3)
    with pytest.raises(ValueError, match="closest-point"):
        hybrid_collision_gap(points, 0.3, points.detach(), empty, empty)
    with pytest.raises(ValueError, match="closest-point"):
        nearest_obstacle_signed_distance(points, points.detach(), empty, empty)


def _quad_terrain(quads):
    vertices, faces = [], []
    for quad in quads:
        offset = len(vertices)
        vertices.extend(quad)
        faces.extend(((offset, offset + 1, offset + 2), (offset, offset + 2, offset + 3)))
    return TerrainDistance(vertices, faces, "cpu")


def _verify_original_mesh_exit(terrain, point, result):
    """Check the returned anchor, not the triangle's centroid classification."""
    assert bool(result.inside.all()) and bool(result.exit_valid.all())
    assert bool((result.distance < 0).all())
    anchor = result.closest_point
    on_mesh = (anchor - terrain.closest_points(anchor)).norm(dim=-1)
    assert bool((on_mesh < 2e-6).all())
    direction = (anchor - point.detach()) / (anchor - point.detach()).norm(dim=-1, keepdim=True)
    before = terrain.signed_query(anchor - terrain._normal_probe * direction)
    after = terrain.signed_query(anchor + terrain._normal_probe * direction)
    assert bool(before.inside.all()) and not bool(after.inside.any())
    gradient = torch.autograd.grad(result.distance.sum(), point)[0]
    torch.testing.assert_close(gradient, direction, rtol=0, atol=2e-5)
    return direction


def test_partially_covered_triangle_keeps_exposed_piece_and_rejects_hidden_exit():
    # The broad z=1 roof is deliberately retained underneath the z=2 cap. Its
    # centroid is hidden, but points outside the cap remain valid real exits.
    terrain = _quad_terrain(
        [
            [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)],
            [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
            [(-0.5, -0.5, 2), (0.5, -0.5, 2), (0.5, 0.5, 2), (-0.5, 0.5, 2)],
            [(-0.5, -0.5, 1), (-0.5, 0.5, 1), (-0.5, 0.5, 2), (-0.5, -0.5, 2)],
            [(0.5, -0.5, 1), (0.5, 0.5, 1), (0.5, 0.5, 2), (0.5, -0.5, 2)],
            [(-0.5, -0.5, 1), (0.5, -0.5, 1), (0.5, -0.5, 2), (-0.5, -0.5, 2)],
            [(-0.5, 0.5, 1), (0.5, 0.5, 1), (0.5, 0.5, 2), (-0.5, 0.5, 2)],
        ]
    )
    exposed = torch.tensor([[-0.8, 0.0, 0.95]], requires_grad=True)
    exposed_result = terrain.signed_query(exposed)
    torch.testing.assert_close(exposed_result.distance, torch.tensor([-0.05]), rtol=0, atol=2e-6)
    assert not bool(exposed_result.used_exit_search.any())
    direction = _verify_original_mesh_exit(terrain, exposed, exposed_result)
    torch.testing.assert_close(direction, torch.tensor([[0, 0, 1.0]]), rtol=0, atol=2e-5)

    hidden = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.01]], requires_grad=True)
    result = terrain.signed_query(hidden)
    assert bool(result.used_exit_search.all())
    # At the internal sheet, old norm gave zero and a false penetration exit.
    torch.testing.assert_close(result.mesh_closest_point[:, 2], torch.ones(2), rtol=0, atol=1e-6)
    # At z=1 the horizontal candidate ends exactly on the exposed lower roof.
    # It has zero after-probe clearance and must not be accepted as a robust
    # exit. A farther original-mesh candidate is valid; the proxy does not
    # promise the mathematically closest boundary point. At z=1.01 the ordinary
    # horizontal exit is well separated and must retain its exact distance.
    assert float(result.distance[0].detach()) < -0.5
    torch.testing.assert_close(result.distance[1:], torch.tensor([-0.5]), rtol=0, atol=2e-5)
    direction = _verify_original_mesh_exit(terrain, hidden, result)
    assert float(direction[0, :2].norm()) > 0.8
    assert float(direction[1, :2].norm()) > 0.99
    before = terrain.signed_query(result.closest_point - terrain._normal_probe * direction)
    after = terrain.signed_query(result.closest_point + terrain._normal_probe * direction)
    assert bool(
        (
            before.surface_height - (result.closest_point - terrain._normal_probe * direction)[:, 2]
            > terrain.exit_height_guard_m
        ).all()
    )
    assert bool(
        (
            (result.closest_point + terrain._normal_probe * direction)[:, 2] - after.surface_height
            > terrain.exit_height_guard_m
        ).all()
    )


def test_zero_thickness_fin_keeps_original_exterior_distance_and_gradient():
    terrain = _quad_terrain(
        [
            [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)],
            [(0, -0.5, 1), (0, 0.5, 1), (0, 0.5, 2), (0, -0.5, 2)],
        ]
    )
    point = torch.tensor([[-0.05, 0, 1.5], [0.05, 0, 1.5]], requires_grad=True)
    result = terrain.signed_query(point)
    assert not bool(result.inside.any()) and not bool(result.used_exit_search.any())
    legacy = (point - terrain.closest_points(point)).norm(dim=-1)
    torch.testing.assert_close(result.distance, legacy, rtol=0, atol=0)
    new_gradient = torch.autograd.grad(result.distance.sum(), point, retain_graph=True)[0]
    old_gradient = torch.autograd.grad(legacy.sum(), point)[0]
    torch.testing.assert_close(new_gradient, old_gradient, rtol=0, atol=0)
    torch.testing.assert_close(new_gradient, torch.tensor([[-1, 0, 0], [1, 0, 0.0]]), rtol=0, atol=1e-6)


def test_exact_installed_slope_converter_crossing_has_no_internal_zero_exit():
    # Exact installed converter output for heights
    # [[0,0,0,0],[0,1,1,0],[0,3,2,0],[0,0,0,0]], scales=1,
    # slope_threshold=.75. Faces 0 and 16 intersect strictly in their interiors
    # at (1.5,1.75,.5). This fixture preserves those actual crossing triangles.
    vertices = [
        [1, 1, 0],
        [1, 2, 0],
        [1, 2, 0],
        [0, 3, 0],
        [2, 1, 0],
        [2, 2, 1],
        [2, 2, 1],
        [1, 2, 0],
        [2, 1, 0],
        [2, 1, 3],
        [2, 1, 2],
        [1, 2, 0],
        [3, 0, 0],
        [2, 1, 0],
        [2, 1, 0],
        [2, 2, 0],
    ]
    faces = [
        [0, 5, 1],
        [0, 4, 5],
        [1, 6, 2],
        [1, 5, 6],
        [2, 7, 3],
        [2, 6, 7],
        [4, 9, 5],
        [4, 8, 9],
        [5, 10, 6],
        [5, 9, 10],
        [6, 11, 7],
        [6, 10, 11],
        [8, 13, 9],
        [8, 12, 13],
        [9, 14, 10],
        [9, 13, 14],
        [10, 15, 11],
        [10, 14, 15],
    ]
    terrain = TerrainDistance(vertices, faces, "cpu")
    point = torch.tensor([[1.5, 1.75, 0.5]], requires_grad=True)
    result = terrain.signed_query(point)
    assert bool(result.used_exit_search[0]) and float(result.distance[0]) < -0.01
    torch.testing.assert_close(result.mesh_closest_point, point.detach(), rtol=0, atol=1e-6)
    _verify_original_mesh_exit(terrain, point, result)
