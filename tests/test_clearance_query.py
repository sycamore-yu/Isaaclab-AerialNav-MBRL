"""Shared clearance applies query-time extrapolation and body radius exactly once."""

import pytest
import torch

from isaaclab_navrl.tasks.navigation.geometry import nearest_obstacle_signed_distance


def test_query_matches_existing_time_aligned_values_and_gradients():
    from isaaclab_navrl.tasks.navigation.geometry import query_clearance

    points = torch.tensor([[[0.0, 0.0, 2.0], [0.5, 0.0, 2.0]], [[0.0, 1.0, 2.0], [0.5, 1.0, 2.0]]], requires_grad=True)
    positions = torch.tensor([[2.0, 0.0, 2.0], [2.0, 1.0, 2.0]])
    velocities = torch.tensor([[-1.0, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    sizes = torch.full((2, 3), 0.4)
    times = torch.tensor([[0.0, 0.25]])
    radius = torch.tensor([[0.3], [0.4]])
    static = points[..., 0] * 0 + 100.0
    aligned = positions + velocities * times[..., None, None]
    old, speed = nearest_obstacle_signed_distance(points, static, aligned, sizes, dynamic_velocity=velocities)
    actual, actual_speed = query_clearance(
        points,
        static,
        positions,
        sizes,
        body_radius=radius,
        query_time=times,
        dynamic_velocity=velocities,
        return_velocity=True,
    )
    torch.testing.assert_close(actual, old - radius, rtol=0, atol=0)
    torch.testing.assert_close(actual_speed, speed, rtol=0, atol=0)
    old_grad = torch.autograd.grad((old - radius).sum(), points, retain_graph=True)[0]
    new_grad = torch.autograd.grad(actual.sum(), points)[0]
    torch.testing.assert_close(new_grad, old_grad, rtol=0, atol=0)


def test_already_aligned_substep_snapshot_is_not_advanced_again():
    from isaaclab_navrl.tasks.navigation.geometry import query_clearance

    points = torch.tensor([[0.0, 0.0, 2.0]])
    sizes = torch.tensor([[0.4, 0.4, 0.4]])
    distance = query_clearance(points, torch.tensor([100.0]), torch.tensor([[[1.0, 0.0, 2.0]]]), sizes, body_radius=0.3)
    torch.testing.assert_close(distance, torch.tensor([0.5]))


def test_no_dynamic_objects_ground_and_surface_radius_contract():
    from isaaclab_navrl.tasks.navigation.geometry import query_clearance

    points = torch.tensor([[0.0, 0.0, 0.21]], requires_grad=True)
    empty = torch.empty(0, 3)
    distance = query_clearance(points, torch.tensor([100.0]), empty, empty, body_radius=0.3)
    torch.testing.assert_close(distance, torch.tensor([-0.1]))
    distance.sum().backward()
    torch.testing.assert_close(points.grad, torch.tensor([[0.0, 0.0, 1.0]]))


def test_nonzero_time_without_velocity_is_rejected():
    from isaaclab_navrl.tasks.navigation.geometry import query_clearance

    with pytest.raises(ValueError, match="velocity"):
        query_clearance(torch.zeros(2, 3), torch.ones(2), torch.empty(0, 3), torch.empty(0, 3), query_time=0.1)


def test_shared_obstacle_snapshots_are_not_materialized_per_drone():
    from isaaclab_navrl.tasks.navigation.geometry import query_clearance

    shapes = []

    def backend(query, positions, sizes, *, return_index):
        shapes.append(positions.shape)
        return query[..., 0] * 0 + 10, None

    query = torch.zeros(64, 10, 3)
    positions = torch.ones(8, 3)
    sizes = torch.ones_like(positions)
    velocity = torch.zeros_like(positions)
    for time in (0.0, torch.linspace(0.0, 0.1, 10)[None]):
        query_clearance(
            query,
            torch.ones(64, 10),
            positions,
            sizes,
            query_time=time,
            dynamic_velocity=velocity,
            dynamic_backend=backend,
        )
    assert shapes == [(8, 3), (1, 10, 8, 3)]
