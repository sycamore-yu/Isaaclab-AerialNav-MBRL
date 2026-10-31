"""Precision regression for a measured Isaac Lab vertical terrain face."""

import pytest
import torch

from aerialnav.tasks.navigation.geometry import TerrainDistance


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_actual_large_coordinate_thin_wall_has_no_tangential_projection_drift(device):
    """Use the exact failing plane, with analytic horizontal distance/gradient."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; the CPU precision contract still runs")
    x0, x1 = -15.899999618530273, -15.800000190734863
    y0, y1 = -15.100000381469727, -14.100000381469727
    height = 5.900000095367432
    vertices = [(x0, y0, 0), (x1, y0, 0), (x1, y0, height), (x0, y0, height)]
    faces = [(0, 1, 3), (1, 2, 3)]
    quads = [
        [(x0 - 1, y0 - 1, 0), (x1 + 1, y0 - 1, 0), (x1 + 1, y1 + 1, 0), (x0 - 1, y1 + 1, 0)],
        [(x0, y0, height), (x1, y0, height), (x1, y1, height), (x0, y1, height)],
        [(x0, y0, 0), (x0, y1, 0), (x0, y1, height), (x0, y0, height)],
        [(x1, y0, 0), (x1, y1, 0), (x1, y1, height), (x1, y0, height)],
        [(x0, y1, 0), (x1, y1, 0), (x1, y1, height), (x0, y1, height)],
    ]
    for quad in quads:
        offset = len(vertices)
        vertices.extend(quad)
        faces.extend(((offset, offset + 1, offset + 2), (offset, offset + 2, offset + 3)))
    terrain = TerrainDistance(vertices, faces, device)
    # Exact real failure, its exterior counterpart, and an interior face point.
    point = torch.tensor(
        [
            [-15.833333015441895, -15.09990119934082, 3.933333396911621],
            [-15.833333015441895, -15.100099563598633, 3.933333396911621],
            [-15.833333015441895, y0, 3.933333396911621],
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    expected_closest = point.detach().clone()
    expected_closest[:, 1] = y0
    closest = terrain.closest_points(point)
    torch.testing.assert_close(closest, expected_closest, rtol=0, atol=0)
    torch.testing.assert_close(terrain.closest_points(closest), expected_closest, rtol=0, atol=0)
    result = terrain.signed_query(point)
    torch.testing.assert_close(result.closest_point, expected_closest, rtol=0, atol=0)
    torch.testing.assert_close(result.distance, point.new_tensor(y0) - point[:, 1], rtol=0, atol=0)
    assert bool(result.inside[0]) and not bool(result.inside[1])
    assert bool(result.exit_valid[0]) and not bool(result.used_exit_search.any())
    gradient = torch.autograd.grad(result.distance.sum(), point)[0]
    torch.testing.assert_close(gradient, point.new_tensor([[0.0, -1.0, 0.0]]).expand_as(point), rtol=0, atol=0)
    assert float(gradient[:, 2].abs().max()) == 0.0
