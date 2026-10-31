"""Independent checks of the explicit static-mesh/dynamic-DCOL hybrid."""

import pytest
import torch

from aerialnav.diffnav.dcol import dynamic_min_alpha, hybrid_collision_gap


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_hybrid_does_not_mix_old_dynamic_distance_into_the_static_branch():
    # The sphere overlaps a real dynamic box.  Its old metric clearance is
    # -0.15 m, whereas the actual DCOL metre-scaled proxy is -0.034615... m.
    # Static geometry is independently supplied as its signed exterior distance.
    # Using the old combined nearest-obstacle result as static geometry would
    # select -0.15 instead and silently suppress the DCOL branch.
    query = torch.tensor([[[1.15, 0.0, 2.0]]], dtype=torch.float64, requires_grad=True)
    static_closest = torch.tensor([[[1.15, 0.0, 0.01]]], dtype=torch.float64, requires_grad=True)
    positions = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64)
    sizes = torch.tensor([[2.0, 2.0, 1.0]], dtype=torch.float64)
    radius = torch.tensor([[0.3]], dtype=torch.float64)

    gap = hybrid_collision_gap(query, radius, (query - static_closest.detach()).norm(dim=-1), positions, sizes)
    expected = torch.tensor([[0.3 * (1.15 / 1.3 - 1)]], dtype=torch.float64)
    torch.testing.assert_close(gap, expected, rtol=1e-12, atol=1e-12)
    assert abs(gap.item() - (-0.15)) > 0.1

    query_gradient, static_gradient = torch.autograd.grad(gap.sum(), (query, static_closest), allow_unused=True)
    torch.testing.assert_close(query_gradient, torch.tensor([[[0.3 / 1.3, 0.0, 0.0]]], dtype=torch.float64))
    assert static_gradient is None


@pytest.mark.parametrize("empty_dynamic", [False, True])
def test_hybrid_static_and_ground_use_independent_metric_clearances(empty_dynamic):
    query = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 0.2]], dtype=torch.float64, requires_grad=True)
    # First query is 0.2 m from an actual static wall.  The second is near
    # ground and far from the independently supplied static surface.
    static_closest = torch.tensor([[0.2, 0.0, 2.0], [5.0, 0.0, 0.2]], dtype=torch.float64, requires_grad=True)
    positions = (
        torch.empty((0, 3), dtype=torch.float64)
        if empty_dynamic
        else torch.tensor([[3.0, 0.0, 2.0]], dtype=torch.float64)
    )
    sizes = torch.ones_like(positions)

    gap = hybrid_collision_gap(
        query, 0.3, (query - static_closest.detach()).norm(dim=-1), positions, sizes, ground_z=0.01
    )
    torch.testing.assert_close(gap, torch.tensor([-0.1, -0.11], dtype=torch.float64))
    query_gradient, static_gradient = torch.autograd.grad(gap.sum(), (query, static_closest), allow_unused=True)
    torch.testing.assert_close(query_gradient, torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64))
    assert static_gradient is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("use_hybrid", [False, True])
def test_dynamic_cpu_and_cuda_values_and_position_gradients_agree(use_hybrid):
    generator = torch.Generator().manual_seed(7221)
    source = torch.randn(8, 10, 3, generator=generator)
    source[..., 2] = source[..., 2].abs() + 0.15
    positions = 2 * torch.randn(17, 3, generator=generator)
    positions[:, 2] = 1.5
    sizes = 0.4 + 0.8 * torch.rand(17, 3, generator=generator)
    sizes[:, 2] = torch.where(torch.arange(17) % 2 == 0, 0.8, 2.4)
    radii = 0.2 + 0.2 * torch.rand(8, 1, generator=generator)
    static_closest = source.clone()
    static_closest[..., 2] = -1.0
    weights = torch.linspace(0.7, 1.3, source.shape[0] * source.shape[1]).reshape(source.shape[:-1])
    results = []

    for device in ("cpu", "cuda:0"):
        query = source.detach().to(device).requires_grad_()
        if use_hybrid:
            value = hybrid_collision_gap(
                query,
                radii.to(device),
                (query - static_closest.to(device)).norm(dim=-1),
                positions.to(device),
                sizes.to(device),
            )
        else:
            value = dynamic_min_alpha(
                query, radii.to(device), positions.to(device), sizes.to(device), obstacle_chunk_size=5
            )
        assert value.shape == source.shape[:-1]
        gradient = torch.autograd.grad((value * weights.to(device)).sum(), query)[0]
        assert torch.isfinite(value).all() and torch.isfinite(gradient).all()
        results.append((value.detach().cpu(), gradient.detach().cpu()))

    torch.testing.assert_close(results[0][0], results[1][0], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
    ],
)
def test_each_query_uses_its_own_collision_substep_snapshot_and_owns_the_winner(device):
    # The first two queries are identical, but their collision substeps had
    # different obstacle poses and therefore different winning primitives.
    query = torch.tensor(
        [[1.3, 0.0, 2.0], [1.3, 0.0, 2.0], [0.0, 0.0, 3.5]], dtype=torch.float64, device=device, requires_grad=True
    )
    positions = torch.tensor(
        [[[0.0, 0.0, 2.0], [10.0, 0.0, 2.0]], [[10.0, 0.0, 2.0], [0.7, 0.0, 2.0]], [[10.0, 0.0, 2.0], [0.0, 0.0, 2.0]]],
        dtype=torch.float64,
        device=device,
    )
    sizes = torch.tensor([[2.0, 2.0, 1.0], [1.0, 1.0, 2.4]], dtype=torch.float64, device=device)
    alpha = dynamic_min_alpha(query, 0.3, positions, sizes, obstacle_chunk_size=1)
    torch.testing.assert_close(alpha, torch.tensor([1.0, 0.75, 1.0], dtype=torch.float64, device=device))

    # A later rollout/reset can reuse and mutate the motion-history buffer.
    positions.add_(1000)
    sizes.mul_(0.1)
    gradient = torch.autograd.grad(alpha.sum(), query)[0]
    expected = torch.tensor(
        [[1 / 1.3, 0.0, 0.0], [1.25, 0.0, 0.0], [0.0, 0.0, 1 / 1.5]], dtype=torch.float64, device=device
    )
    torch.testing.assert_close(gradient, expected)


def test_per_environment_positions_broadcast_over_lookahead_samples():
    query = torch.tensor(
        [[[0.8, 0.0, 2.0], [1.2, 0.0, 2.0]], [[0.8, 0.0, 2.0], [1.0, 0.0, 2.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    positions = torch.tensor(
        [[[[0.0, 0.0, 2.0], [10.0, 0.0, 2.0]]], [[[10.0, 0.0, 2.0], [0.2, 0.0, 2.0]]]], dtype=torch.float64
    )
    sizes = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 2.4]], dtype=torch.float64)
    radius = torch.full((2, 1), 0.3, dtype=torch.float64)
    alpha = dynamic_min_alpha(query, radius, positions, sizes, obstacle_chunk_size=1)
    torch.testing.assert_close(alpha, torch.tensor([[1.0, 1.5], [0.75, 1.0]], dtype=torch.float64))
    expected_gradient = torch.zeros_like(query)
    expected_gradient[..., 0] = 1.25
    torch.testing.assert_close(torch.autograd.grad(alpha.sum(), query)[0], expected_gradient)
    # The public hybrid helper accepts the same per-query snapshot contract.
    static_closest = query.detach().clone()
    static_closest[..., 2] = 0.01
    gap = hybrid_collision_gap(query, radius, (query - static_closest.detach()).norm(dim=-1), positions, sizes)
    torch.testing.assert_close(gap, radius * (alpha.detach() - 1))


def test_per_query_motion_history_does_not_grow_saved_graph_with_obstacle_count():
    saved_counts, gradients = [], []
    for count in (2, 130):
        query = torch.tensor([[1.3, 0.0, 2.0], [1.3, 0.0, 2.0]], requires_grad=True)
        positions = torch.zeros(2, count, 3)
        positions[..., 2] = 2.0
        positions[:, 1:, 0] = torch.arange(1, count) * 10.0
        positions[1, 0, 0] = 0.5
        sizes = torch.tensor([2.0, 2.0, 1.0]).repeat(count, 1)
        saved = []

        def pack(tensor, saved=saved):
            saved.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            alpha = dynamic_min_alpha(query, 0.3, positions, sizes, obstacle_chunk_size=16)
        gradients.append(torch.autograd.grad(alpha.sum(), query)[0])
        saved_counts.append(sum(saved))
    assert saved_counts[0] == saved_counts[1]
    torch.testing.assert_close(gradients[0], gradients[1])


def test_snapshot_batch_cannot_silently_add_an_extra_query_dimension():
    with pytest.raises(ValueError, match="must not expand the query batch"):
        dynamic_min_alpha(torch.zeros(2, 3), 0.3, torch.zeros(2, 2, 1, 3), torch.ones(1, 3))
