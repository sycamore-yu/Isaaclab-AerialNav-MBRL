"""Running statistics must not change an in-flight rollout or its gradients."""

import copy
import io

import torch
from torch.utils.checkpoint import checkpoint

from aerialnav.diffnav.normalization import PolicyObservationNormalizer


def _inputs():
    depth = torch.arange(24, dtype=torch.float64).reshape(4, 1, 2, 3) / 7
    state = torch.arange(40, dtype=torch.float64).reshape(4, 10) / 9
    state = state * torch.linspace(0.3, 2, 10) + torch.arange(10)
    return depth, state


def test_population_moments_merge_and_snapshot_survives_later_observations():
    depth, state = _inputs()
    normalizer = PolicyObservationNormalizer(enabled=True)
    first = normalizer.begin_rollout()
    actual_depth, actual_state = normalizer.normalize(depth, state, first)
    torch.testing.assert_close(actual_depth, depth, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, state, rtol=0, atol=0)
    large_floor = PolicyObservationNormalizer(enabled=True, variance_floor=4.0)
    for actual, expected in zip(
        large_floor.normalize(depth, state, large_floor.begin_rollout()), (depth, state), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for start, end in ((0, 1), (1, 3), (3, 4)):
        normalizer.observe(depth[start:end], state[start:end])
    torch.testing.assert_close(normalizer.depth_moments.mean, depth.mean())
    torch.testing.assert_close(normalizer.state_moments.mean, state.mean(0))
    torch.testing.assert_close(normalizer.depth_moments.m2 / normalizer.depth_moments.count, depth.var(correction=0))
    torch.testing.assert_close(normalizer.state_moments.m2 / normalizer.state_moments.count, state.var(0, correction=0))
    assert int(normalizer.depth_moments.count) == depth.numel()
    assert int(normalizer.state_moments.count) == state.shape[0]
    fixed = normalizer.begin_rollout()
    before = normalizer.normalize(depth, state, fixed)
    normalizer.observe(depth + 100, state - 50)
    after = normalizer.normalize(depth, state, fixed)
    for actual, expected in zip(after, before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(not buffer.requires_grad and buffer.grad_fn is None for buffer in normalizer.buffers())


def test_checkpoint_recomputation_keeps_input_gradients_and_does_not_collect_again():
    data_depth, data_state = _inputs()
    normalizer = PolicyObservationNormalizer(enabled=True)
    normalizer.observe(data_depth, data_state)
    snapshot = normalizer.begin_rollout()
    depth = data_depth.float().clone().requires_grad_()
    state = data_state.float().clone().requires_grad_()
    reference_depth = depth.detach().clone().requires_grad_()
    reference_state = state.detach().clone().requires_grad_()

    def objective(first, second):
        normalized_depth, normalized_state = normalizer.normalize(first, second, snapshot)
        return normalized_depth.sin().square().sum() + normalized_state.tanh().square().sum()

    loss = checkpoint(objective, depth, state, use_reentrant=False)
    expected = objective(reference_depth, reference_state)
    # A later policy input changes live moments before checkpoint recomputation.
    normalizer.observe(data_depth + 30, data_state - 20)
    count_before_backward = normalizer.state_moments.count.clone()
    loss.backward()
    expected.backward()
    torch.testing.assert_close(depth.grad, reference_depth.grad, rtol=0, atol=0)
    torch.testing.assert_close(state.grad, reference_state.grad, rtol=0, atol=0)
    assert bool(depth.grad.abs().max() > 0) and bool(state.grad.abs().max() > 0)
    torch.testing.assert_close(normalizer.state_moments.count, count_before_backward, rtol=0, atol=0)
    assert all(buffer.grad is None for buffer in normalizer.buffers())


def test_disabled_is_identity_and_inference_restores_all_normalization_state():
    depth, state = _inputs()
    disabled = PolicyObservationNormalizer()
    disabled.observe(depth, state)
    actual_depth, actual_state = disabled.normalize(depth, state, disabled.begin_rollout())
    assert actual_depth is depth and actual_state is state
    assert int(disabled.depth_moments.count) == int(disabled.state_moments.count) == 0

    trained = PolicyObservationNormalizer(enabled=True, variance_floor=0.002)
    trained.observe(depth, state)
    payload = io.BytesIO()
    torch.save(trained.state_dict(), payload)
    payload.seek(0)
    inference = PolicyObservationNormalizer().eval()
    inference.load_state_dict(torch.load(payload, weights_only=True))
    before = copy.deepcopy(inference.state_dict())
    inference.observe(depth + 500, state + 500)  # eval mode forbids collection.
    assert inference.enabled and inference.variance_floor == trained.variance_floor
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(inference.state_dict()[key], value, rtol=0, atol=0)
        else:
            assert inference.state_dict()[key] == value
    for actual, expected in zip(
        inference.normalize(depth, state, inference.begin_rollout()),
        trained.normalize(depth, state, trained.begin_rollout()),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_constant_features_have_bounded_gain_and_nonfinite_batches_do_not_poison_statistics():
    normalizer = PolicyObservationNormalizer(enabled=True, variance_floor=0.01)
    depth = torch.ones(3, 1, 2, 2)
    state = torch.full((3, 10), 0.3)
    normalizer.observe(depth, state)
    snapshot = normalizer.begin_rollout()
    state.requires_grad_()
    normalized_depth, normalized_state = normalizer.normalize(depth, state, snapshot)
    normalized_state.sum().backward()
    torch.testing.assert_close(state.grad, torch.full_like(state, 10.0))
    torch.testing.assert_close(normalized_depth, torch.zeros_like(depth))
    before = normalizer.state_moments.mean.clone()
    invalid = state.detach().clone()
    invalid[0, 0] = float("nan")
    normalizer.observe(depth, invalid)
    torch.testing.assert_close(normalizer.state_moments.mean, before, rtol=0, atol=0)
    assert int(normalizer.state_moments.count) == 3
    assert int(normalizer.state_moments.skipped_batches) == 1
    # Collection filtering must not hide invalid policy inputs.
    assert torch.isnan(normalizer.normalize(depth, invalid, snapshot)[1][0, 0])
