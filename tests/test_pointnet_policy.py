"""Point-set symmetry, missing returns, and recurrent gradient contracts."""

import torch

from aerialnav.pointnet.model import PointNetPolicy


def test_pointnet_is_order_invariant_and_ignores_invalid_returns():
    torch.manual_seed(5)
    policy = PointNetPolicy(motion=True)
    points = torch.randn(2, 5, 31)
    points[:, 3] = 0
    points[:, 4] = 1
    points[:, 4, -3:] = 0
    state = torch.randn(2, 10)
    expected, _ = policy(points, state)
    changed = points.clone()
    changed[:, :4, -3:] = float("nan")
    actual, _ = policy(changed[:, :, torch.randperm(31)], state)
    torch.testing.assert_close(expected, actual)
    empty = points.clone()
    empty[:, -1] = 0
    assert torch.isfinite(policy(empty, state)[0]).all()


def test_point_policy_backpropagates_through_history_and_motion_feature():
    policy = PointNetPolicy(motion=True)
    points = torch.rand(2, 5, 17)
    points[:, -1] = 1
    points.requires_grad_()
    state = torch.randn(2, 10, requires_grad=True)
    hidden = torch.zeros(2, 192)
    for _ in range(3):
        command, hidden = policy.forward_checkpointed(points, state, hidden)
    assert command.shape == (2, 3)
    command.square().sum().backward()
    assert points.grad[:, 3].abs().sum() > 0
    assert state.grad.abs().sum() > 0
    assert torch.isfinite(points.grad).all()


def test_pointnet_state_and_acceleration_use_full_body_attitude():
    from aerialnav.diffnav.model import decode_action, policy_state

    q = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]])
    features, body, _, _ = policy_state(
        torch.zeros(1, 3),
        torch.tensor([[0.0, 0.0, 1.0]]),
        q,
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.ones(1, 1),
        torch.tensor([0.3]),
        body_frame=True,
    )
    torch.testing.assert_close(features[:, :3], torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6, rtol=0)
    command, prediction = decode_action(torch.tensor([[0.0, 1.0, 0.0]]), body, torch.ones(1, 1))
    torch.testing.assert_close(command, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6, rtol=0)
    assert torch.count_nonzero(prediction) == 0


def test_acceleration_head_supports_training_diagnostics_without_velocity_loss():
    from aerialnav.diffnav.diagnostics import GradientDiagnostic
    from aerialnav.diffnav.optimization import OptimizerStepMonitor

    model = PointNetPolicy()
    points = torch.rand(1, 4, 9)
    points[:, -1] = 1
    output, _ = model(points, torch.zeros(1, 10))
    parts = {"acceleration": output.square().mean(), "velocity_prediction": torch.zeros(())}
    diagnostic = GradientDiagnostic(model, {}, None, components=parts)
    assert torch.count_nonzero(diagnostic.component_gradient("velocity_prediction")) == 0
    monitor = OptimizerStepMonitor(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    monitor.before_step()
    parts["acceleration"].backward()
    optimizer.step()
    measured = monitor.after_step()
    assert measured["Optim/head/az/update_norm"] > 0
    assert not any("/vz/" in key for key in measured)


def test_environment_chunking_preserves_recurrent_outputs_and_gradients():
    import copy

    torch.manual_seed(7)
    full = PointNetPolicy()
    chunked = copy.deepcopy(full)
    count = full.checkpoint_batch_size + 3
    points = torch.randn(count, 4, 13)
    points[:, -1] = 1
    state = torch.randn(count, 10)
    full_hidden = torch.randn(count, 192)
    chunked_hidden = full_hidden.clone()
    full_loss = chunked_loss = 0
    for _ in range(3):
        expected, full_hidden = full(points, state, full_hidden)
        actual, chunked_hidden = chunked.forward_checkpointed(points, state, chunked_hidden)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=2e-5)
        torch.testing.assert_close(chunked_hidden, full_hidden, atol=2e-6, rtol=2e-5)
        full_loss = full_loss + expected.square().mean()
        chunked_loss = chunked_loss + actual.square().mean()
    full_loss.backward()
    chunked_loss.backward()
    for expected, actual in zip(full.parameters(), chunked.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-8, rtol=2e-4)


def test_pointnet_rejects_an_unimplemented_recurrent_backend():
    import pytest

    with pytest.raises(ValueError, match="gru"):
        PointNetPolicy(recurrent_model="sru_lstm_gated")
