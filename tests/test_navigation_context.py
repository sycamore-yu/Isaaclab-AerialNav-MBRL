"""Observation namespaces are explicit; constructing a context draws no noise."""

import pytest
import torch


def test_policy_and_privileged_values_are_selected_independently():
    from isaaclab_navrl.tasks.navigation.context import ObservationContext

    measured, truth = torch.ones(2, 3), torch.full((2, 3), 9.0)
    policy = {"position": measured, "depth_features": torch.ones(2, 1, 12, 16)}
    context = ObservationContext.build(policy=policy, privileged={"position": truth, "episode_id": torch.zeros(2)})
    assert context.policy["position"] is measured
    assert context.privileged["position"] is truth
    assert "episode_id" not in context.policy
    policy["position"] = truth
    assert context.policy["position"] is measured
    with pytest.raises(TypeError):
        context.policy["position"] = truth
    with pytest.raises(TypeError):
        context.privileged["position"] = measured


def test_context_preserves_tensor_graph_without_copying_or_resampling():
    from isaaclab_navrl.tasks.navigation.context import ObservationContext

    value = torch.tensor([2.0], requires_grad=True)
    rng = torch.random.get_rng_state().clone()
    context = ObservationContext.build(policy={"feature": value * 3}, privileged={"state": value})
    context.policy["feature"].sum().backward()
    torch.testing.assert_close(value.grad, torch.tensor([3.0]))
    torch.testing.assert_close(torch.random.get_rng_state(), rng)


def test_inference_features_do_not_read_privileged_physical_state():
    from types import SimpleNamespace

    from isaaclab_navrl.tasks.navigation.context import ObservationContext
    from isaaclab_navrl.zhang.config import ZhangTrainCfg
    from isaaclab_navrl.zhang.inference import ZhangInference

    sensor = {
        "position": torch.tensor([[0.0, 0.0, 2.0]]),
        "velocity": torch.zeros(1, 3),
        "quaternion": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        "target": torch.tensor([[5.0, 0.0, 2.0]]),
        "depth_features": torch.ones(1, 1, 12, 16),
    }
    privileged = dict(sensor) | {"episode_id": torch.tensor([0])}
    context = ObservationContext.build(policy=sensor, privileged=privileged)
    captured = []

    class Actor:
        def __call__(self, depth, state, hidden):
            captured.append((depth.clone(), state.clone()))
            return torch.zeros(1, 6), hidden

    inference = ZhangInference.__new__(ZhangInference)
    parameters = {"max_speed": torch.tensor([[6.0]]), "margin": torch.tensor([0.3]), "thr_est_error": torch.ones(1, 1)}
    inference.env = SimpleNamespace(get_zhang_observation=lambda: context, zhang_parameters=parameters)
    inference.cfg = ZhangTrainCfg(goal_speed_profile="linear")
    inference.hidden, inference.episode_id = torch.zeros(1, 192), torch.tensor([-1])
    inference.model = Actor()
    inference.normalizer = SimpleNamespace(normalize=lambda depth, state, snapshot: (depth, state))
    inference.normalization_snapshot = None
    inference()
    corrupted = {key: torch.full_like(value, float("nan")) for key, value in sensor.items()}
    context = ObservationContext.build(policy=sensor, privileged=privileged | corrupted)
    inference()
    for before, after in zip(captured[0], captured[1], strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        assert torch.isfinite(after).all()
