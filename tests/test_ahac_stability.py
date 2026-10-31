"""Counterexamples for rare-risk dilution, horizon gating and critic work growth."""

import math

import pytest
import torch

from aerialnav.diffnav.actor_critic import (
    ActorCriticSettings,
    AdaptiveHorizonController,
    CostActorCritic,
    critic_split_indices,
)
from aerialnav.diffnav.dcol import height_bounded_gap, hybrid_collision_gap, metric_navigation_gap


def test_single_dangerous_trajectory_is_not_diluted_by_256_environments():
    c = AdaptiveHorizonController()
    risk = torch.zeros(32, 256)
    risk[-1, 0] = 5.713
    risk.requires_grad_()
    assert float(risk.detach().mean()) < c.settings.risk_budget
    penalty = c.penalty(risk)
    assert float(penalty.detach()) == pytest.approx(5.613)
    penalty.backward()
    assert risk.grad[-1, 0] > 0 and risk.grad.count_nonzero() == 1
    result = c.update(risk)
    assert c.horizon < 32 and result["ahac_constraint_risk"] == pytest.approx(5.713)
    assert c.dual > 1 and result["ahac_unsafe"] == 1


@pytest.mark.parametrize("signal", [{"terminal_fraction": 0.1}, {"gradient_ratio": 3.0}, {"model_velocity_error": 1.0}])
def test_horizon_shrinks_on_failure_or_gradient_or_model_instability(signal):
    c = AdaptiveHorizonController()
    result = c.update(torch.zeros(32, 256), **signal)
    assert c.horizon < 32 and result["ahac_unsafe"] == 1


def test_growth_requires_three_consecutive_stable_updates_and_restores_streak():
    c = AdaptiveHorizonController()
    for _ in range(2):
        c.update(torch.zeros(32, 256))
        assert c.horizon == 32
    restored = AdaptiveHorizonController()
    restored.load_state_dict(c.state_dict())
    assert c.update(torch.zeros(32, 256)) == restored.update(torch.zeros(32, 256))
    assert c.horizon == 36
    c.update(torch.zeros(36, 256), gradient_ratio=1.5)
    assert c.horizon == 36 and c.stable_updates == 0


def test_metric_risk_uses_actual_dynamic_distance_and_exact_height_bounds():
    query = torch.tensor([[1.5, 0.0, 2.0], [11.5, 0.0, 2.0]], dtype=torch.float64, requires_grad=True)
    centers = torch.tensor([[[0.0, 0.0, 2.0]], [[10.0, 0.0, 2.0]]], dtype=query.dtype)
    sizes = torch.tensor([[2.0, 2.0, 1.0]], dtype=query.dtype)
    gap = metric_navigation_gap(query, query.new_full((2,), 10.0), centers, sizes)
    torch.testing.assert_close(gap, query.new_full((2,), 0.2))
    proxy = hybrid_collision_gap(query, 0.3, query.new_full((2,), 10.0), centers, sizes)
    assert not torch.allclose(proxy, gap)
    gap.sum().backward()
    torch.testing.assert_close(query.grad, query.new_tensor([[1, 0, 0], [1, 0, 0]]))
    high = torch.tensor([[0.0, 0.0, 4.1]], requires_grad=True)
    upper = metric_navigation_gap(high, high.new_full((1,), 10.0), torch.empty(0, 3), torch.empty(0, 3))
    torch.testing.assert_close(upper, torch.tensor([-0.1]))
    upper.sum().backward()
    torch.testing.assert_close(high.grad, torch.tensor([[0.0, 0.0, -1.0]]))
    low = torch.tensor([[0.0, 0.0, 0.1]], requires_grad=True)
    lower = height_bounded_gap(low, torch.tensor([10.0]))
    torch.testing.assert_close(lower, torch.tensor([-0.1]))
    lower.sum().backward()
    torch.testing.assert_close(low.grad, torch.tensor([[0.0, 0.0, 1.0]]))


@pytest.mark.parametrize("horizon", [16, 288])
def test_critic_optimizer_budget_is_independent_of_horizon(horizon):
    cfg = ActorCriticSettings(algorithm="ahac", hidden_sizes=(8,), critic_convergence_window=64)
    support = CostActorCritic(2, cfg)
    shape = (horizon, 256)
    metrics = support.fit(
        torch.zeros(*shape, 2), torch.ones(shape), torch.zeros(shape), torch.ones(shape, dtype=torch.bool)
    )
    assert metrics["critic_optimizer_steps"] == 64
    assert metrics["critic_minibatches"] == 4
    assert metrics["critic_iterations"] == 16
    assert metrics["critic_training_samples"] == horizon * 230
    assert metrics["critic_validation_samples"] == horizon * 26
    assert math.isfinite(metrics["critic_validation_td_mse"])


def test_validation_environments_are_disjoint_and_never_affect_optimizer_updates():
    cfg = ActorCriticSettings(algorithm="shac", hidden_sizes=(8,), critic_iterations=2, seed=19)
    rng = torch.Generator().manual_seed(cfg.seed + 1)
    train, holdout = critic_split_indices((8, 16), 0.1, "cpu", rng)
    assert set(train.tolist()).isdisjoint(holdout.tolist())
    assert set((train % 16).tolist()).isdisjoint((holdout % 16).tolist())
    features = torch.zeros(8, 16, 2)
    first, second = CostActorCritic(2, cfg), CostActorCritic(2, cfg)
    costs = torch.ones(8, 16)
    changed = costs.clone()
    changed.view(-1)[holdout] = 100
    one = first.fit(features, costs, torch.zeros_like(costs), torch.ones_like(costs, dtype=torch.bool))
    two = second.fit(features, changed, torch.zeros_like(costs), torch.ones_like(costs, dtype=torch.bool))
    for a, b in zip(first.critics.parameters(), second.critics.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert one["critic_validation_td_mse"] != two["critic_validation_td_mse"]
