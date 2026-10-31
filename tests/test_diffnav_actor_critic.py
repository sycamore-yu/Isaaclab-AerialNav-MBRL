"""Analytic return/gradient and independent value-learning contracts."""

import copy

import pytest
import torch
from torch import nn

from aerialnav.diffnav.actor_critic import (
    ActorCriticSettings,
    AdaptiveHorizonController,
    AHACSettings,
    CostActorCritic,
    CostValueNetwork,
    short_horizon_cost_objective,
    td_lambda_cost_targets,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_td_lambda_matches_analytic_endpoints_and_mixture():
    costs = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    following = torch.tensor([[10.0], [20.0], [30.0]], requires_grad=True)
    cuts = torch.zeros_like(costs, dtype=torch.bool)
    one_step = td_lambda_cost_targets(costs, following, cuts, discount=0.5, td_lambda=0)
    monte_carlo = td_lambda_cost_targets(costs, following, cuts, discount=0.5, td_lambda=1)
    mixed = td_lambda_cost_targets(costs, following, cuts, discount=0.5, td_lambda=0.5)
    torch.testing.assert_close(one_step[:, 0], torch.tensor([6.0, 12.0, 18.0]))
    torch.testing.assert_close(monte_carlo[:, 0], torch.tensor([6.5, 11.0, 18.0]))
    torch.testing.assert_close(mixed[:, 0], torch.tensor([6.375, 11.5, 18.0]))
    assert not mixed.requires_grad
    assert costs.grad is None and following.grad is None


def test_reset_cuts_lambda_trace_but_keeps_actual_reset_value():
    costs = torch.tensor([[1.0], [2.0], [1000.0]])
    following = torch.tensor([[10.0], [7.0], [30.0]])
    cuts = torch.tensor([[False], [True], [False]])
    actual = td_lambda_cost_targets(costs, following, cuts, discount=0.5, td_lambda=1)
    torch.testing.assert_close(actual[:, 0], torch.tensor([3.75, 5.5, 1015.0]))
    changed = costs.clone()
    changed[-1] = 9000
    other = td_lambda_cost_targets(changed, following, cuts, discount=0.5, td_lambda=1)
    torch.testing.assert_close(actual[:2], other[:2])
    assert actual[1, 0] != costs[1, 0]  # A reset is not an absorbing zero-value terminal.


def test_actor_bootstrap_keeps_live_endpoint_gradient_and_discount():
    action = torch.tensor(2.0, requires_grad=True)
    costs = torch.stack((action.square(), (2 * action).square()))[:, None]
    endpoint = 3 * action
    values = torch.stack((action * 0, endpoint.square()))[:, None]
    cuts = torch.zeros_like(costs, dtype=torch.bool)
    result = short_horizon_cost_objective(costs, values, cuts, discount=0.5)
    # (a^2 + .5*(2a)^2 + .25*(3a)^2) / 2 = 2.625 a^2.
    torch.testing.assert_close(result["objective"], torch.tensor(10.5))
    torch.testing.assert_close(result["bootstrap_reset"], torch.tensor(0.0))
    torch.testing.assert_close(result["bootstrap_live_endpoint"], torch.tensor(4.5))
    assert not result["bootstrap_reset"].requires_grad and not result["bootstrap_live_endpoint"].requires_grad
    result["objective"].backward()
    torch.testing.assert_close(action.grad, torch.tensor(10.5))


def test_actor_reset_bootstraps_detached_state_and_restarts_discount():
    first = torch.tensor(2.0, requires_grad=True)
    second = torch.tensor(3.0, requires_grad=True)
    reset_state = torch.tensor(7.0, requires_grad=True)
    costs = torch.stack((first.square(), second.square()))[:, None]
    following = torch.stack((reset_state.detach(), second.square()))[:, None]
    cuts = torch.tensor([[True], [False]])
    result = short_horizon_cost_objective(costs, following, cuts, discount=0.5)
    # Two one-step segments: (4 + .5*7 + 9 + .5*9) / 2.
    torch.testing.assert_close(result["objective"], torch.tensor(10.5))
    torch.testing.assert_close(result["bootstrap_reset"], torch.tensor(1.75))
    torch.testing.assert_close(result["bootstrap_live_endpoint"], torch.tensor(2.25))
    torch.testing.assert_close(result["bootstrap_reset"] + result["bootstrap_live_endpoint"], result["bootstrap"])
    assert not result["bootstrap_reset"].requires_grad and not result["bootstrap_live_endpoint"].requires_grad
    result["objective"].backward()
    torch.testing.assert_close(first.grad, torch.tensor(2.0))
    torch.testing.assert_close(second.grad, torch.tensor(4.5))
    assert reset_state.grad is None


def test_bootstrap_diagnostics_classify_resets_at_the_final_step():
    costs = torch.ones(2, 2, requires_grad=True)
    values = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    cuts = torch.tensor([[False, True], [True, True]])
    result = short_horizon_cost_objective(costs, values, cuts, discount=0.5)
    torch.testing.assert_close(result["bootstrap_live_endpoint"], torch.tensor(0.0))
    torch.testing.assert_close(result["bootstrap_reset"], result["bootstrap"])
    result["objective"].backward()
    torch.testing.assert_close(costs.grad, torch.tensor([[0.25, 0.25], [0.125, 0.25]]))


def test_frozen_value_preserves_position_and_memory_gradients_without_parameter_grads():
    support = CostActorCritic(2, ActorCriticSettings(hidden_sizes=(4,)))
    linear = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[2.0, 3.0]]))
    support.frozen_critics = nn.ModuleList([nn.Sequential(linear, nn.Flatten(0))]).requires_grad_(False)
    position = torch.tensor([[4.0]], requires_grad=True)
    hidden = torch.tensor([[5.0]], requires_grad=True)
    support.actor_value(torch.cat((position, hidden), dim=-1)).sum().backward()
    torch.testing.assert_close(position.grad, torch.tensor([[2.0]]))
    torch.testing.assert_close(hidden.grad, torch.tensor([[3.0]]))
    assert all(parameter.grad is None for parameter in support.frozen_critics.parameters())
    assert all(parameter.grad is None for parameter in support.critics.parameters())


def test_twin_cost_bootstrap_and_actor_both_use_pessimistic_value():
    support = CostActorCritic(1, ActorCriticSettings(algorithm="ahac", hidden_sizes=(4,)))
    for index, critic in enumerate(support.frozen_critics):
        for parameter in critic.parameters():
            parameter.zero_()
        critic.network[-1].bias.fill_(2.0 + 4 * index)
    observation = torch.ones(3, 1, requires_grad=True)
    torch.testing.assert_close(support.actor_value(observation), torch.full((3,), 6.0))
    torch.testing.assert_close(support.bootstrap_value(observation), torch.full((3,), 6.0))
    assert support.actor_value(observation).requires_grad
    assert not support.bootstrap_value(observation).requires_grad


def test_twin_parameters_are_independent_and_actor_uses_larger_cost_gradient():
    support = CostActorCritic(1, ActorCriticSettings(algorithm="ahac", hidden_sizes=(4,)))
    first = list(support.critics[0].parameters())
    second = list(support.critics[1].parameters())
    assert all(left.data_ptr() != right.data_ptr() for left, right in zip(first, second, strict=True))
    networks = []
    for slope in (2.0, 5.0):
        linear = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            linear.weight.fill_(slope)
        networks.append(nn.Sequential(linear, nn.Flatten(0)))
    support.frozen_critics = nn.ModuleList(networks).requires_grad_(False)
    observation = torch.tensor([[3.0]], requires_grad=True)
    support.actor_value(observation).sum().backward()
    torch.testing.assert_close(observation.grad, torch.tensor([[5.0]]))
    assert all(parameter.grad is None for parameter in support.frozen_critics.parameters())


def test_value_fit_is_independent_of_actor_graph_and_reduces_fixed_target_error():
    settings = ActorCriticSettings(
        hidden_sizes=(16, 16), critic_lr=0.01, critic_iterations=50, critic_minibatches=4, target_tau=1.0, seed=5
    )
    support = CostActorCritic(2, settings)
    observation = torch.linspace(-1, 1, 64).reshape(4, 8, 2).requires_grad_()
    costs = observation.detach().square().sum(dim=-1) + 1
    next_values = torch.zeros_like(costs)
    cuts = torch.ones_like(costs, dtype=torch.bool)
    before = (support.bootstrap_value(observation) - costs).square().mean()
    metrics = support.fit(observation, costs.requires_grad_(), next_values, cuts)
    after = (support.bootstrap_value(observation) - costs.detach()).square().mean()
    assert after < before * 0.1
    assert metrics["critic_iterations"] == 16
    assert metrics["critic_optimizer_steps"] == 64
    assert observation.grad is None and costs.grad is None
    assert all(parameter.grad is None for parameter in support.frozen_critics.parameters())


def test_shac_target_ema_and_ahac_hard_snapshot():
    for algorithm, expected in (("shac", 0.5), ("ahac", 2.0)):
        support = CostActorCritic(2, ActorCriticSettings(algorithm=algorithm, hidden_sizes=(4,), target_tau=0.25))
        with torch.no_grad():
            for parameter in support.critics.parameters():
                parameter.fill_(2.0)
            for parameter in support.frozen_critics.parameters():
                parameter.zero_()
        support.update_frozen_critics()
        for parameter in support.frozen_critics.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, expected))


@pytest.mark.parametrize("value_transform", ("linear", "softplus"))
def test_checkpoint_restores_optimizer_value_snapshot_and_private_shuffle_rng(value_transform):
    settings = ActorCriticSettings(
        hidden_sizes=(8,), critic_iterations=2, critic_minibatches=3, seed=11, value_transform=value_transform
    )
    support = CostActorCritic(2, settings)
    observation = torch.linspace(-1, 1, 16).reshape(2, 4, 2)
    costs = observation.square().sum(dim=-1)
    next_values = torch.zeros_like(costs)
    cuts = torch.zeros_like(costs, dtype=torch.bool)
    support.fit(observation, costs, next_values, cuts)
    restored = CostActorCritic(2, settings)
    restored.load_state_dict(copy.deepcopy(support.state_dict()))
    first = support.fit(observation, costs, next_values, cuts)
    second = restored.fit(observation, costs, next_values, cuts)
    assert first == second
    for left, right in zip(support.critics.parameters(), restored.critics.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    torch.testing.assert_close(
        support.bootstrap_value(observation), restored.bootstrap_value(observation), atol=0, rtol=0
    )


@pytest.mark.parametrize("value_transform", ("linear", "softplus"))
def test_critic_construction_preserves_global_rng(value_transform):
    rng = torch.get_rng_state().clone()
    CostActorCritic(2, ActorCriticSettings(hidden_sizes=(4,), seed=17, value_transform=value_transform))
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)


def test_default_value_output_keeps_signed_raw_values_and_original_gradients():
    assert ActorCriticSettings().value_transform == "linear"
    critic = CostValueNetwork(2, (4,))
    critic.network = nn.Linear(2, 1)
    with torch.no_grad():
        critic.network.weight.copy_(torch.tensor([[2.0, -3.0]]))
        critic.network.bias.fill_(-4.0)
    observation = torch.tensor([[0.0, 1.0], [3.0, 0.0]], requires_grad=True)
    value = critic(observation)
    torch.testing.assert_close(value, torch.tensor([-7.0, 2.0]), atol=0, rtol=0)
    value.sum().backward()
    torch.testing.assert_close(observation.grad, torch.tensor([[2.0, -3.0], [2.0, -3.0]]), atol=0, rtol=0)


def test_softplus_has_identical_initial_weights_and_independent_cost_twins():
    linear = CostActorCritic(2, ActorCriticSettings(algorithm="ahac", hidden_sizes=(4,), seed=23))
    positive = CostActorCritic(
        2,
        ActorCriticSettings(
            algorithm="ahac",
            hidden_sizes=(4,),
            seed=23,
            value_transform="softplus",
        ),
    )
    for old, new in zip(linear.critics.parameters(), positive.critics.parameters(), strict=True):
        torch.testing.assert_close(old, new, atol=0, rtol=0)
    twins = [list(critic.parameters()) for critic in positive.critics]
    assert all(left.data_ptr() != right.data_ptr() for left, right in zip(*twins, strict=True))
    assert linear.optimizer.state_dict()["param_groups"] == positive.optimizer.state_dict()["param_groups"]
    observation = torch.linspace(-2, 2, 12).reshape(6, 2)
    expected = torch.nn.functional.softplus(linear.actor_value(observation), beta=1, threshold=20)
    torch.testing.assert_close(positive.actor_value(observation), expected, atol=0, rtol=0)
    assert (positive.bootstrap_value(observation) >= 0).all()


@pytest.mark.parametrize("algorithm", ("shac", "ahac"))
@pytest.mark.parametrize("td_lambda", (0.0, 0.95, 1.0))
def test_nonnegative_value_and_cost_produce_nonnegative_td_targets(algorithm, td_lambda):
    support = CostActorCritic(
        2,
        ActorCriticSettings(
            algorithm=algorithm,
            hidden_sizes=(4,),
            value_transform="softplus",
        ),
    )
    for index, critic in enumerate(support.frozen_critics):
        for parameter in critic.parameters():
            parameter.zero_()
        critic.network[-1].bias.fill_(-3.0 - index)
    next_observation = torch.zeros(3, 2, 2)
    next_values = support.bootstrap_value(next_observation)
    costs = torch.tensor([[0.1, 1.0], [2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    cuts = torch.tensor([[False, True], [True, False], [False, False]])
    targets = td_lambda_cost_targets(costs, next_values, cuts, discount=0.99, td_lambda=td_lambda)
    assert targets.shape == costs.shape
    assert (next_values >= 0).all() and (targets >= costs).all()
    assert not targets.requires_grad and not next_values.requires_grad


def test_softplus_frozen_value_keeps_position_and_memory_input_gradients():
    support = CostActorCritic(2, ActorCriticSettings(hidden_sizes=(4,), value_transform="softplus"))
    critic = support.frozen_critics[0]
    critic.network = nn.Linear(2, 1)
    with torch.no_grad():
        critic.network.weight.copy_(torch.tensor([[2.0, 3.0]]))
        critic.network.bias.fill_(-2.0)
    critic.requires_grad_(False)
    position = torch.tensor([[0.4]], requires_grad=True)
    memory = torch.tensor([[0.1]], requires_grad=True)
    support.actor_value(torch.cat((position, memory), dim=-1)).sum().backward()
    slope = torch.sigmoid(torch.tensor(-0.9))
    torch.testing.assert_close(position.grad, (2 * slope).reshape(1, 1))
    torch.testing.assert_close(memory.grad, (3 * slope).reshape(1, 1))
    assert all(parameter.grad is None for parameter in support.frozen_critics.parameters())
    assert all(parameter.grad is None for parameter in support.critics.parameters())


def test_softplus_extreme_values_are_finite_and_large_positive_values_are_linear():
    critic = CostValueNetwork(1, (4,), value_transform="softplus")
    critic.network = nn.Identity()
    observation = torch.tensor(
        [[-1000.0], [-20.0], [-1.0], [0.0], [30.0], [1000.0]], dtype=torch.float64, requires_grad=True
    )
    values = critic(observation)
    assert torch.isfinite(values).all() and (values >= 0).all()
    torch.testing.assert_close(values[-2:], observation[-2:, 0], atol=0, rtol=0)
    values.sum().backward()
    expected = torch.where(observation.detach() > 20, torch.ones_like(observation), torch.sigmoid(observation.detach()))
    torch.testing.assert_close(observation.grad, expected)
    assert observation.grad[1].item() < 1e-8  # Documents lower-tail saturation; it is not hidden by a clamp.


def test_legacy_linear_checkpoints_load_only_into_linear_value_networks():
    settings = ActorCriticSettings(hidden_sizes=(4,), seed=29)
    original = CostActorCritic(2, settings)
    legacy = copy.deepcopy(original.state_dict())
    legacy["settings"].pop("value_transform")
    restored = CostActorCritic(2, settings)
    restored.load_state_dict(legacy)
    observation = torch.ones(3, 2)
    torch.testing.assert_close(original.actor_value(observation), restored.actor_value(observation), atol=0, rtol=0)
    positive = CostActorCritic(2, ActorCriticSettings(hidden_sizes=(4,), seed=29, value_transform="softplus"))
    with pytest.raises(ValueError, match="value_transform"):
        positive.load_state_dict(legacy)
    with pytest.raises(ValueError, match="value_transform"):
        restored.load_state_dict(positive.state_dict())
    assert positive.state_dict()["settings"]["value_transform"] == "softplus"


def test_fit_next_value_diagnostics_describe_supplied_snapshot_without_actor_gradients():
    support = CostActorCritic(2, ActorCriticSettings(hidden_sizes=(4,), critic_iterations=1))
    observations = torch.zeros(2, 2, 2, requires_grad=True)
    costs = torch.ones(2, 2, requires_grad=True)
    following = torch.tensor([[-3.0, -1.0], [2.0, 6.0]], requires_grad=True)
    cuts = torch.ones(2, 2, dtype=torch.bool)
    metrics = support.fit(observations, costs, following, cuts)
    assert metrics["critic_next_value_min"] == -3.0
    assert metrics["critic_next_value_mean"] == 1.0
    assert metrics["critic_next_value_negative_fraction"] == 0.5
    assert observations.grad is None and costs.grad is None and following.grad is None


def test_dcol_risk_uses_nearest_gap_and_preserves_penalty_gradient():
    controller = AdaptiveHorizonController(AHACSettings(initial_dual=2.0))
    gaps = torch.tensor([[[0.1, 0.3, 0.2]], [[0.4, 0.2, 0.5]]], requires_grad=True)
    risk = controller.risk(gaps)
    expected = torch.nn.functional.softplus(torch.tensor([[0.0], [-2.0]]))
    torch.testing.assert_close(risk, expected)
    penalty = controller.penalty(risk)
    torch.testing.assert_close(penalty, 2.0 * (expected.max() - 0.1))
    penalty.backward()
    expected_gradient = torch.zeros_like(gaps)
    expected_gradient[0, 0, 0] = -20.0
    expected_gradient[1, 0, 1] = 0
    torch.testing.assert_close(gaps.grad, expected_gradient)
    assert controller.dual == 2.0
    empty_geometry = controller.risk(torch.full((2, 3, 1), float("inf")))
    torch.testing.assert_close(empty_geometry, torch.zeros(2, 3))


def test_synchronous_horizon_feedback_grows_shrinks_and_holds_in_band():
    controller = AdaptiveHorizonController(
        AHACSettings(
            initial_horizon=10,
            min_horizon=4,
            max_horizon=16,
            risk_budget=0.125,
            initial_dual=0.0,
            stable_updates_required=1,
        )
    )
    grown = controller.update(torch.zeros(10, 3, requires_grad=True))
    assert controller.horizon == 14 and grown["ahac_horizon_used"] == 10
    assert grown["ahac_horizon_next"] == 14 and controller.dual == 0
    controller.update(torch.zeros(14, 3))
    assert controller.horizon == 16
    for exact_boundary in (0.0625, 0.125):
        controller.update(torch.full((16, 3), exact_boundary))
        assert controller.horizon == 16
    shrunken = controller.update(torch.full((16, 3), 0.5))
    assert controller.horizon == 12
    assert shrunken["ahac_risk_over_budget_fraction"] == 1
    assert controller.dual == pytest.approx(0.0375)
    while controller.horizon > 4:
        controller.update(torch.full((controller.horizon, 3), 0.5))
    controller.update(torch.full((4, 3), 0.5))
    assert type(controller.horizon) is int and controller.horizon == 4


def test_dual_projection_and_horizon_use_global_parallel_risk_not_individual_resets():
    settings = AHACSettings(
        initial_horizon=4, min_horizon=2, max_horizon=8, dual_lr=10.0, max_dual=1.0, stable_updates_required=1
    )
    controller = AdaptiveHorizonController(settings)
    risk = torch.tensor([[0.0, 1.0]]).expand(4, 2)
    metrics = controller.update(risk)
    assert controller.horizon == 3
    assert controller.dual == 1.0
    assert metrics["ahac_risk_mean"] == 0.5
    assert metrics["ahac_risk_over_budget_fraction"] == 0.5
    controller.update(torch.zeros(3, 2))
    assert controller.horizon == 7 and controller.dual == 0
    repeated = AdaptiveHorizonController(settings)
    repeated.update(risk.repeat(1, 5))
    assert repeated.horizon == 3 and repeated.dual == 1.0


def test_ahac_penalty_uses_dual_snapshot_even_if_controller_updates_before_backward():
    controller = AdaptiveHorizonController(
        AHACSettings(
            initial_horizon=2,
            min_horizon=2,
            max_horizon=8,
            initial_dual=2.0,
        )
    )
    risk = torch.tensor([[0.3], [0.5]], requires_grad=True)
    penalty = controller.penalty(risk)
    controller.update(risk)
    assert controller.dual > 2.0
    penalty.backward()
    torch.testing.assert_close(risk.grad, torch.tensor([[0.0], [2.0]]))


def test_ahac_checkpoint_resumes_dual_and_horizon_exactly():
    settings = AHACSettings(initial_horizon=10, min_horizon=4, max_horizon=16)
    controller = AdaptiveHorizonController(settings)
    controller.update(torch.full((10, 2), 0.5))
    restored = AdaptiveHorizonController(settings)
    restored.load_state_dict(copy.deepcopy(controller.state_dict()))
    risk = torch.full((controller.horizon, 2), 0.01)
    assert controller.update(risk) == restored.update(risk)
    assert controller.state_dict() == restored.state_dict()
    incompatible = AdaptiveHorizonController(
        AHACSettings(
            initial_horizon=10,
            min_horizon=4,
            max_horizon=16,
            risk_budget=0.2,
        )
    )
    with pytest.raises(ValueError, match="incompatible"):
        incompatible.load_state_dict(controller.state_dict())


def test_ahac_rejects_partial_rollouts_nonfinite_risk_and_invalid_settings():
    controller = AdaptiveHorizonController()
    with pytest.raises(ValueError, match="horizon"):
        controller.update(torch.zeros(controller.horizon - 1, 3))
    for invalid in (float("nan"), float("inf"), -0.1):
        with pytest.raises(FloatingPointError, match="finite and nonnegative"):
            controller.update(torch.full((controller.horizon, 3), invalid))
    with pytest.raises(ValueError, match="sample"):
        controller.risk(torch.ones(3, 2))
    for changes in (
        {"risk_scale": 0},
        {"risk_budget": 0},
        {"initial_horizon": 1},
        {"shrink_factor": 1},
        {"growth_steps": 0},
        {"initial_dual": -1},
        {"max_dual": 0},
        {"safe_budget_fraction": 0},
        {"dual_lr": float("nan")},
    ):
        with pytest.raises(ValueError):
            AHACSettings(**changes).validate()


def test_invalid_trajectory_shapes_and_value_settings_fail_clearly():
    costs = torch.ones(3, 2)
    with pytest.raises(ValueError, match="match"):
        td_lambda_cost_targets(costs, costs[:1], torch.zeros_like(costs, dtype=torch.bool))
    with pytest.raises(ValueError, match="boolean"):
        short_horizon_cost_objective(costs, costs, torch.zeros_like(costs))
    for changes in (
        {"algorithm": "bptt"},
        {"discount": 1.0},
        {"td_lambda": -0.1},
        {"target_tau": 0},
        {"critic_minibatches": 0},
        {"hidden_sizes": ()},
        {"value_transform": "relu"},
    ):
        with pytest.raises(ValueError):
            ActorCriticSettings(**changes).validate()
    with pytest.raises(ValueError, match="value_transform"):
        CostValueNetwork(2, (4,), value_transform="relu")
