"""An absorbing failure is charged once, in the same actor/TD cost units."""

import copy

import pytest
import torch
from test_diffnav_contract import loss_history

from aerialnav.diffnav.actor_critic import short_horizon_cost_objective, td_lambda_cost_targets
from aerialnav.diffnav.cli import parse_train_args
from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.diffnav.losses import diffnav_transition_costs, height_boundary_loss


def test_terminal_failure_cost_is_shared_by_actor_and_td_without_bootstrapping_failure():
    cfg = DiffNavTrainCfg(algorithm="shac", height_loss_mode="boundary_huber", failure_terminal_cost=300)
    velocity = torch.zeros(2, 3, 3, requires_grad=True)
    history = loss_history(velocity)
    history["dcol_gap"] = [torch.ones(3, 9), torch.ones(3, 9)]
    # Failure, success and timeout are different event outcomes.
    failed = torch.tensor([[True, False, False], [False, False, False]])
    history["failure_terminal"] = list(failed)
    baseline = copy.copy(cfg)
    baseline.failure_terminal_cost = 0
    _, _, old_cost = diffnav_transition_costs(history, baseline)
    losses, _, cost = diffnav_transition_costs(history, cfg)
    torch.testing.assert_close(cost - old_cost, 300 * failed.float())
    assert losses["failure_terminal"] == 50
    done = torch.tensor([[True, True, True], [False, False, False]])
    next_values = torch.tensor([[0.0, 0.0, 100.0], [100.0, 100.0, 100.0]])
    targets = td_lambda_cost_targets(cost, next_values, done)
    assert targets[0, 0] == cost[0, 0]
    assert targets[0, 2] == cost[0, 2] + 99
    objective = short_horizon_cost_objective(cost, next_values, done)
    original = short_horizon_cost_objective(old_cost, next_values, done)
    assert float((objective["objective"] - original["objective"]).detach()) == pytest.approx(50)
    torch.testing.assert_close(
        torch.autograd.grad(cost.sum(), velocity, retain_graph=True)[0],
        torch.autograd.grad(old_cost.sum(), velocity)[0],
    )
    upper = torch.tensor([[0.0, 0.0, 3.9]], requires_grad=True)
    assert torch.autograd.grad(height_boundary_loss(upper), upper)[0][0, 2] > 0


def test_failure_cost_boundary_validation_and_shared_cli():
    values = parse_train_args(["--algorithm", "shac", "--failure_terminal_cost", "300", "--critic_target_tau", ".005"])
    DiffNavTrainCfg(**values).validate()
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="failure_terminal_cost"):
            DiffNavTrainCfg(algorithm="shac", failure_terminal_cost=value).validate()
    with pytest.raises(ValueError, match="requires SHAC"):
        parse_train_args(["--failure_terminal_cost", "1"])
    for value in (0, 1.1):
        with pytest.raises(ValueError):
            parse_train_args(["--algorithm", "shac", "--critic_target_tau", str(value)])
