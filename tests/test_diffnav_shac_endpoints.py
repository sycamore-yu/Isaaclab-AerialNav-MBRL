"""Exercise real runner bootstrap graphs with deliberately different reset states."""

import copy
from types import SimpleNamespace

import pytest
import torch
from test_diffnav_contract import AnalyticEnv

from aerialnav.diffnav.actor_critic import td_lambda_cost_targets
from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.diffnav.runner import DiffNavTrainer


class EndpointEnv(AnalyticEnv):
    max_episode_length = 10

    def __init__(self, trigger_step=1):
        super().__init__()
        self.trigger_step = trigger_step
        self.cfg.variant = "dcol"
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.motion = SimpleNamespace(position=torch.empty(0, 3), velocity=torch.empty(0, 3), size=torch.empty(0, 3))
        self.terrain_distance = SimpleNamespace(signed_distance=lambda q: q[..., 0] * 0 + 100)
        self.target = torch.tensor([[10.0, 0.0, 2.0]]).repeat(2, 1)
        self.terminated_flags = torch.zeros(2, dtype=torch.bool)
        self.timeout_flags = torch.zeros(2, dtype=torch.bool)

    def reset(self):
        super().reset()
        self.episode_length_buf.zero_()

    def get_diffnav_observation(self):
        result = super().get_diffnav_observation()
        result["target"] = result["policy_target"] = self.target.clone()
        result["depth_features"] = self.depth.clone()
        result["policy_depth_features"] = self.depth.clone()
        result["height_range"] = torch.tensor([[2.0, 2.0]]).repeat(2, 1)
        return result

    def step(self, command):
        executed = self.perturbation.execute(command)
        self.total_steps += 1
        self.episode_length_buf += 1
        self.velocity += executed * self.step_dt
        self.position += self.velocity * self.step_dt
        if self.total_steps == self.trigger_step:
            self.position[:, 0] = torch.tensor([3.0, 4.0])
        transition = {
            "position": self.position.clone(),
            "velocity": self.velocity.clone(),
            "quaternion": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
            "target": self.target.clone(),
            "depth_features": self.depth.clone(),
            "height_target": self.target[:, 2].clone(),
            "step": self.episode_length_buf.clone(),
            "input_delay_steps": torch.zeros(2, dtype=torch.long),
            "action_delay_steps": self.perturbation.action_buffer.time_lags.clone(),
            "critic_parameters": copy.deepcopy(self.diffnav_parameters),
            "executed_action": executed.clone(),
            "actuator": executed.clone(),
            "collision_position": self.position.clone(),
            "collision_substep": torch.zeros(2, dtype=torch.long),
            "dynamic_positions_by_substep": torch.empty(10, 0, 3),
            "bounds": torch.zeros(2, dtype=torch.bool),
        }
        self.terminated_flags.copy_(torch.tensor([self.total_steps == self.trigger_step, False]))
        self.timeout_flags.copy_(torch.tensor([False, self.total_steps == self.trigger_step]))
        terminated, timeout = self.terminated_flags, self.timeout_flags
        transition["collision"] = terminated.clone()
        transition["first_event_code"] = torch.where(terminated, 2, torch.where(timeout, 5, 0))
        if self.total_steps == self.trigger_step:
            self.position[:, 0] = 100.0
            self.target[:, 0] = 200.0
            self.depth.fill_(9.0)
            self.velocity.zero_()
            self.episode_length_buf.zero_()
            self.perturbation.reset(torch.arange(2))
            self.episode_id += 1
        return {}, torch.zeros(2), terminated, timeout, {"transition": transition}


def _run_endpoint_case(tmp_path, trigger_step=1, algorithm="shac", failure_cost=0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    env = EndpointEnv(trigger_step)
    cfg = DiffNavTrainCfg(
        algorithm=algorithm,
        physics_dt=0.01,
        policy_dt=0.1,
        ahac_min_horizon=2,
        ahac_initial_horizon=2,
        ahac_max_horizon=2,
        timesteps=2,
        max_iterations=1,
        critic_iterations=1,
        critic_max_iterations=1,
        failure_terminal_cost=failure_cost,
    )
    trainer = DiffNavTrainer(env, cfg, tmp_path)
    trainer._dcol_gap = lambda query, radius, positions=None, **kwargs: query[..., 0] * 0 + 5
    original_initialize = trainer._initialize_critic
    endpoints, values, targets = [], [], []

    def initialize(example):
        original_initialize(example)
        support = trainer.actor_critic
        actor_value, fit = support.actor_value, support.fit

        def observe_endpoint(observation):
            observation.retain_grad()
            endpoints.append(observation)
            value = actor_value(observation)
            values.append(value.detach().clone())
            return value

        def observe_targets(observations, costs, next_values, cuts):
            targets.append((next_values.clone(), cuts.clone(), costs.detach().clone()))
            return fit(observations, costs, next_values, cuts)

        support.actor_value, support.fit = observe_endpoint, observe_targets

    trainer._initialize_critic = initialize
    result = trainer.run()
    assert result["completed_updates"] == 1 and result["nonfinite_grad_events"] == 0
    assert trainer.cfg.resolved_velocity_loss_mode == "per_step"
    return cfg, endpoints, values, targets


@pytest.mark.parametrize("algorithm", ["shac", "ahac"])
def test_shac_absorbing_timeout_and_live_final_endpoint(tmp_path, algorithm):
    cfg, endpoints, values, targets = _run_endpoint_case(tmp_path, algorithm=algorithm)
    # The timeout uses old target 10 and position 4, never reset target 200/position 100.
    torch.testing.assert_close(endpoints[0][:, 0].detach(), torch.tensor([0.14, 0.12]))
    torch.testing.assert_close(endpoints[0][:, 25:217].detach(), torch.ones(2, 192))
    assert endpoints[0][:, 217:409].detach().abs().sum() > 0
    assert not endpoints[0].grad[0].any()  # True terminal: no bootstrap input gradient.
    assert endpoints[0].grad[1].abs().sum() > 0  # Timeout: live original endpoint.
    assert endpoints[-1].grad.abs().sum(dim=-1).min() > 0  # Ordinary horizon endpoint.
    next_values, cuts, costs = targets[0]
    assert next_values[0, 0] == 0
    torch.testing.assert_close(next_values[0, 1], values[0][1])
    assert cuts[0].all() and not cuts[1].any()
    td = td_lambda_cost_targets(costs, next_values, cuts, discount=cfg.discount, td_lambda=cfg.td_lambda)
    torch.testing.assert_close(td[0, 0], costs[0, 0])
    torch.testing.assert_close(td[0, 1], costs[0, 1] + cfg.discount * values[0][1])


@pytest.mark.parametrize("algorithm", ["shac", "ahac"])
def test_absorbing_terminal_at_horizon_does_not_gain_a_bootstrap(tmp_path, algorithm):
    cfg, endpoints, values, targets = _run_endpoint_case(tmp_path, trigger_step=2, algorithm=algorithm)
    next_values, cuts, costs = targets[0]
    assert cuts[-1].all() and next_values[-1, 0] == 0
    assert not endpoints[-1].grad[0].any()
    assert endpoints[-1].grad[1].abs().sum() > 0
    td = td_lambda_cost_targets(costs, next_values, cuts, discount=cfg.discount)
    torch.testing.assert_close(td[-1, 0], costs[-1, 0])
    torch.testing.assert_close(td[-1, 1], costs[-1, 1] + cfg.discount * values[-1][1])


def test_failure_cost_flows_through_real_runner_without_changing_terminal_bootstrap(tmp_path):
    torch.manual_seed(27)
    _, _, _, baseline = _run_endpoint_case(tmp_path / "base")
    torch.manual_seed(27)
    _, _, _, repaired = _run_endpoint_case(tmp_path / "repair", failure_cost=300)
    old_values, old_cuts, old_costs = baseline[0]
    new_values, new_cuts, new_costs = repaired[0]
    torch.testing.assert_close(old_values, new_values)
    assert new_values[0, 0] == 0
    torch.testing.assert_close(old_cuts, new_cuts)
    torch.testing.assert_close(new_costs - old_costs, torch.tensor([[300.0, 0.0], [0.0, 0.0]]))
