"""Reset-safe losses and strict, disjoint, completed-episode S0 acceptance."""

import torch

from aerialnav.pointnet.config import PointNetTrainCfg
from aerialnav.pointnet.losses import causal_velocity_average, pointcloud_transition_costs
from aerialnav.tasks.navigation.curriculum import TrainingCurriculum


def test_velocity_average_handles_young_episodes_resets_and_update_boundaries():
    velocity = torch.tensor([1.0, 3.0, 100.0, 200.0])[:, None, None].expand(-1, 1, 3).clone().requires_grad_()
    episode = torch.tensor([[0], [0], [1], [1]])
    expected, _ = causal_velocity_average(velocity, episode, 3)
    torch.testing.assert_close(expected[:, 0, 0], torch.tensor([1.0, 2.0, 100.0, 150.0]))
    _, prefix = causal_velocity_average(velocity[:1], episode[:1], 3)
    actual, _ = causal_velocity_average(velocity[1:], episode[1:], 3, prefix)
    torch.testing.assert_close(actual, expected[1:])
    actual.sum().backward()
    assert velocity.grad[0].abs().sum() == 0
    assert (velocity.grad[1:].abs().sum(-1) > 0).all()


def make_history():
    command = torch.tensor(
        [[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]], requires_grad=True
    )
    distance = torch.linspace(0.8, 0.7, 10)[None, None].expand(4, 1, -1).clone().requires_grad_()
    velocity = torch.zeros(4, 1, 3, requires_grad=True)
    history = {
        "velocity": list(velocity),
        "episode_id": list(torch.tensor([[0], [0], [1], [1]])),
        "target_velocity": list(torch.tensor([[[1.0, 0.0, 0.0]]]).expand(4, 1, 3)),
        "obstacle_signed_distance": list(distance),
        "margin": list(torch.full((4, 1), 0.3)),
        "command": list(command),
        "previous_command": list(
            torch.tensor([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]])
        ),
        "episode_step": list(torch.tensor([[1], [2], [1], [2]])),
        "position": list(torch.tensor([[[0.0, 0.0, 1.0]]]).expand(4, 1, 3)),
        "height_range": list(torch.tensor([[[0.5, 2.5]]]).expand(4, 1, 2)),
    }
    return history, velocity, distance, command


def test_pointnet_objective_has_early_velocity_and_escape_gradients_without_dcol():
    cfg = PointNetTrainCfg()
    cfg.validate()
    history, velocity, distance, command = make_history()
    losses, _, cost = pointcloud_transition_costs(history, cfg)
    assert losses["velocity_valid_samples"] == 4
    assert losses["velocity"] > 0
    assert cost.shape == (4, 1)
    assert losses["velocity_prediction"] == 0
    assert "dcol_gap" not in history
    # Only one noninitial jerk sample per episode: each segment's variance is zero.
    torch.testing.assert_close(losses["jerk"], torch.tensor(5.0))
    losses["total"].backward()
    assert (velocity.grad[..., 0] < 0).all()
    assert (distance.grad[..., 1:] < 0).all()
    assert torch.isfinite(command.grad).all()


def test_retreating_obstacle_has_zero_cost_and_gradient():
    history, _, distance, _ = make_history()
    history["obstacle_signed_distance"] = list(distance.flip(-1))
    losses, _, _ = pointcloud_transition_costs(history, PointNetTrainCfg())
    assert losses["obstacle_avoidance"] == 0
    assert losses["collision"] == 0
    gradient = torch.autograd.grad(losses["collision"], distance)[0]
    assert (gradient == 0).all()


def test_velocity_penalizes_speed_difference_not_vector_error_norm():
    history, _, _, _ = make_history()
    history["velocity"] = list(torch.tensor([[[-1.0, 0.0, 0.0]]]).expand(4, 1, 3))
    losses, _, _ = pointcloud_transition_costs(history, PointNetTrainCfg())
    # Equal speeds, opposite directions: speed term is zero, axis term is 0.6*1.5.
    torch.testing.assert_close(losses["velocity"], torch.tensor(0.9))


def test_acceleration_is_squared_l2_with_exact_gradient():
    history, _, _, command = make_history()
    losses, _, _ = pointcloud_transition_costs(history, PointNetTrainCfg())
    torch.testing.assert_close(losses["acceleration"], command.square().sum(-1).mean())
    gradient = torch.autograd.grad(losses["acceleration"], command)[0]
    torch.testing.assert_close(gradient, 2 * command / 4)


def test_entire_collision_penalty_is_scaled_by_closing_speed():
    from torch.nn import functional as F

    history, _, distance, _ = make_history()
    cfg = PointNetTrainCfg()
    losses, _, _ = pointcloud_transition_costs(history, cfg)
    clearance = distance - 0.3
    closing = (-clearance.diff(dim=-1) * (9 / cfg.policy_dt)).clamp_min(0).detach()
    expected = (
        closing
        * (
            cfg.coef_obj_avoidance * (1 - clearance[..., 1:]).relu().square()
            + cfg.coef_collide * F.softplus(-32 * clearance[..., 1:])
        )
    ).mean()
    actual = cfg.coef_obj_avoidance * losses["obstacle_avoidance"] + cfg.coef_collide * losses["collision"]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, distance, retain_graph=True)[0], torch.autograd.grad(expected, distance)[0]
    )


def test_release_terminal_outcome_overrides_early_arrival():
    curriculum = TrainingCurriculum(1, "cpu", stage="pretrain_s0")
    curriculum.min_episodes = 1
    curriculum.observe(
        {
            "first_event_code": torch.tensor([1]),
            "first_event_time_s": torch.tensor([1.0]),
            "terminal_event_code": torch.tensor([5]),
            "arrive_time": torch.tensor([0.0]),
            "terminated": torch.tensor([False]),
            "timeout": torch.tensor([True]),
            "episode_return": torch.tensor([0.0]),
            "step": torch.tensor([600]),
        }
    )
    report = curriculum.pop_report()
    assert report["success_rate"] == 0 and report["timeout_rate"] == 1
