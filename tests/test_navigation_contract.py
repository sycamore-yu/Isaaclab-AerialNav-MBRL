"""Numerical task examples and first-event accounting required by the ADR."""

import math

import torch

from aerialnav.evaluation import EpisodeEvaluation
from aerialnav.tasks.navigation.task_logic import (
    DynamicObstacleMotion,
    goal_frame,
    navigation_terms,
    obstacle_categories,
)


def test_navigation_free_space_observation_and_reward():
    position = torch.tensor([[0.0, 0.0, 2.0]])
    velocity = torch.tensor([[1.0, 0.0, 0.0]])
    goal = torch.tensor([[10.0, 0.0, 2.0]])
    result = navigation_terms(
        position,
        velocity,
        goal,
        goal - position,
        torch.full((1, 1, 36, 4), 4.0),
        DynamicObstacleMotion(0, 0, "cpu", 0.016),
        velocity,
        torch.tensor([[2.0, 2.0]]),
    )
    torch.testing.assert_close(result["state"], torch.tensor([[1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 0.0]]))
    torch.testing.assert_close(result["reward"], torch.tensor([2 + math.log(4)]))
    assert result["lidar"].count_nonzero() == 0
    assert result["dynamic_obstacle"].count_nonzero() == 0


def test_goal_frame_is_the_fixed_episode_frame():
    heading = torch.tensor([[0.0, 5.0, 4.0]])
    local = torch.tensor([[1.0, 2.0, 3.0]])
    torch.testing.assert_close(goal_frame(local, heading, inverse=True), torch.tensor([[-2.0, 1.0, 3.0]]))
    torch.testing.assert_close(goal_frame(torch.tensor([[-2.0, 1.0, 3.0]]), heading), local)


def test_requested_obstacle_count_is_exact_and_not_rounded_to_96():
    counts = obstacle_categories(100)
    assert counts == [13, 13, 12, 12, 13, 13, 12, 12]
    motion = DynamicObstacleMotion(100, 0, "cpu", 0.016)
    assert motion.position.shape == (100, 3)
    assert motion.cylinders.sum() == 50
    motion.advance()
    assert torch.isfinite(motion.position).all()
    speed = motion.velocity.norm(dim=-1)
    assert torch.all((speed >= 0.5 - 1e-6) & (speed <= 1.5 + 1e-6))


def test_first_arrival_survives_later_collision_and_same_step_collision_wins():
    evaluation = EpisodeEvaluation([[0, 0, 2]] * 2, [[10, 0, 2]] * 2, seed=10000)
    transition = {
        "reached": torch.tensor([True, True]),
        "collision": torch.tensor([False, True]),
        "bounds": torch.tensor([False, False]),
        "invalid": torch.tensor([False, False]),
        "timeout": torch.tensor([False, False]),
        "time_s": torch.tensor([4.0, 5.0]),
        "step": torch.tensor([40, 50]),
    }
    evaluation.update(transition)
    transition["collision"][:] = True
    evaluation.update(transition)
    summary = evaluation.summary()
    assert summary["success_rate"] == 0.5
    assert summary["collision_rate"] == 0.5
    assert summary["success_time_mean_s"] == 4.0


def test_evaluation_preserves_an_earlier_physics_substep_arrival():
    from aerialnav.evaluation import EpisodeEvaluation

    evaluation = EpisodeEvaluation([[0, 0, 2]], [[1, 0, 2]], seed=4)
    transition = {
        "reached": torch.tensor([False]),
        "collision": torch.tensor([True]),
        "bounds": torch.tensor([False]),
        "invalid": torch.tensor([False]),
        "timeout": torch.tensor([False]),
        "time_s": torch.tensor([0.1]),
        "step": torch.tensor([1]),
        "first_event_code": torch.tensor([1]),
        "first_event_time_s": torch.tensor([0.03]),
    }
    evaluation.update(transition)
    assert evaluation.summary()["success_rate"] == 1
    assert abs(evaluation.summary()["success_time_mean_s"] - 0.03) < 1e-6
