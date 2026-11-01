"""Training outcomes, source-style episode aggregation and curriculum promotion."""

import pytest
import torch

from aerialnav.tasks.navigation.curriculum import TrainingCurriculum, active_obstacle_indices
from aerialnav.tasks.navigation.task_logic import dynamic_layout


def transition(codes, done, times=None):
    code = torch.tensor(codes)
    return {
        "first_event_code": code,
        "first_event_time_s": torch.tensor(times if times is not None else [1.0] * len(codes)),
        "terminated": torch.tensor(done),
        "timeout": code == 5,
        "episode_return": torch.zeros(len(codes)),
        "step": torch.ones(len(codes), dtype=torch.long),
    }


def test_first_result_is_counted_only_when_the_training_episode_finishes():
    curriculum = TrainingCurriculum(4, "cpu")
    curriculum.observe(transition([1, 0, 0, 0], [False] * 4, [2.5, 0, 0, 0]))
    assert curriculum.pop_report() is None
    curriculum.observe(transition([2, 2, 3, 5], [True] * 4))
    report = curriculum.pop_report()
    assert report["episodes"] == 4
    assert report["counts"] == {"success": 1, "collision": 1, "bounds": 1, "invalid": 0, "timeout": 1}
    assert report["success_rate"] == 0.25
    assert report["success_time_mean_s"] == 2.5
    assert curriculum.pop_report() is None


def test_promotion_uses_current_stage_samples_and_saves_100_stage_target():
    curriculum = TrainingCurriculum(10, "cpu")
    for difficulty in (60, 80, 100):
        assert curriculum.active_obstacles == difficulty
        curriculum.observe(transition([1] * 8 + [2] * 2, [True] * 10))
        report = curriculum.pop_report()
        assert report["success_rate"] == 0.8
        assert not curriculum.ready_to_advance
        curriculum.observe(transition([1] * 9 + [2], [True] * 10))
        report = curriculum.pop_report()
        assert report["episodes"] == 10
        assert report["success_rate"] == 0.9
        assert curriculum.ready_to_advance
        assert curriculum.acceptance_passed == (difficulty == 100)
        curriculum.advance()
        assert curriculum.pop_report() is None
    assert curriculum.active_obstacles == 120
    assert curriculum.best_by_stage[100]["success_rate"] == 0.9
    curriculum.observe(transition([1] * 10, [True] * 10))
    curriculum.pop_report()
    assert not curriculum.ready_to_advance
    assert curriculum.acceptance_passed


def test_aggregate_includes_all_completed_samples_without_step_mean_bias():
    curriculum = TrainingCurriculum(4, "cpu")
    first = transition([2, 0, 0, 0], [True, False, False, False])
    first.update(episode_return=torch.tensor([2.0, 999.0, 999.0, 999.0]), step=torch.tensor([2, 99, 99, 99]))
    curriculum.observe(first)
    assert curriculum.pop_report() is None
    second = transition([1, 1, 1, 1], [True] * 4)
    second.update(episode_return=torch.tensor([8.0, 12.0, 16.0, 20.0]), step=torch.tensor([4, 6, 8, 10]))
    curriculum.observe(second)
    report = curriculum.pop_report()
    assert report["episodes"] == 5
    assert report["success_rate"] == 4 / 5
    assert report["episode_return_mean"] == 58 / 5
    assert report["episode_length_mean"] == 30 / 5
    assert report["reward_per_step_mean"] == 58 / 30
    curriculum.observe(transition([2] * 4, [True] * 4))
    next_report = curriculum.pop_report()
    assert next_report["episode_return_mean"] == 0
    assert next_report["episode_length_mean"] == 1


def test_navigation_collision_wins_same_step_and_reset_clears_partial_history():
    curriculum = TrainingCurriculum(2, "cpu")
    values = {name: torch.tensor([False, False]) for name in ("bounds", "invalid", "timeout")}
    values.update(
        reached=torch.tensor([True, True]),
        collision=torch.tensor([True, False]),
        terminated=torch.tensor([True, False]),
        time_s=torch.tensor([2.0, 3.0]),
        episode_return=torch.zeros(2),
        step=torch.ones(2, dtype=torch.long),
    )
    curriculum.observe(values)
    curriculum.reset_episodes()
    curriculum.observe(transition([2, 2], [True, True]))
    report = curriculum.pop_report()
    assert report["counts"]["success"] == 0
    assert report["counts"]["collision"] == 3


@pytest.mark.parametrize("count", [0, 60, 80, 100, 120])
def test_active_mesh_selection_preserves_each_size_category(count):
    indices = active_obstacle_indices(count, 120)
    _, pool_sizes = dynamic_layout(120, 0)
    _, expected_sizes = dynamic_layout(count, 0)
    assert len(indices) == len(set(indices)) == count
    assert (pool_sizes[indices] == expected_sizes).all()
