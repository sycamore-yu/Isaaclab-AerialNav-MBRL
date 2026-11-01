"""DiffAero arrival wait gates on current goal occupancy and the first entry time."""

import pytest
import torch

from aerialnav.diffnav.termination import arrival_wait_timeout, record_arrival, terminal_outcomes, termination_mask
from aerialnav.evaluation import EpisodeEvaluation
from aerialnav.tasks.navigation.curriculum import TrainingCurriculum


def terms(reached, *, collision=False, bounds=False, invalid=False):
    return {
        key: torch.tensor([value])
        for key, value in dict(reached=reached, collision=collision, bounds=bounds, invalid=invalid).items()
    }


def test_first_arrival_time_is_not_restarted_when_leaving_and_returning():
    arrived, arrive_time = torch.zeros(1, dtype=torch.bool), torch.zeros(1)
    for time, reached in [(1.0, True), (2.0, False), (4.0, True)]:
        record_arrival(arrived, arrive_time, torch.tensor([reached]), torch.tensor([time]))
        assert arrived.item() and arrive_time.item() == 1.0
    assert not arrival_wait_timeout(arrived, arrive_time, torch.tensor([5.99]), 5.0).item()
    assert not arrival_wait_timeout(torch.tensor([True]), arrive_time, torch.tensor([6.0]), 5.0).item()
    assert arrival_wait_timeout(torch.tensor([True]), arrive_time, torch.tensor([6.05]), 5.0).item()
    assert not arrival_wait_timeout(torch.tensor([False]), arrive_time, torch.tensor([7.0]), 5.0).item()
    assert arrival_wait_timeout(torch.tensor([True]), arrive_time, torch.tensor([8.0]), 5.0).item()
    assert not termination_mask(terms(True)).item()
    assert not arrival_wait_timeout(torch.tensor([False]), torch.zeros(1), torch.tensor([100.0]), 5.0).item()


@pytest.mark.parametrize(
    ("reached", "collision", "bounds", "invalid", "expected"),
    [
        (True, False, False, False, 1),
        (False, False, False, False, 5),
        (True, True, False, False, 2),
        (True, True, True, False, 2),
        (True, True, True, True, 4),
        (True, False, True, False, 3),
    ],
)
def test_final_goal_predicate_and_failure_priority(reached, collision, bounds, invalid, expected):
    flags = terms(reached, collision=collision, bounds=bounds, invalid=invalid)
    history = torch.zeros(1, dtype=torch.long)
    assert terminal_outcomes(flags, torch.tensor([True]), history).item() == expected
    assert terminal_outcomes(flags, torch.tensor([False]), history).item() == 0


def test_continuous_failure_before_arrival_cannot_become_success():
    history = torch.zeros(1, dtype=torch.long)
    terminal_outcomes(terms(False, collision=True), torch.tensor([False]), history)
    assert terminal_outcomes(terms(True), torch.tensor([True]), history).item() == 2


@pytest.mark.parametrize("episode_end_time", [3.0, 35.0])
def test_training_and_evaluation_wait_for_final_result(episode_end_time):
    curriculum = TrainingCurriculum(2, "cpu", stage="pretrain_s1")
    evaluation = EpisodeEvaluation([[0, 0, 2]] * 2, [[1, 0, 2]] * 2, 17)
    transition = dict(
        reached=torch.tensor([True, True]),
        collision=torch.zeros(2, dtype=torch.bool),
        bounds=torch.zeros(2, dtype=torch.bool),
        invalid=torch.zeros(2, dtype=torch.bool),
        terminated=torch.zeros(2, dtype=torch.bool),
        timeout=torch.zeros(2, dtype=torch.bool),
        first_event_code=torch.ones(2, dtype=torch.int8),
        first_event_time_s=torch.ones(2),
        arrived=torch.ones(2, dtype=torch.bool),
        arrive_time=torch.ones(2),
        terminal_event_code=torch.zeros(2, dtype=torch.long),
        time_s=torch.ones(2),
        step=torch.full((2,), 20),
        episode_return=torch.zeros(2),
    )
    curriculum.observe(transition)
    evaluation.update(transition)
    assert curriculum.pop_report() is None and evaluation.rows == [None, None]
    # The existing episode time limit scores final occupancy, including an early timeout.
    transition.update(
        reached=torch.tensor([False, True]),
        timeout=torch.ones(2, dtype=torch.bool),
        terminal_event_code=torch.tensor([5, 1]),
        time_s=torch.full((2,), episode_end_time),
        step=torch.full((2,), int(episode_end_time * 20)),
    )
    curriculum.observe(transition)
    evaluation.update(transition)
    report = curriculum.pop_report(update=1)
    assert report["counts"]["success"] == report["counts"]["timeout"] == 1
    assert report["success_time_mean_s"] == 1.0
    assert [r["outcome"] for r in evaluation.rows] == ["timeout", "success"]
    assert evaluation.summary()["success_rate"] == 0.5
    assert evaluation.summary()["success_time_mean_s"] == 1.0
    assert evaluation.rows[1]["arrive_time"] == 1.0
    assert evaluation.rows[1]["time_s"] == episode_end_time
    assert curriculum.pop_report() is None


def test_collision_during_wait_overrides_first_arrival_in_both_consumers():
    flags = terms(True, collision=True)
    code = terminal_outcomes(flags, torch.tensor([True]), torch.zeros(1, dtype=torch.long))
    transition = dict(
        flags,
        terminal_event_code=code,
        first_event_code=torch.tensor([1]),
        first_event_time_s=torch.tensor([1.0]),
        arrived=torch.tensor([True]),
        arrive_time=torch.tensor([1.0]),
        time_s=torch.tensor([2.0]),
        step=torch.tensor([40]),
        terminated=torch.tensor([True]),
        timeout=torch.tensor([False]),
        episode_return=torch.zeros(1),
    )
    evaluation = EpisodeEvaluation([[0, 0, 2]], [[1, 0, 2]], 17)
    evaluation.update(transition)
    curriculum = TrainingCurriculum(1, "cpu", stage="pretrain_s1")
    curriculum.observe(transition)
    assert evaluation.rows[0]["outcome"] == "collision"
    assert curriculum.pop_report()["counts"]["collision"] == 1


def test_wait_default_and_validation():
    from aerialnav.tasks.navigation.config.hummingbird.env_cfg import DiffNavEnvCfg

    cfg = DiffNavEnvCfg()
    assert cfg.wait_before_truncate == 10.0
    first_arrival = torch.tensor([1.0])
    assert cfg.wait_before_truncate > 150 * 0.05
    assert not arrival_wait_timeout(torch.tensor([True]), first_arrival, torch.tensor([8.5]), cfg.wait_before_truncate)
    assert not arrival_wait_timeout(torch.tensor([True]), first_arrival, torch.tensor([11.0]), cfg.wait_before_truncate)
    assert arrival_wait_timeout(torch.tensor([True]), first_arrival, torch.tensor([11.05]), cfg.wait_before_truncate)
    for invalid in (-1.0, float("nan"), float("inf")):
        cfg.wait_before_truncate = invalid
        with pytest.raises(ValueError, match="wait_before_truncate"):
            cfg.configure_scene()
