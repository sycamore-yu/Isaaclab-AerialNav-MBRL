"""Collision/bounds continuation preserves the first outcome and reset scope."""

import pytest
import torch

from aerialnav.diffnav.termination import latch_first_event, reset_first_event, termination_mask
from aerialnav.tasks.navigation.curriculum import TrainingCurriculum


def event_terms(count, **events):
    terms = {key: torch.zeros(count, dtype=torch.bool) for key in ("reached", "collision", "bounds", "invalid")}
    for key, indices in events.items():
        terms[key][indices] = True
    return terms


def test_collision_and_bounds_can_continue_but_invalid_always_terminates():
    terms = event_terms(5, collision=[0, 3], bounds=[1, 3], invalid=[2], reached=[4])
    original = {name: value.clone() for name, value in terms.items()}
    torch.testing.assert_close(termination_mask(terms), torch.tensor([True, True, True, True, False]))
    torch.testing.assert_close(
        termination_mask(terms, collision_terminates=False), torch.tensor([False, True, True, True, False])
    )
    torch.testing.assert_close(
        termination_mask(terms, bounds_terminates=False), torch.tensor([True, False, True, True, False])
    )
    torch.testing.assert_close(
        termination_mask(terms, collision_terminates=False, bounds_terminates=False),
        torch.tensor([False, False, True, False, False]),
    )
    for name, value in terms.items():
        torch.testing.assert_close(value, original[name])


@pytest.mark.parametrize(("event", "expected"), [("collision", 2), ("bounds", 3)])
def test_failure_then_recovery_and_arrival_remains_first_failure(event, expected):
    code, elapsed = torch.zeros(1, dtype=torch.int8), torch.zeros(1)
    latch_first_event(code, elapsed, event_terms(1, **{event: [0]}), torch.tensor([1.25]))
    latch_first_event(code, elapsed, event_terms(1), torch.tensor([1.5]))
    latch_first_event(code, elapsed, event_terms(1, reached=[0]), torch.tensor([2.5]))
    torch.testing.assert_close(code, torch.tensor([expected], dtype=torch.int8))
    torch.testing.assert_close(elapsed, torch.tensor([1.25]), rtol=0, atol=0)


@pytest.mark.parametrize("event", ["collision", "bounds"])
def test_arrival_then_failure_remains_first_arrival(event):
    code, elapsed = torch.zeros(1, dtype=torch.int8), torch.zeros(1)
    latch_first_event(code, elapsed, event_terms(1, reached=[0]), torch.tensor([1.25]))
    latch_first_event(code, elapsed, event_terms(1, **{event: [0]}), torch.tensor([2.5]))
    torch.testing.assert_close(code, torch.tensor([1], dtype=torch.int8))
    torch.testing.assert_close(elapsed, torch.tensor([1.25]), rtol=0, atol=0)


def test_same_substep_keeps_original_event_priority():
    code, elapsed = torch.zeros(4, dtype=torch.int8), torch.zeros(4)
    terms = event_terms(4, reached=[0, 1, 2, 3], collision=[1, 2, 3], bounds=[0, 2, 3], invalid=[3])
    latch_first_event(code, elapsed, terms, torch.full((4,), 1.5))
    torch.testing.assert_close(code, torch.tensor([3, 2, 2, 4], dtype=torch.int8))


def test_partial_reset_clears_only_selected_episode_outcomes():
    code = torch.tensor([2, 1, 4], dtype=torch.int8)
    elapsed = torch.tensor([1.0, 2.0, 3.0])
    reset_first_event(code, elapsed, torch.tensor([0, 2]))
    torch.testing.assert_close(code, torch.tensor([0, 1, 0], dtype=torch.int8))
    torch.testing.assert_close(elapsed, torch.tensor([0.0, 2.0, 0.0]), rtol=0, atol=0)
    latch_first_event(code, elapsed, event_terms(3, reached=[0, 1, 2]), torch.full((3,), 4.0))
    torch.testing.assert_close(code, torch.tensor([1, 1, 1], dtype=torch.int8))
    torch.testing.assert_close(elapsed, torch.tensor([4.0, 2.0, 4.0]), rtol=0, atol=0)


@pytest.mark.parametrize("event", ["collision", "bounds"])
def test_continuous_failure_is_counted_only_at_natural_timeout(event):
    curriculum = TrainingCurriculum(1, "cpu", stage="pretrain_s0")
    code, elapsed = torch.zeros(1, dtype=torch.int8), torch.zeros(1)
    terms_sequence = (event_terms(1, **{event: [0]}), event_terms(1), event_terms(1, reached=[0]), event_terms(1))
    for step, terms in enumerate(terms_sequence, 1):
        latch_first_event(code, elapsed, terms, torch.tensor([step * 0.1]))
        curriculum.observe(
            {
                "first_event_code": code.clone(),
                "first_event_time_s": elapsed.clone(),
                "terminated": termination_mask(terms, collision_terminates=False, bounds_terminates=False),
                "timeout": torch.tensor([step == 4]),
                "episode_return": torch.zeros(1),
                "step": torch.tensor([step]),
            }
        )
        if step < 4:
            assert curriculum.pop_report(update=1) is None
    report = curriculum.pop_report(update=1)
    assert report["episodes"] == 1 and report["counts"][event] == 1
    assert report["success_rate"] == 0.0
    assert report[f"{event}_rate"] == 1.0
    assert report["timeout_rate"] == 0.0
    assert curriculum.pop_report(update=1) is None
