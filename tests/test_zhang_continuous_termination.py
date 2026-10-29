"""Collision continuation preserves the first episode outcome and reset scope."""

import torch

from isaaclab_navrl.tasks.navigation.curriculum import TrainingCurriculum
from isaaclab_navrl.zhang.termination import latch_first_event, reset_first_event, termination_mask


def event_terms(count, **events):
    terms = {key: torch.zeros(count, dtype=torch.bool) for key in ("reached", "collision", "bounds", "invalid")}
    for key, indices in events.items():
        terms[key][indices] = True
    return terms


def test_only_collision_becomes_nonterminal():
    terms = event_terms(5, collision=[0, 3], bounds=[1, 3], invalid=[2], reached=[4])
    torch.testing.assert_close(termination_mask(terms), torch.tensor([True, True, True, True, False]))
    torch.testing.assert_close(
        termination_mask(terms, collision_terminates=False), torch.tensor([False, True, True, True, False])
    )


def test_collision_then_arrival_remains_first_collision():
    code, elapsed = torch.zeros(1, dtype=torch.int8), torch.zeros(1)
    latch_first_event(code, elapsed, event_terms(1, collision=[0]), torch.tensor([1.25]))
    latch_first_event(code, elapsed, event_terms(1, reached=[0]), torch.tensor([2.5]))
    torch.testing.assert_close(code, torch.tensor([2], dtype=torch.int8))
    torch.testing.assert_close(elapsed, torch.tensor([1.25]), rtol=0, atol=0)


def test_arrival_then_collision_remains_first_arrival():
    code, elapsed = torch.zeros(1, dtype=torch.int8), torch.zeros(1)
    latch_first_event(code, elapsed, event_terms(1, reached=[0]), torch.tensor([1.25]))
    latch_first_event(code, elapsed, event_terms(1, collision=[0]), torch.tensor([2.5]))
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


def test_continuous_collision_counts_as_failure_at_natural_timeout():
    curriculum = TrainingCurriculum(1, "cpu", stage="pretrain_s1")
    code, elapsed = torch.zeros(1, dtype=torch.int8), torch.zeros(1)
    for step, terms in enumerate((event_terms(1, collision=[0]), event_terms(1, reached=[0]), event_terms(1)), 1):
        latch_first_event(code, elapsed, terms, torch.tensor([step * 0.1]))
        curriculum.observe({
            "first_event_code": code.clone(), "first_event_time_s": elapsed.clone(),
            "terminated": termination_mask(terms, collision_terminates=False),
            "timeout": torch.tensor([step == 3]), "episode_return": torch.zeros(1), "step": torch.tensor([step]),
        })
    report = curriculum.pop_report(update=1)
    assert report["counts"]["collision"] == 1
    assert report["success_rate"] == 0.0
    assert report["collision_rate"] == 1.0
    assert report["timeout_rate"] == 0.0
