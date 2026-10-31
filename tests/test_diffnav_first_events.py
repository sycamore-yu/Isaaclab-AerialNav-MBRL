"""All DiffNav variants share geometry and first-event priority."""

import torch

from aerialnav.diffnav.termination import first_events, latch_first_event, termination_mask


def test_clearance_boundary_and_continuous_first_failure():
    state = torch.zeros(5, 13)
    state[:, 2] = 2
    target = state[:, :3].clone()
    target[:, 0] = 10
    target[3] = state[3, :3]
    state[4, 8] = float("nan")
    gaps = torch.tensor([-1.0, 0.0, 1e-6, -0.2, 1.0])
    terms = first_events(state, target, gaps)
    assert terms["collision"].tolist() == [True, True, False, True, False]
    assert termination_mask(terms).tolist() == [True, True, False, True, True]
    assert termination_mask(terms, collision_terminates=False, bounds_terminates=False).tolist() == [False] * 4 + [True]
    code, time = torch.zeros(5, dtype=torch.int8), torch.zeros(5)
    latch_first_event(code, time, terms, torch.ones(5))
    assert code.tolist() == [2, 2, 0, 2, 4]
    latch_first_event(code, time, first_events(state, state[:, :3], torch.ones(5)), torch.full((5,), 2.0))
    assert code.tolist() == [2, 2, 1, 2, 4]
