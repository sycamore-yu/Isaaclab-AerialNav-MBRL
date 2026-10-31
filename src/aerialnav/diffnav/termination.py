"""DiffNav first-event and termination rules shared by all variants."""

import torch


def termination_mask(terms, *, collision_terminates=True, bounds_terminates=True):
    """Keep invalid states terminal; timeout remains the environment's own flag."""
    terminated = terms["invalid"].clone()
    if bounds_terminates:
        terminated = terminated | terms["bounds"]
    if collision_terminates:
        terminated = terminated | terms["collision"]
    return terminated


@torch.no_grad()
def latch_first_event(event_code, event_time, terms, elapsed_s):
    """Record the first episode event, with the original same-substep priority."""
    code = torch.zeros_like(event_code)
    code[terms["reached"]] = 1
    code[terms["bounds"]] = 3
    code[terms["collision"]] = 2
    code[terms["invalid"]] = 4
    first = (event_code == 0) & (code != 0)
    event_code[first] = code[first]
    event_time[first] = elapsed_s[first]


@torch.no_grad()
def reset_first_event(event_code, event_time, env_ids):
    """Clear only the episodes being reset."""
    event_code[env_ids] = 0
    event_time[env_ids] = 0


def first_events(root_state, target, collision_gap):
    """Shared DiffNav policy-boundary and physics-substep sphere predicates."""
    position = root_state[:, :3]
    return {
        "collision": collision_gap <= 0,
        "bounds": (position[:, 2] < 0.2) | (position[:, 2] > 4.0),
        "reached": (target - position).norm(dim=-1) < 0.5,
        "invalid": ~torch.isfinite(root_state).all(dim=-1),
    }
