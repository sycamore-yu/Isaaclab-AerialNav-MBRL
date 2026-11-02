"""DiffNav first-event and termination rules shared by all variants."""

import torch

from aerialnav.training import MAP_SIZE


def termination_mask(terms, *, collision_terminates=True, bounds_terminates=True):
    """Failure termination is separate from arrival-wait and episode truncation."""
    terminated = terms["invalid"].clone()
    if bounds_terminates:
        terminated = terminated | terms["bounds"]
    if collision_terminates:
        terminated = terminated | terms["collision"]
    return terminated


@torch.no_grad()
def record_arrival(arrived, arrive_time, reached, elapsed_s):
    """Keep the first entry time even when a drone leaves and returns."""
    first = reached & ~arrived
    arrive_time.copy_(torch.where(first, elapsed_s, arrive_time))
    arrived |= reached


def arrival_wait_timeout(in_goal, arrive_time, elapsed_s, wait_before_truncate):
    return in_goal & (elapsed_s > arrive_time + wait_before_truncate)


@torch.no_grad()
def terminal_outcomes(terms, done, episode_failure):
    """Score the final goal predicate; preserve failures in continuous rollouts."""
    failure = torch.zeros_like(episode_failure)
    failure[terms["bounds"]] = 3
    failure[terms["collision"]] = 2
    failure[terms["invalid"]] = 4
    first_failure = (episode_failure == 0) & (failure != 0)
    episode_failure.copy_(torch.where(first_failure, failure, episode_failure))
    code = torch.where(terms["reached"], 1, 5)
    code = torch.where(episode_failure != 0, episode_failure, code)
    return torch.where(done, code, 0)


@torch.no_grad()
def latch_first_event(event_code, event_time, terms, elapsed_s):
    """Record the first episode event, with the original same-substep priority."""
    code = torch.zeros_like(event_code)
    code[terms["reached"]] = 1
    code[terms["bounds"]] = 3
    code[terms["collision"]] = 2
    code[terms["invalid"]] = 4
    first = (event_code == 0) & (code != 0)
    event_code.copy_(torch.where(first, code, event_code))
    event_time.copy_(torch.where(first, elapsed_s, event_time))


@torch.no_grad()
def reset_first_event(event_code, event_time, env_ids):
    """Clear only the episodes being reset."""
    env_ids = torch.as_tensor(env_ids, device=event_code.device, dtype=torch.long)
    event_code.index_fill_(0, env_ids, 0)
    event_time.index_fill_(0, env_ids, 0)


def first_events(root_state, target, collision_gap, *, height_bounds_reset=True):
    """Shared DiffNav policy-boundary and physics-substep sphere predicates."""
    position = root_state[:, :3]
    return {
        "collision": collision_gap <= 0,
        "bounds": (position[:, 0].abs() > MAP_SIZE[0] / 2)
        | (position[:, 1].abs() > MAP_SIZE[1] / 2)
        | (((position[:, 2] < 0.2) | (position[:, 2] > 4.0)) & height_bounds_reset),
        "reached": (target - position).norm(dim=-1) < 0.5,
        "invalid": ~torch.isfinite(root_state).all(dim=-1),
    }
