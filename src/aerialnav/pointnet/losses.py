"""Windowed velocity, approach-weighted clearance, acceleration and jerk costs."""

import torch
from torch.nn import functional as F

from aerialnav.diffnav.losses import navigation_height_loss


def causal_velocity_average(velocity, episode, window, prefix=None):
    """Average up to W samples of this episode, including short new episodes."""
    previous = 0
    if prefix is not None:
        previous = prefix["velocity"].shape[0]
        velocity = torch.cat((prefix["velocity"].detach(), velocity))
        episode = torch.cat((prefix["episode_id"], episode))
    steps, envs = episode.shape
    indices = torch.arange(steps, device=velocity.device)[:, None].expand(steps, envs)
    reset = torch.cat((torch.ones_like(episode[:1], dtype=torch.bool), episode[1:] != episode[:-1]))
    starts = torch.where(reset, indices, 0).cummax(0).values
    starts = torch.maximum(starts, indices - window + 1)
    cumulative = torch.cat((velocity.new_zeros(1, envs, 3), velocity.cumsum(0)))
    rows = torch.arange(envs, device=velocity.device)[None]
    average = (cumulative[1:] - cumulative[starts, rows]) / (indices - starts + 1)[..., None]
    keep = max(window - 1, 0)
    memory = {
        "velocity": velocity[-keep:].detach() if keep else velocity[:0].detach(),
        "episode_id": episode[-keep:].detach() if keep else episode[:0].detach(),
    }
    return average[previous:], memory


def pointcloud_transition_costs(history, cfg, velocity_history=None):
    velocity = torch.stack(history["velocity"])
    episode = torch.stack(history["episode_id"])
    averaged, memory = causal_velocity_average(velocity, episode, cfg.velocity_window_steps, velocity_history)
    target = torch.stack(history["target_velocity"])
    error = target - averaged
    norm = torch.linalg.vector_norm(target, dim=-1) - torch.linalg.vector_norm(averaged, dim=-1)
    velocity_cost = cfg.velocity_norm_weight * F.huber_loss(
        norm, torch.zeros_like(norm), delta=cfg.huber_delta, reduction="none"
    )
    velocity_cost += cfg.velocity_axis_weight * F.huber_loss(
        error, torch.zeros_like(error), delta=cfg.huber_delta, reduction="none"
    ).sum(-1)
    clearance = torch.stack(history["obstacle_signed_distance"]) - torch.stack(history["margin"])[..., None]
    closing = (-clearance.diff(dim=-1) * (9 / cfg.policy_dt)).clamp_min(0).detach()
    command = torch.stack(history["command"])
    jerk = torch.linalg.vector_norm((command - torch.stack(history["previous_command"])) / cfg.policy_dt, dim=-1)
    valid = torch.stack(history["episode_step"]) > 1
    # Each episode segment has its own jerk variance; environments never share statistics.
    reset = torch.cat((torch.ones_like(episode[:1], dtype=torch.bool), episode[1:] != episode[:-1]))
    groups = reset.long().cumsum(0) - 1
    groups = groups * episode.shape[1] + torch.arange(episode.shape[1], device=episode.device)[None]
    flat_groups = groups.flatten()
    size = episode.numel()
    counts = jerk.new_zeros(size).scatter_add(0, flat_groups, valid.to(jerk).flatten())
    sums = jerk.new_zeros(size).scatter_add(0, flat_groups, (jerk * valid).flatten())
    mean = (sums / counts.clamp_min(1))[groups]
    jerk_cost = (cfg.jerk_mean_weight * jerk + cfg.jerk_variance_weight * (jerk - mean).square()) * valid
    terms = {
        "velocity": velocity_cost,
        "velocity_prediction": torch.zeros_like(velocity_cost),
        "obstacle_avoidance": (closing * (1 - clearance[..., 1:]).relu().square()).mean(-1),
        "collision": (closing * F.softplus(-32 * clearance[..., 1:])).mean(-1),
        "acceleration": command.square().sum(-1),
        "jerk": jerk_cost,
        "height_boundary": navigation_height_loss(
            torch.stack(history["position"]), torch.stack(history["height_range"])
        ),
    }
    weights = {
        "velocity": cfg.coef_v,
        "velocity_prediction": 0,
        "obstacle_avoidance": cfg.coef_obj_avoidance,
        "collision": cfg.coef_collide,
        "acceleration": cfg.coef_d_acc,
        "jerk": cfg.coef_d_jerk,
        "height_boundary": cfg.coef_height_boundary,
    }
    cost = sum(weights[name] * value for name, value in terms.items())
    losses = {name: value.mean() for name, value in terms.items()}
    losses.update(
        total=cost.mean(),
        velocity_valid_samples=velocity.new_tensor(episode.numel()),
        velocity_sample_fraction=velocity.new_tensor(1.0),
    )
    return losses, memory, cost
