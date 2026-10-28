"""Approved source losses; all time windows are measured in policy steps."""

import torch
from torch.nn import functional as F


def masked_mean(value, valid):
    valid = valid.to(value.dtype)
    return (value * valid).sum() / valid.sum().clamp_min(1)


def velocity_window_loss(velocity, target, episode_ids, window, current_start=0):
    """Match source cumsum[W:]-cumsum[:-W] against target[1:1-W].

    A window may use detached preceding-update history. It must end in this
    update and every sample must belong to one episode. Invalid windows produce
    zero contribution rather than mixing velocities from different resets.
    """
    if velocity.shape[0] <= window:
        return velocity.sum() * 0
    cumulative = velocity.cumsum(dim=0)
    average = (cumulative[window:] - cumulative[:-window]) / window
    target_window = target[1:1 - window] if window > 1 else target[1:]
    ids = episode_ids.unfold(0, window, 1)[1:]
    valid = (ids == ids[..., :1]).all(dim=-1)
    endpoints = torch.arange(window, velocity.shape[0], device=velocity.device)
    valid &= (endpoints >= current_start)[:, None]
    delta = (average - target_window).norm(dim=-1)
    return masked_mean(F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none"), valid)


def zhang_losses(history, cfg, velocity_history=None):
    velocity = torch.stack(history["velocity"])
    target = torch.stack(history["target_velocity"])
    episode = torch.stack(history["episode_id"])
    prefix_length = 0
    if velocity_history is not None and velocity_history["velocity"].shape[0]:
        prefix_length = velocity_history["velocity"].shape[0]
        all_velocity = torch.cat((velocity_history["velocity"], velocity), dim=0)
        all_target = torch.cat((velocity_history["target_velocity"], target), dim=0)
        all_episode = torch.cat((velocity_history["episode_id"], episode), dim=0)
    else:
        all_velocity, all_target, all_episode = velocity, target, episode
    distance = torch.stack(history["obstacle_vector"]).norm(dim=-1) - torch.stack(history["margin"])[:, :, None]
    # Ten source look-ahead points include both endpoints: dt_sample = policy_dt / 9.
    approach = (-distance.diff(dim=2) * ((distance.shape[2] - 1) / cfg.policy_dt)).clamp_min(1).detach()
    avoidance = (approach * (1 - distance[:, :, 1:]).relu().square()).mean()
    collision = (F.softplus(-32 * distance[:, :, 1:]) * approach).mean()
    command = torch.stack(history["command"])
    command_previous = torch.stack(history["previous_command"])
    jerk_squared = ((command - command_previous) / cfg.policy_dt).square().sum(dim=-1)
    losses = {
        "velocity": velocity_window_loss(all_velocity, all_target, all_episode,
                                         cfg.velocity_window_steps, prefix_length),
        "velocity_prediction": F.mse_loss(torch.stack(history["velocity_prediction"]),
                                           torch.stack(history["observed_velocity"])),
        "obstacle_avoidance": avoidance,
        "collision": collision,
        "acceleration": command.square().sum(dim=-1).mean(),
        "jerk": jerk_squared.mean(),
    }
    total = (cfg.coef_v * losses["velocity"] + cfg.coef_v_pred * losses["velocity_prediction"]
             + cfg.coef_obj_avoidance * losses["obstacle_avoidance"] + cfg.coef_collide * losses["collision"]
             + cfg.coef_d_acc * losses["acceleration"] + cfg.coef_d_jerk * losses["jerk"])
    losses["total"] = total
    keep = cfg.velocity_window_steps
    detached_history = {"velocity": all_velocity[-keep:].detach(),
                        "target_velocity": all_target[-keep:].detach(),
                        "episode_id": all_episode[-keep:].detach()}
    return losses, detached_history
