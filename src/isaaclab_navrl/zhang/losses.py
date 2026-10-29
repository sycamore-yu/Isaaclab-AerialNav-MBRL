"""Zhang losses with the approved velocity variants and altitude boundary term."""

import torch
from torch.nn import functional as F


def masked_mean(value, valid):
    valid = valid.to(value.dtype)
    return (value * valid).sum() / valid.sum().clamp_min(1)


def _velocity_window_samples(velocity, target, episode_ids, window, current_start=0):
    if velocity.shape[0] <= window:
        values = velocity.sum(dim=-1)[:0]
        return values, torch.zeros_like(values, dtype=torch.bool)
    cumulative = velocity.cumsum(dim=0)
    average = (cumulative[window:] - cumulative[:-window]) / window
    target_window = target[1:1 - window] if window > 1 else target[1:]
    ids = episode_ids.unfold(0, window, 1)[1:]
    valid = (ids == ids[..., :1]).all(dim=-1)
    endpoints = torch.arange(window, velocity.shape[0], device=velocity.device)
    valid &= (endpoints >= current_start)[:, None]
    delta = (average - target_window).norm(dim=-1)
    return F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none"), valid


def velocity_window_loss(velocity, target, episode_ids, window, current_start=0):
    """Keep the source forward window and exclude samples from different episodes.

    Preceding-update history is detached by the caller. Each scored window must
    end in this update. The source's first omitted sample remains unchanged.
    """
    values, valid = _velocity_window_samples(velocity, target, episode_ids, window, current_start)
    return masked_mean(values, valid)


def velocity_step_loss(velocity, target):
    """Match every post-step velocity to its own pre-step target, including done."""
    delta = (velocity - target).norm(dim=-1)
    return F.smooth_l1_loss(delta, torch.zeros_like(delta))


def _short_collision_samples(velocity, target, episode_ids, window, done, collision, current_start, full_valid):
    time = torch.arange(velocity.shape[0], device=velocity.device)[:, None]
    changed = torch.cat((torch.ones_like(episode_ids[:1], dtype=torch.bool),
                         episode_ids[1:] != episode_ids[:-1]), dim=0)
    start = torch.where(changed, time, 0).cummax(dim=0).values
    length = time - start + 1
    gather_start = start[..., None].expand(-1, -1, velocity.shape[-1])
    cumulative = torch.cat((torch.zeros_like(velocity[:1]), velocity.cumsum(dim=0)), dim=0)
    average = (cumulative[1:] - cumulative.gather(0, gather_start)) / length[..., None]
    start_target = target.gather(0, gather_start)
    delta = (average - start_target).norm(dim=-1)
    full_scored = torch.zeros_like(done)
    full_scored[window:] = full_valid
    # The source omits sample zero: the first episode can reach W samples without
    # producing a full-window score. Include that terminal once as well.
    valid = done & collision & (length <= window) & ~full_scored & (time >= current_start)
    return F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none"), valid


def truncated_velocity_window_loss(velocity, target, episode_ids, window, done, collision, current_start=0):
    """Add one short mean-velocity sample when a collision resets an episode.

    Do not rescore an endpoint already covered by the full-window rule. The
    first episode's exactly-W terminal is included because the source window
    omits sample zero. Sum both sample sets before dividing, so each scored
    window has the same weight. Keep a prefix of ``window`` samples across updates.
    """
    values, valid = _velocity_window_samples(velocity, target, episode_ids, window, current_start)
    short_values, short_valid = _short_collision_samples(
        velocity, target, episode_ids, window, done, collision, current_start, valid
    )
    count = valid.sum() + short_valid.sum()
    return ((values * valid).sum() + (short_values * short_valid).sum()) / count.clamp_min(1)


def height_boundary_loss(position, bounds=(0.2, 4.0), margin=0.5, scale=0.5):
    """Bounded C1 penalty outside the interior altitude band, including terminals."""
    height = position[..., 2]
    violation = (bounds[0] + margin - height).relu() + (height - bounds[1] + margin).relu()
    return -torch.expm1(-(violation / scale).square()).mean()


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
    collision_gap = torch.stack(history["dcol_gap"]) if cfg.variant == "dcol" else distance[:, :, 1:]
    collision = (F.softplus(-32 * collision_gap) * approach).mean()
    command = torch.stack(history["command"])
    command_previous = torch.stack(history["previous_command"])
    jerk_squared = ((command - command_previous) / cfg.policy_dt).square().sum(dim=-1)
    short_samples = velocity.new_zeros(())
    if cfg.variant == "step":
        velocity_loss = velocity_step_loss(velocity, target)
        valid_samples = velocity.new_tensor(velocity.shape[0] * velocity.shape[1])
    else:
        values, valid = _velocity_window_samples(
            all_velocity, all_target, all_episode, cfg.velocity_window_steps, prefix_length
        )
        total_velocity = (values * valid).sum()
        valid_samples = valid.sum()
        if cfg.variant == "truncated_window":
            prefix_done = torch.zeros_like(all_episode[:prefix_length], dtype=torch.bool)
            done = torch.cat((prefix_done, torch.stack(history["done"])), dim=0)
            collided = torch.cat((prefix_done, torch.stack(history["collision"])), dim=0)
            short_values, short_valid = _short_collision_samples(
                all_velocity, all_target, all_episode, cfg.velocity_window_steps,
                done, collided, prefix_length, valid,
            )
            total_velocity = total_velocity + (short_values * short_valid).sum()
            short_samples = short_valid.sum()
            valid_samples = valid_samples + short_samples
        velocity_loss = total_velocity / valid_samples.clamp_min(1)
    losses = {
        "velocity": velocity_loss,
        "velocity_prediction": F.mse_loss(torch.stack(history["velocity_prediction"]),
                                           torch.stack(history["observed_velocity"])),
        "obstacle_avoidance": avoidance,
        "collision": collision,
        "acceleration": command.square().sum(dim=-1).mean(),
        "jerk": jerk_squared.mean(),
        "height_boundary": height_boundary_loss(
            torch.stack(history["position"]), cfg.height_bounds, cfg.boundary_margin, cfg.boundary_scale
        ),
    }
    total = (cfg.coef_v * losses["velocity"] + cfg.coef_v_pred * losses["velocity_prediction"]
             + cfg.coef_obj_avoidance * losses["obstacle_avoidance"] + cfg.coef_collide * losses["collision"]
             + cfg.coef_d_acc * losses["acceleration"] + cfg.coef_d_jerk * losses["jerk"]
             + cfg.coef_height_boundary * losses["height_boundary"])
    losses["total"] = total
    losses["velocity_valid_samples"] = valid_samples.detach()
    losses["velocity_sample_fraction"] = valid_samples.detach() / (velocity.shape[0] * velocity.shape[1])
    losses["velocity_short_terminal_samples"] = short_samples.detach()
    if cfg.variant == "step":
        return losses, None
    keep = cfg.velocity_window_steps
    detached_history = {"velocity": all_velocity[-keep:].detach(),
                        "target_velocity": all_target[-keep:].detach(),
                        "episode_id": all_episode[-keep:].detach()}
    return losses, detached_history
