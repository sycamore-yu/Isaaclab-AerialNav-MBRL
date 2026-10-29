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


def velocity_step_loss(velocity, target, *, reduction="mean"):
    """Match every post-step velocity to its own pre-step target, including done."""
    delta = (velocity - target).norm(dim=-1)
    return F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction=reduction)


def height_boundary_loss(position, bounds=(0.2, 4.0), margin=0.5, scale=0.5, *, reduction="mean"):
    """C1 near-boundary penalty with a nonzero inward gradient far outside.

    SmoothL1 is quadratic for a violation smaller than ``scale`` and linear
    beyond it. Its derivative is zero on the interior band and continuous at
    both the activation boundary and the quadratic-to-linear shoulder.
    """
    height = position[..., 2]
    violation = (bounds[0] + margin - height).relu() + (height - bounds[1] + margin).relu()
    scaled = violation / scale
    return F.smooth_l1_loss(scaled, torch.zeros_like(scaled), reduction=reduction)


def zhang_transition_costs(history, cfg, velocity_history=None):
    """Return scalar losses, detached window history and actual ``cost[T, N]``.

    All terms keep their original transition timing: velocity and altitude use
    post-step states; velocity prediction uses the pre-step observation; the
    command/jerk and obstacle look-ahead terms belong to that same transition.
    A full velocity window is charged only at its current-update endpoint. Its
    cost is scaled by ``T*N/valid_count``, preserving the valid-window mean.
    Cached prefix samples may complete a window, but receive neither a new
    score nor a gradient. Windows crossing an episode boundary remain masked.
    The independent per-step velocity mode instead scores every transition,
    including terminal velocities, and needs no window cache or sample scaling.
    It does not change the variant's obstacle geometry or any other cost.

    Continuous rollouts also require ``ground_signed_distance`` at the ten
    original pre-step look-ahead queries: query_z minus 0.01, before subtracting
    the drone margin. Only queries below that plane replace unsigned clearance
    with signed clearance. This permits ground recovery without changing the
    above-ground loss or the other variants. General nonconvex static-object
    penetration still uses the original unsigned nearest-surface geometry.
    """
    if cfg.variant not in ("step", "continuous", "dcol"):
        raise ValueError(f"Unsupported Zhang loss variant: {cfg.variant}")
    if cfg.variant == "continuous" and "ground_signed_distance" not in history:
        raise ValueError("continuous requires ground_signed_distance from the pre-step look-ahead queries")
    velocity = torch.stack(history["velocity"])
    target = torch.stack(history["target_velocity"])
    episode = torch.stack(history["episode_id"])
    timesteps, num_envs = velocity.shape[:2]
    detached_history = None
    if cfg.resolved_velocity_loss_mode == "per_step":
        velocity_cost = velocity_step_loss(velocity, target, reduction="none")
        valid_samples = velocity.new_tensor(timesteps * num_envs)
    else:
        prefix_length = 0
        if velocity_history is not None and velocity_history["velocity"].shape[0]:
            prefix_length = velocity_history["velocity"].shape[0]
            all_velocity = torch.cat((velocity_history["velocity"].detach(), velocity), dim=0)
            all_target = torch.cat((velocity_history["target_velocity"].detach(), target), dim=0)
            all_episode = torch.cat((velocity_history["episode_id"].detach(), episode), dim=0)
        else:
            all_velocity, all_target, all_episode = velocity, target, episode
        window = cfg.velocity_window_steps
        values, valid = _velocity_window_samples(
            all_velocity, all_target, all_episode, window, prefix_length
        )
        valid_samples = valid.sum()
        # The source omits its first velocity sample, so its first full-window
        # score is at index W. Preserve this alignment, then remove old prefix
        # endpoints instead of spreading their costs over the current rollout.
        padding = values.new_zeros(min(window, all_velocity.shape[0]), num_envs)
        endpoint_cost = torch.cat((padding, values * valid), dim=0)[prefix_length:]
        velocity_cost = endpoint_cost * (timesteps * num_envs / valid_samples.clamp_min(1))
        detached_history = {
            "velocity": all_velocity[-window:].detach(),
            "target_velocity": all_target[-window:].detach(),
            "episode_id": all_episode[-window:].detach(),
        }
    margin = torch.stack(history["margin"])[:, :, None]
    distance = torch.stack(history["obstacle_vector"]).norm(dim=-1) - margin
    if cfg.variant == "continuous":
        ground = torch.stack(history["ground_signed_distance"])
        if ground.shape != distance.shape:
            raise ValueError("ground_signed_distance must match the [T, N, 10] obstacle look-ahead samples")
        distance = torch.where(ground < 0, ground - margin, distance)
    # Ten source look-ahead points include both endpoints: dt_sample = policy_dt / 9.
    approach = (-distance.diff(dim=2) * ((distance.shape[2] - 1) / cfg.policy_dt)).clamp_min(1).detach()
    collision_gap = torch.stack(history["dcol_gap"]) if cfg.variant == "dcol" else distance[:, :, 1:]
    command = torch.stack(history["command"])
    command_previous = torch.stack(history["previous_command"])
    terms = {
        "velocity": velocity_cost,
        "velocity_prediction": F.mse_loss(torch.stack(history["velocity_prediction"]),
                                           torch.stack(history["observed_velocity"]), reduction="none").mean(dim=-1),
        "obstacle_avoidance": (approach * (1 - distance[:, :, 1:]).relu().square()).mean(dim=-1),
        "collision": (F.softplus(-32 * collision_gap) * approach).mean(dim=-1),
        "acceleration": command.square().sum(dim=-1),
        "jerk": ((command - command_previous) / cfg.policy_dt).square().sum(dim=-1),
        "height_boundary": height_boundary_loss(
            torch.stack(history["position"]), cfg.height_bounds, cfg.boundary_margin, cfg.boundary_scale,
            reduction="none",
        ),
    }
    cost = (cfg.coef_v * terms["velocity"] + cfg.coef_v_pred * terms["velocity_prediction"]
            + cfg.coef_obj_avoidance * terms["obstacle_avoidance"] + cfg.coef_collide * terms["collision"]
            + cfg.coef_d_acc * terms["acceleration"] + cfg.coef_d_jerk * terms["jerk"]
            + cfg.coef_height_boundary * terms["height_boundary"])
    losses = {name: value.mean() for name, value in terms.items()}
    losses["total"] = cost.mean()
    losses["velocity_valid_samples"] = valid_samples.detach()
    losses["velocity_sample_fraction"] = valid_samples.detach() / (timesteps * num_envs)
    # Preserve the inert log key for existing readers of the historical runs.
    losses["velocity_short_terminal_samples"] = velocity.new_zeros(())
    return losses, detached_history, cost


def zhang_losses(history, cfg, velocity_history=None):
    """Compatibility wrapper for the existing two-return BPTT interface."""
    losses, detached_history, _ = zhang_transition_costs(history, cfg, velocity_history)
    return losses, detached_history
