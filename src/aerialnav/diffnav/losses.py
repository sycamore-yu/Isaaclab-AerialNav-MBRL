"""DiffNav transition costs with explicit velocity, height and signed-geometry inputs."""

import torch
from torch.nn import functional as F


class VelocityEMA:
    """Episode-local DiffAero velocity memory with a separate random stream."""

    def __init__(self, zero, seed):
        self.value = torch.zeros_like(zero)
        self.generator = torch.Generator(device=zero.device).manual_seed(seed)
        self.alpha = self._sample(zero.shape[0])

    def _sample(self, count):
        return 0.08 + 0.04 * torch.rand(count, 1, device=self.value.device, generator=self.generator)

    def update(self, velocity):
        self.value = torch.lerp(self.value, velocity, self.alpha)
        return self.value

    def reset(self, done):
        self.value = torch.where(done[:, None], torch.zeros_like(self.value), self.value)
        self.alpha = torch.where(done[:, None], self._sample(len(done)), self.alpha)

    def detach(self):
        self.value = self.value.detach()


def relative_avoidance(relative_velocity, normal, clearance):
    """Outward normals: positive closing means approaching; lateral is XY only."""
    closing = -(relative_velocity * normal).sum(-1)
    normal_xy = F.normalize(normal[..., :2], dim=-1)
    velocity_xy = relative_velocity[..., :2]
    lateral = (velocity_xy - (velocity_xy * normal_xy).sum(-1, keepdim=True) * normal_xy).norm(dim=-1)
    # Detach risk so the lateral reward cannot attract the drone toward obstacles.
    danger = (-clearance.detach().clamp_min(0)).exp()
    penalty, index = (closing.relu() * danger).max(dim=-1)
    # Ground/ceiling normals must never produce a horizontal-motion reward.
    reward = torch.where((closing > 0) & (normal[..., :2].norm(dim=-1) > 1e-6), lateral * danger, 0.0)
    return closing, penalty - 0.5 * reward.gather(-1, index[..., None]).squeeze(-1)


def risk_speed_scale(closing, clearance, max_speed, floor):
    """Detached risk changes the target, not its gradient through the geometry."""
    risk = (closing.detach().clamp_min(0) / max_speed.clamp_min(1e-6)) * (-clearance.detach().clamp_min(0)).exp()
    return 1 - (1 - floor) * risk.amax(dim=-1, keepdim=True).clamp(0, 1)


def masked_mean(value, valid):
    valid = valid.to(value.dtype)
    return (value * valid).sum() / valid.sum().clamp_min(1)


def _velocity_window_samples(velocity, target, episode_ids, window, current_start=0):
    if velocity.shape[0] <= window:
        values = velocity.sum(dim=-1)[:0]
        return values, torch.zeros_like(values, dtype=torch.bool)
    cumulative = velocity.cumsum(dim=0)
    average = (cumulative[window:] - cumulative[:-window]) / window
    target_window = target[1 : 1 - window] if window > 1 else target[1:]
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


def _truncated_terminal_samples(velocity, target, episode, window, current_start, full_valid, history):
    """Complete eligible terminal windows, using their true episode length.

    The returned samples belong only to the current rollout. A left-truncated
    prefix may begin inside a long episode, but a completed <=W episode must
    have every one of its samples visible. Targets use the episode's first
    pre-step reference; velocities include its policy-step terminal state.
    """
    names = ("done", "collision", "bounds", "timeout", "invalid", "episode_step")
    missing = [name for name in names if name not in history]
    if missing:
        raise ValueError(f"truncated_window requires terminal history fields: {missing}")
    fields = {name: torch.stack(history[name]) for name in names}
    shape = velocity[current_start:].shape[:2]
    if any(value.shape != shape for value in fields.values()):
        raise ValueError("truncated_window terminal history must match current [T, N] samples")
    if any(fields[name].dtype != torch.bool for name in names[:-1]) or fields["episode_step"].dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("truncated_window requires boolean events and integer episode_step")
    time = torch.arange(velocity.shape[0], device=velocity.device)[:, None]
    changed = torch.cat((torch.ones_like(episode[:1], dtype=torch.bool), episode[1:] != episode[:-1]), dim=0)
    start = torch.where(changed, time, 0).cummax(dim=0).values[current_start:]
    visible_length = time[current_start:] - start + 1
    actual_length = fields["episode_step"]
    full_padding = torch.zeros(
        min(window, velocity.shape[0]), velocity.shape[1], dtype=torch.bool, device=velocity.device
    )
    full_scored = torch.cat((full_padding, full_valid), dim=0)[current_start:]
    eligible = fields["done"] & (fields["collision"] | fields["bounds"] | fields["timeout"]) & ~fields["invalid"]
    valid = eligible & (actual_length <= window) & ~full_scored
    # start==0 can be a bounded cache's left edge, not the true episode start.
    # Known internal episode starts must agree, and terminal completion always
    # needs all actual samples; fail instead of silently averaging a suffix.
    inconsistent = (
        (actual_length < 1)
        | (actual_length < visible_length)
        | ((start > 0) & (actual_length != visible_length))
        | (valid & (actual_length != visible_length))
    )
    if bool(inconsistent.any()):
        raise ValueError("truncated_window episode_step disagrees with visible episode samples")
    gather_start = start[..., None].expand(-1, -1, velocity.shape[-1])
    cumulative = torch.cat((torch.zeros_like(velocity[:1]), velocity.cumsum(dim=0)), dim=0)
    average = (cumulative[current_start + 1 :] - cumulative.gather(0, gather_start)) / actual_length[..., None]
    start_target = target.gather(0, gather_start)
    delta = (average - start_target).norm(dim=-1)
    return F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none"), valid


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


def height_target_loss(position, height_target, mode="target_exp", *, reduction="mean"):
    """Track the current episode's fixed target, with all distances in meters.

    ``target_exp`` is exactly ``1 - exp(-abs(z - target))``, with a 1 m
    exponential scale. Its loss saturates and its restoring gradient decays
    far from the target; no extra boundary or linear term is added.
    ``target_huber`` uses SmoothL1
    with beta=1 m. Targets are nonlearnable episode data. At zero error, abs
    uses PyTorch's zero subgradient; the exponential loss has a cusp there.
    """
    height = position[..., 2]
    if height_target.shape != height.shape:
        raise ValueError("height_target must match the position's leading sample dimensions")
    error = height - height_target.detach()
    if mode == "target_exp":
        values = -torch.expm1(-error.abs())
    elif mode == "target_huber":
        values = F.smooth_l1_loss(error, torch.zeros_like(error), beta=1.0, reduction="none")
    else:
        raise ValueError("height_target_loss mode must be target_exp or target_huber")
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("height_target_loss reduction must be none, mean, or sum")


def aerialnav_height_loss(position, height_range):
    """Original Navigation C1 height penalty, with a fixed 0.2 m episode-band margin."""
    if height_range.shape != (*position.shape[:-1], 2):
        raise ValueError("height_range must match position's leading dimensions with two episode endpoints")
    low, high = height_range.detach().unbind(-1)
    z = position[..., 2]
    return (low - 0.2 - z).relu().square() + (z - high - 0.2).relu().square()


def diffnav_transition_costs(history, cfg, velocity_history=None):
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
    Truncated mode adds one eligible <=W terminal window when that endpoint
    has no full-window score. This includes the fresh episode's exactly-W
    terminal omitted by the source offset; the original full helper is kept.
    Full and terminal samples share one denominator and keep their endpoints.

    Every variant requires ``obstacle_signed_distance[T, N, 10]`` at the ten
    original pre-step look-ahead queries, before subtracting the drone margin.
    The geometry producer combines signed static/ground distance and the
    dynamic-body distance. No unsigned-vector fallback is used.

    Target height modes pair each post-step position, including the terminal
    position, with its pre-step cloned ``height_target[T, N]``. A reset target
    belongs only to the following transition. The historical ``height_boundary``
    metric and coefficient name now denote the selected height term. The
    explicit ``boundary_huber`` mode retains the previous boundary penalty.
    """
    if cfg.variant not in ("step", "continuous", "dcol", "truncated_window"):
        raise ValueError(f"Unsupported DiffNav loss variant: {cfg.variant}")
    if "obstacle_signed_distance" not in history:
        raise ValueError("DiffNav losses require obstacle_signed_distance from the pre-step look-ahead queries")
    if cfg.height_loss_mode not in ("target_exp", "target_huber", "boundary_huber", "goal_height_band"):
        raise ValueError("height_loss_mode must be target_exp, target_huber, boundary_huber, or goal_height_band")
    if cfg.height_loss_mode in ("target_exp", "target_huber") and "height_target" not in history:
        raise ValueError("target height losses require the pre-step episode height_target")
    if cfg.height_loss_mode == "goal_height_band" and "height_range" not in history:
        raise ValueError("goal_height_band requires the pre-step episode height_range")
    velocity = torch.stack(history["velocity"])
    target = torch.stack(history["target_velocity"])
    episode = torch.stack(history["episode_id"])
    timesteps, num_envs = velocity.shape[:2]
    detached_history = None
    short_samples = velocity.new_zeros(())
    if cfg.resolved_velocity_loss_mode in ("per_step", "ema"):
        scored_velocity = torch.stack(history["velocity_ema"]) if cfg.resolved_velocity_loss_mode == "ema" else velocity
        velocity_cost = velocity_step_loss(scored_velocity, target, reduction="none")
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
        values, valid = _velocity_window_samples(all_velocity, all_target, all_episode, window, prefix_length)
        valid_samples = valid.sum()
        # The source omits its first velocity sample, so its first full-window
        # score is at index W. Preserve this alignment, then remove old prefix
        # endpoints instead of spreading their costs over the current rollout.
        padding = values.new_zeros(min(window, all_velocity.shape[0]), num_envs)
        endpoint_cost = torch.cat((padding, values * valid), dim=0)[prefix_length:]
        if cfg.resolved_velocity_loss_mode == "truncated_window":
            full_samples = valid_samples
            short_values, short_valid = _truncated_terminal_samples(
                all_velocity,
                all_target,
                all_episode,
                window,
                prefix_length,
                valid,
                history,
            )
            short_samples = short_valid.sum()
            valid_samples = valid_samples + short_samples
            endpoint_cost = endpoint_cost + short_values * short_valid
        velocity_cost = endpoint_cost * (timesteps * num_envs / valid_samples.clamp_min(1))
        detached_history = {
            "velocity": all_velocity[-window:].detach(),
            "target_velocity": all_target[-window:].detach(),
            "episode_id": all_episode[-window:].detach(),
        }
    margin = torch.stack(history["margin"])[:, :, None]
    obstacle_distance = torch.stack(history["obstacle_signed_distance"])
    if obstacle_distance.shape != (timesteps, num_envs, 10):
        raise ValueError("obstacle_signed_distance must match the [T, N, 10] obstacle look-ahead samples")
    distance = obstacle_distance - margin
    # Ten source look-ahead points include both endpoints: dt_sample = policy_dt / 9.
    approach = (-distance.diff(dim=2) * ((distance.shape[2] - 1) / cfg.policy_dt)).clamp_min(1).detach()
    lateral_cost = 0.0
    if cfg.approach_mode == "relative":
        closing, lateral_cost = relative_avoidance(
            torch.stack(history["relative_velocity"]), torch.stack(history["obstacle_normal"]), distance[:, :, 1:]
        )
        approach = closing.clamp_min(1).detach()
    collision_gap = torch.stack(history["dcol_gap"]) if cfg.variant == "dcol" else distance[:, :, 1:]
    command = torch.stack(history["command"])
    command_previous = torch.stack(history["previous_command"])
    position = torch.stack(history["position"])
    if cfg.height_loss_mode == "goal_height_band":
        height_cost = aerialnav_height_loss(position, torch.stack(history["height_range"]))
    elif cfg.height_loss_mode == "boundary_huber":
        height_cost = height_boundary_loss(
            position,
            cfg.height_bounds,
            cfg.boundary_margin,
            cfg.boundary_scale,
            reduction="none",
        )
    else:
        height_cost = height_target_loss(
            position,
            torch.stack(history["height_target"]),
            cfg.height_loss_mode,
            reduction="none",
        )
    terms = {
        "velocity": velocity_cost,
        "velocity_prediction": F.mse_loss(
            torch.stack(history["velocity_prediction"]), torch.stack(history["observed_velocity"]), reduction="none"
        ).mean(dim=-1),
        "obstacle_avoidance": (approach * (1 - distance[:, :, 1:]).relu().square()).mean(dim=-1)
        + cfg.lateral_avoidance * lateral_cost,
        "collision": (F.softplus(-32 * collision_gap) * approach).mean(dim=-1),
        "acceleration": command.square().sum(dim=-1),
        "jerk": ((command - command_previous) / cfg.policy_dt).square().sum(dim=-1),
        "height_boundary": height_cost,
    }
    cost = (
        cfg.coef_v * terms["velocity"]
        + cfg.coef_v_pred * terms["velocity_prediction"]
        + cfg.coef_obj_avoidance * terms["obstacle_avoidance"]
        + cfg.coef_collide * terms["collision"]
        + cfg.coef_d_acc * terms["acceleration"]
        + cfg.coef_d_jerk * terms["jerk"]
        + cfg.coef_height_boundary * terms["height_boundary"]
    )
    if cfg.failure_terminal_cost:
        # One immediate cost at absorbing failure, shared by actor and TD targets.
        # Keep a differentiable height/clearance penalty; this event has no path derivative.
        failure = torch.stack(history["failure_terminal"])
        if failure.dtype != torch.bool or failure.shape != cost.shape:
            raise ValueError("failure_terminal must be a boolean transition mask")
        terms["failure_terminal"] = cfg.failure_terminal_cost * failure.to(cost)
        cost = cost + terms["failure_terminal"]
    losses = {name: value.mean() for name, value in terms.items()}
    losses["total"] = cost.mean()
    losses["velocity_valid_samples"] = valid_samples.detach()
    losses["velocity_sample_fraction"] = valid_samples.detach() / (timesteps * num_envs)
    if cfg.approach_mode == "relative":
        losses["relative_closing_mps"] = closing.detach().mean()
        losses["lateral_avoidance"] = lateral_cost.detach().mean()
        losses["target_speed_scale"] = torch.stack(history["target_speed_scale"]).mean()
    # Includes an exactly-W terminal completion when the source omitted it.
    losses["velocity_short_terminal_samples"] = short_samples.detach()
    if cfg.resolved_velocity_loss_mode == "truncated_window":
        losses["velocity_full_window_samples"] = full_samples.detach()
    return losses, detached_history, cost
