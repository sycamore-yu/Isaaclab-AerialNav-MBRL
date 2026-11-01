"""Smooth final-approach references; the legacy linear contract stays readable."""

import torch


def goal_velocity(position, target, cruise_speed, *, profile="linear", approach_time_s=2.0, clear_weight=None):
    """Quintic C2 cruise join with linear capture at the goal centre."""
    error = target.detach() - position.detach()
    distance = error.norm(dim=-1, keepdim=True)
    linear = error / distance.clamp_min(1e-6) * torch.minimum(distance, cruise_speed)
    if profile == "linear":
        return linear
    if profile != "smooth":
        raise ValueError("goal_speed_profile must be linear or smooth")
    radius = approach_time_s * cruise_speed
    s = (distance / radius).clamp(0, 1)
    # This form has no direction singularity at r=0.
    capture = error / approach_time_s * (1 + 4 * s.square() - 7 * s**3 + 3 * s**4)
    smooth = torch.where(distance < radius, capture, error / distance.clamp_min(1e-6) * cruise_speed)
    if clear_weight is None:
        return smooth
    weight = clear_weight.detach().reshape(-1, 1)
    return linear + weight * (smooth - linear)


@torch.no_grad()
def final_approach_weight(env, position, target, cruise_speed, margin, approach_time_s):
    """Soft ESDF line-of-sight gate, not a replacement for collision checking.

    Query the full existing world. Half a sample interval is subtracted from
    endpoint clearance; a 0.1 m quintic blend avoids a hard free/blocked switch.
    """
    position, target = position.detach(), target.detach()
    distance = (target - position).norm(dim=-1)
    ids = (distance < approach_time_s * cruise_speed.flatten()).nonzero().flatten()
    weight = distance.new_zeros(distance.shape)
    if ids.numel() == 0:
        return weight
    samples = torch.linspace(0, 1, 128, device=position.device, dtype=position.dtype)
    query = position[ids, None] + (target[ids] - position[ids])[:, None] * samples[None, :, None]
    clearance = env.nearest_obstacle_signed_distance(query) - margin[ids, None]
    clearance = clearance.amin(dim=1) - distance[ids] / (2 * (samples.numel() - 1))
    x = (clearance / 0.1).clamp(0, 1)
    weight[ids] = x**3 * (10 - 15 * x + 6 * x.square())
    return weight
