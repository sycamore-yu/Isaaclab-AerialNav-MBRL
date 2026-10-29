"""NavRL terminal predicates needed between Zhang policy steps.

Keep the original arithmetic, nearest-five selection and thresholds. The full
navigation_terms function still supplies rewards and observations at policy
boundaries; these intermediate checks need no feature encoding or reward logs.
"""

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


def first_events(root_state, target, lidar_distance, motion):
    position = root_state[:, :3]
    distance = (target - position).norm(dim=-1, keepdim=True)
    lidar_scan = 4.0 - lidar_distance.clamp_max(4.0)
    dynamic_collision = torch.zeros(len(position), device=position.device, dtype=torch.bool)
    if len(motion.position):
        relative_all = motion.position[None] - position[:, None]
        relative_all[:, motion.cylinders, 2] = 0.0
        distances_xy = relative_all[..., :2].norm(dim=-1)
        selected_dist, indices = distances_xy.topk(5, dim=-1, largest=False)
        outside = selected_dist > 4.0
        relative_selected = relative_all.gather(1, indices[..., None].expand(-1, -1, 3))
        sizes = motion.size[indices]
        widths, heights = sizes[..., 0:1], sizes[..., 2:3]
        dynamic_collision = (
            (relative_selected[..., :2].norm(dim=-1) <= widths[..., 0] / 2 + 0.3)
            & (relative_selected[..., 2].abs() <= heights[..., 0] / 2 + 0.3)
            & ~outside
        ).any(dim=-1)
    return {
        "collision": (lidar_scan.flatten(1).amax(dim=-1) > 3.7) | dynamic_collision,
        "bounds": (position[:, 2] < 0.2) | (position[:, 2] > 4.0),
        "reached": distance[:, 0] < 0.5,
        "invalid": ~torch.isfinite(root_state).all(dim=-1),
    }
