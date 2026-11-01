"""Navigation observation, reward and shared-obstacle rules, independent of Kit.

Adapted from Zhefan-Xu/Navigation, commit 3725bcc (MIT). World coordinates are
intentional: cloned drones share one map and never offset it by env_origins.
"""

from __future__ import annotations

import numpy as np
import torch

from aerialnav.training import END_CLEARANCE, MAP_SIZE


def sample_navigation_route(speed, generator, height_range=(0.5, 2.5)):
    """Sample feasible XY segments of exactly 18v metres, within +/-8 degrees of +/-X."""
    speed = speed.reshape(-1)
    if not torch.isfinite(speed).all() or (speed <= 0).any() or (18 * speed > MAP_SIZE[0] - 2 * END_CLEARANCE).any():
        raise ValueError("18v routes require finite positive target speeds at most 6 m/s")
    count = len(speed)
    random = torch.rand(count, 6, generator=generator, device=speed.device, dtype=speed.dtype)
    angle = (random[:, 0] * 2 - 1) * (8 * torch.pi / 180)
    sign = torch.where(random[:, 1] < 0.5, -1.0, 1.0)
    delta = 18 * speed[:, None] * sign[:, None] * torch.stack((angle.cos(), angle.sin()), dim=-1)
    half = speed.new_tensor((MAP_SIZE[0] / 2 - END_CLEARANCE, MAP_SIZE[1] / 2))
    low = torch.maximum(-half, -half - delta)
    high = torch.minimum(half, half - delta)
    position = speed.new_empty(count, 3)
    position[:, :2] = low + (high - low).clamp_min(0) * random[:, 2:4]
    position[:, 2] = 0.5 + 2.0 * random[:, 4]
    target = position.clone()
    target[:, :2] += delta
    target[:, 2] = height_range[0] + (height_range[1] - height_range[0]) * random[:, 5]
    return position, target


def goal_frame(vector: torch.Tensor, direction: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """Transform vectors using the episode's fixed horizontal goal frame."""
    horizontal = direction[..., :2]
    unit = horizontal / horizontal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    c, s = unit.unbind(-1)
    while c.ndim < vector.ndim - 1:
        c, s = c.unsqueeze(-1), s.unsqueeze(-1)
    x, y, z = vector.unbind(-1)
    if inverse:
        return torch.stack((c * x - s * y, s * x + c * y, z), dim=-1)
    return torch.stack((c * x + s * y, -s * x + c * y, z), dim=-1)


def obstacle_categories(total: int) -> list[int]:
    """Preserve half cuboids / half cylinders and four evenly populated widths."""
    if total != 0 and total < 8:
        raise ValueError("Use zero or at least eight dynamic obstacles.")
    counts = []
    for half in (total // 2, total - total // 2):
        counts.extend(half // 4 + int(i < half % 4) for i in range(4))
    return counts


def dynamic_layout(total: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Shared rectangular world; end strips stay clear of obstacle footprints."""
    rng = np.random.RandomState(seed)
    positions, sizes, previous = [], [], []
    half_x, half_y = MAP_SIZE[0] / 2 - END_CLEARANCE, MAP_SIZE[1] / 2
    distance = np.sqrt((2 * half_x) * (2 * half_y) / total) if total else 0.0
    for category, count in enumerate(obstacle_categories(total)):
        for _ in range(count):
            spacing, attempts = distance, 0
            while True:
                width = (category % 4 + 1) / 4
                x, y = rng.uniform((-half_x + width / 2, -half_y + width / 2), (half_x - width / 2, half_y - width / 2))
                z = rng.uniform(0, 4.5) if category < 4 else 2.5
                point = np.array((x, y))
                valid = all(np.linalg.norm(point - other) > spacing for other in previous)
                attempts += 1
                if attempts % 1000 == 0:
                    spacing *= 0.8
                if valid:
                    previous.append(point)
                    break
            width = (category % 4 + 1) / 4
            positions.append((x, y, z))
            sizes.append((width, width, 1.0 if category < 4 else 5.0))
    return np.asarray(positions, dtype=np.float32).reshape(-1, 3), np.asarray(sizes, dtype=np.float32).reshape(-1, 3)


class DynamicObstacleMotion:
    """One global Navigation obstacle process, independent of drone resets."""

    def __init__(self, total: int, seed: int, device: str, dt: float, speed_range=(0.5, 1.5)):
        origins, sizes = dynamic_layout(total, seed)
        self.position = torch.tensor(origins, device=device)
        self.origin = self.position.clone()
        self.size = torch.tensor(sizes, device=device)
        self.goal = self.position.clone()
        self.velocity = torch.zeros_like(self.position)
        self.cylinders = self.size[:, 2] > 1.0
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.dt, self.speed_range, self.step_count = dt, speed_range, 0

    def advance(self):
        if len(self.position) == 0:
            return
        needs_goal = (self.position - self.goal).norm(dim=-1) < 0.5
        ids = needs_goal.nonzero().flatten()
        local = torch.empty((len(ids), 3), device=self.position.device)
        for axis, extent in enumerate((5.0, 5.0, 4.5)):
            local[:, axis] = torch.rand(len(ids), generator=self.rng, device=local.device) * (2 * extent) - extent
        self.goal[ids] = self.origin[ids] + local
        half = self.size.new_tensor((MAP_SIZE[0] / 2 - END_CLEARANCE, MAP_SIZE[1] / 2)) - self.size[:, :2] / 2
        self.goal[:, :2] = torch.maximum(torch.minimum(self.goal[:, :2], half), -half)
        self.goal[:, 2].clamp_(0, 4.5)
        self.goal[self.cylinders, 2] = 2.5
        if self.step_count % max(1, int(2.0 / self.dt)) == 0:
            lo, hi = self.speed_range
            speed = lo + (hi - lo) * torch.rand(len(self.position), 1, generator=self.rng, device=self.position.device)
            delta = self.goal - self.position
            self.velocity = speed * delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.position.add_(self.velocity * self.dt)
        # Velocity is held for two seconds; reflect overshoot to preserve the shared world's bounds.
        for axis in range(2):
            limit = half[:, axis]
            outside = self.position[:, axis].abs() > limit
            self.position[:, axis] = torch.where(
                outside,
                self.position[:, axis].sign() * (2 * limit - self.position[:, axis].abs()),
                self.position[:, axis],
            )
            self.velocity[:, axis] = torch.where(outside, -self.velocity[:, axis], self.velocity[:, axis])
        self.step_count += 1


def navigation_terms(position, velocity, target, direction, lidar_distance, motion, previous_velocity, height_range):
    """Compute original inputs, all reward terms, and terminal events."""
    relative = target - position
    distance = relative.norm(dim=-1, keepdim=True)
    state = torch.cat(
        (
            goal_frame(relative / distance.clamp_min(1e-6), direction),
            relative[:, :2].norm(dim=-1, keepdim=True),
            relative[:, 2:3],
            goal_frame(velocity, direction),
        ),
        dim=-1,
    )
    lidar_distance = lidar_distance.clamp_max(4.0)
    lidar_scan = 4.0 - lidar_distance
    dynamic = position.new_zeros(len(position), 1, 5, 10)
    safety_dynamic = position.new_zeros(len(position))
    if len(motion.position):
        relative_all = motion.position[None] - position[:, None]
        relative_all[:, motion.cylinders, 2] = 0.0
        distances_xy = relative_all[..., :2].norm(dim=-1)
        selected_dist, indices = distances_xy.topk(min(5, len(motion.position)), dim=-1, largest=False)
        outside = selected_dist > 4.0
        relative_selected = relative_all.gather(1, indices[..., None].expand(-1, -1, 3))
        relative_goal = goal_frame(relative_selected, direction).masked_fill(outside[..., None], 0)
        norm = relative_selected.norm(dim=-1, keepdim=True)
        obstacle_velocity = motion.velocity[indices].masked_fill(outside[..., None], 0)
        sizes = motion.size[indices]
        widths, heights = sizes[..., 0:1], sizes[..., 2:3]
        width_class = (widths / 0.25 - 1).masked_fill(outside[..., None], 0)
        height_class = torch.where(heights > 1, 0, heights).masked_fill(outside[..., None], 0)
        dynamic = torch.cat(
            (
                relative_goal / norm.clamp_min(1e-6),
                relative_goal[..., :2].norm(dim=-1, keepdim=True),
                relative_goal[..., 2:3],
                goal_frame(obstacle_velocity, direction),
                width_class,
                height_class,
            ),
            dim=-1,
        ).unsqueeze(1)
        dynamic = torch.nn.functional.pad(dynamic, (0, 0, 0, 5 - dynamic.shape[2]))
        surface_distance = (norm[..., 0] - widths[..., 0] / 2).masked_fill(outside, 4.0)
        safety_dynamic = surface_distance.clamp(1e-6, 4.0).log().mean(dim=-1)
    safety_static = lidar_distance.clamp(1e-6, 4.0).log().flatten(1).mean(dim=-1)
    velocity_reward = (velocity * relative / distance.clamp_min(1e-6)).sum(dim=-1)
    smooth = (velocity - previous_velocity).norm(dim=-1)
    below = (height_range[:, 0] - 0.2 - position[:, 2]).clamp_min(0)
    above = (position[:, 2] - height_range[:, 1] - 0.2).clamp_min(0)
    height = below.square() + above.square()
    reward = velocity_reward + 1 + safety_static + safety_dynamic - 0.1 * smooth - 8 * height
    return {
        "state": state,
        "lidar": lidar_scan,
        "dynamic_obstacle": dynamic,
        "reward": reward,
        "bounds": (position[:, 2] < 0.2)
        | (position[:, 2] > 4.0)
        | (position[:, 0].abs() > MAP_SIZE[0] / 2)
        | (position[:, 1].abs() > MAP_SIZE[1] / 2),
        "reached": distance[:, 0] < 0.5,
        "reward_velocity": velocity_reward,
        "reward_static": safety_static,
        "reward_dynamic": safety_dynamic,
        "penalty_smooth": smooth,
        "penalty_height": height,
    }
