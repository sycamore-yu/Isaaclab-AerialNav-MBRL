"""Intermediate Zhang checks must exactly match the full NavRL event rules."""

import pytest
import torch

from isaaclab_navrl.tasks.navigation.task_logic import DynamicObstacleMotion, navigation_terms
from isaaclab_navrl.zhang.termination import first_events


@pytest.mark.parametrize("dynamic_count", [0, 100])
@pytest.mark.parametrize("seed", [0, 37])
def test_intermediate_events_equal_full_task_rules(dynamic_count, seed):
    generator = torch.Generator().manual_seed(seed)
    count = 256
    state = torch.randn(count, 13, generator=generator)
    state[:, :2] *= 18
    state[:, 2] = torch.rand(count, generator=generator) * 5
    state[:, 3:7] = torch.nn.functional.normalize(state[:, 3:7], dim=-1)
    target = torch.randn(count, 3, generator=generator) * 12
    target[::3] = state[::3, :3] + torch.randn(len(state[::3]), 3, generator=generator) * 0.2
    direction = torch.randn(count, 3, generator=generator)
    previous_velocity = torch.randn(count, 3, generator=generator)
    height_range = torch.tensor([[0.5, 2.5]]).repeat(count, 1)
    motion = DynamicObstacleMotion(dynamic_count, seed, "cpu", 0.01)
    motion.advance()
    lidar = torch.full((count, 1, 36, 4), 4.0)
    lidar[:128] = torch.rand(128, 1, 36, 4, generator=generator) * 4.5
    # Exercise threshold equality and both sides without changing arithmetic.
    lidar[:4] = 4.0
    lidar[:4, 0, 0, 0] = torch.tensor([0.3, 0.3 - 1e-6, 0.3 + 1e-6, 4.0])
    state[:4, 2] = torch.tensor([0.2, 0.2 - 1e-6, 4.0, 4.0 + 1e-6])
    target[4:7] = state[4:7, :3]
    target[4:7, 0] += torch.tensor([0.5, 0.5 - 1e-6, 0.5 + 1e-6])
    if dynamic_count:
        ids = torch.tensor([0, 12, 13, 25, 37, 49, 50, 62, 75, 99])
        state[128:138, :3] = motion.position[ids]
        state[128:138, 0] += motion.size[ids, 0] / 2 + 0.3
        state[138:148, :3] = state[128:138, :3]
        state[138:148, 0] += 1e-5
        # Cylinders retain their 2D rule even when altitude differs.
        state[134:138, 2] += 2.0
    state[-3, 8] = float("nan")
    state[-2, 5] = float("inf")
    state[-1, 0] = float("nan")
    reference = navigation_terms(
        state[:, :3], state[:, 7:10], target, direction, lidar,
        motion, previous_velocity, height_range,
    )
    reference["invalid"] = ~torch.isfinite(state).all(dim=-1)
    actual = first_events(state, target, lidar, motion)
    assert set(actual) == {"collision", "reached", "bounds", "invalid"}
    for name in actual:
        assert torch.equal(actual[name], reference[name]), name
