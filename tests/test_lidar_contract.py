"""Navigation yaw compatibility and full-attitude ray alignment, without PhysX."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.utils.math import quat_apply, quat_apply_yaw, quat_from_euler_xyz
from isaaclab.utils.warp import ProxyArray

from aerialnav.tasks.navigation.sensor_mixins import _NavigationRayCasterMixin


@pytest.mark.parametrize("alignment", ["yaw", "base", "world"])
def test_ray_alignment_respects_attitude_and_update_mask(alignment):
    wp.init()
    torch.manual_seed(13)
    count, rays = 7, 144
    euler = torch.deg2rad(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 90.0],
                [30.0, 0.0, 0.0],
                [0.0, 30.0, 0.0],
                [25.0, -20.0, 70.0],
                [90.0, 0.0, 0.0],
                [-20.0, 30.0, -40.0],
            ]
        )
    )
    quaternion = quat_from_euler_xyz(*euler.T)
    position = torch.randn(count, 3)
    transforms = torch.cat((position, quaternion), -1).contiguous()
    starts = torch.randn(count, rays, 3)
    directions = torch.nn.functional.normalize(torch.randn(count, rays, 3), dim=-1)
    mask = wp.array([True] * (count - 1) + [False], dtype=wp.bool, device="cpu")
    zero = wp.zeros(count, dtype=wp.vec3f, device="cpu")
    identity = wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (count, 1)), dtype=wp.quatf, device="cpu")
    out_position = wp.zeros(count, dtype=wp.vec3f, device="cpu")
    out_quaternion = wp.zeros(count, dtype=wp.quatf, device="cpu")
    world_starts = wp.zeros((count, rays), dtype=wp.vec3f, device="cpu")
    world_directions = wp.zeros((count, rays), dtype=wp.vec3f, device="cpu")
    starts_wp = wp.from_torch(starts, dtype=wp.vec3f)
    directions_wp = wp.from_torch(directions, dtype=wp.vec3f)

    class Sensor(_NavigationRayCasterMixin, BaseRayCaster):
        def _get_view_transforms_wp(self):
            return wp.from_torch(transforms, dtype=wp.transformf)

        def __del__(self):
            pass  # No simulator callbacks are registered by this CPU fixture.

    sensor = object.__new__(Sensor)
    sensor.cfg = SimpleNamespace(ray_alignment=alignment)
    sensor._num_envs, sensor.num_rays, sensor._device = count, rays, "cpu"
    sensor._alignment_mode = {"world": 0, "yaw": 1, "base": 2}[alignment]
    sensor._offset_pos_wp, sensor._offset_quat_wp = zero, identity
    sensor.drift, sensor.ray_cast_drift = ProxyArray(zero), ProxyArray(zero)
    sensor.ray_starts, sensor.ray_directions = ProxyArray(starts_wp), ProxyArray(directions_wp)
    sensor._data = SimpleNamespace(_pos_w=out_position, _quat_w=out_quaternion)
    sensor._ray_starts_w, sensor._ray_directions_w = world_starts, world_directions
    sensor._update_ray_infos(mask)
    rotations = quaternion[:, None].expand(-1, rays, -1).contiguous()
    if alignment == "world":
        expected_start, expected_direction = starts + position[:, None], directions
    else:
        rotate = quat_apply_yaw if alignment == "yaw" else quat_apply
        expected_start = rotate(rotations, starts) + position[:, None]
        expected_direction = rotate(rotations, directions)
    torch.testing.assert_close(wp.to_torch(world_starts)[:-1], expected_start[:-1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(wp.to_torch(world_directions)[:-1], expected_direction[:-1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(wp.to_torch(world_starts)[-1], torch.zeros(rays, 3), rtol=0, atol=0)
    torch.testing.assert_close(wp.to_torch(world_directions)[-1], torch.zeros(rays, 3), rtol=0, atol=0)
