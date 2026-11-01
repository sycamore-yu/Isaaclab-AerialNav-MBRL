"""Original LiDAR layout and the full yaw-only ray transform, without PhysX."""

import numpy as np
import torch
import warp as wp
from isaaclab.sensors.ray_caster.kernels import update_ray_caster_kernel
from isaaclab.utils.math import quat_apply_yaw, quat_from_euler_xyz

from aerialnav.tasks.navigation.lidar_kernels import rotate_directions_with_yaw


def test_yaw_rotates_both_origins_and_directions_without_roll_or_pitch():
    wp.init()
    torch.manual_seed(13)
    count, rays = 3, 144
    quaternion = quat_from_euler_xyz(
        torch.tensor([0.0, 0.4, -0.5]),
        torch.tensor([0.0, -0.3, 0.6]),
        torch.tensor([0.0, 1.5707963, -0.7]),
    )
    position = torch.randn(count, 3)
    transforms = torch.cat((position, quaternion), -1).contiguous()
    starts = torch.randn(count, rays, 3)
    directions = torch.nn.functional.normalize(torch.randn(count, rays, 3), dim=-1)
    mask = wp.array([True, True, False], dtype=wp.bool, device="cpu")
    zero = wp.zeros(count, dtype=wp.vec3f, device="cpu")
    identity = wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (count, 1)), dtype=wp.quatf, device="cpu")
    out_position = wp.zeros(count, dtype=wp.vec3f, device="cpu")
    out_quaternion = wp.zeros(count, dtype=wp.quatf, device="cpu")
    world_starts = wp.zeros((count, rays), dtype=wp.vec3f, device="cpu")
    world_directions = wp.zeros((count, rays), dtype=wp.vec3f, device="cpu")
    starts_wp = wp.from_torch(starts, dtype=wp.vec3f)
    directions_wp = wp.from_torch(directions, dtype=wp.vec3f)
    wp.launch(
        update_ray_caster_kernel,
        dim=(count, rays),
        device="cpu",
        inputs=[
            wp.from_torch(transforms, dtype=wp.transformf),
            mask,
            zero,
            identity,
            zero,
            zero,
            starts_wp,
            directions_wp,
            1,
        ],
        outputs=[out_position, out_quaternion, world_starts, world_directions],
    )
    wp.launch(
        rotate_directions_with_yaw,
        dim=(count, rays),
        device="cpu",
        inputs=[
            out_quaternion,
            mask,
            directions_wp,
        ],
        outputs=[world_directions],
    )
    rotations = quaternion[:, None].expand(-1, rays, -1).contiguous()
    expected_start = quat_apply_yaw(rotations, starts) + position[:, None]
    expected_direction = quat_apply_yaw(rotations, directions)
    torch.testing.assert_close(wp.to_torch(world_starts)[:2], expected_start[:2], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(wp.to_torch(world_directions)[:2], expected_direction[:2], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(wp.to_torch(world_starts)[2], torch.zeros(rays, 3), rtol=0, atol=0)
    torch.testing.assert_close(wp.to_torch(world_directions)[2], torch.zeros(rays, 3), rtol=0, atol=0)
