"""Navigation's yaw rotation for LiDAR directions (the official yaw mode omits it)."""

import warp as wp
from isaaclab.sensors.ray_caster.kernels import quat_yaw_only


@wp.kernel(enable_backward=False)
def rotate_directions_with_yaw(
    orientations: wp.array(dtype=wp.quatf),
    env_mask: wp.array(dtype=wp.bool),
    local_directions: wp.array2d(dtype=wp.vec3f),
    world_directions: wp.array2d(dtype=wp.vec3f),
):
    env_id, ray_id = wp.tid()
    if env_mask[env_id]:
        yaw = quat_yaw_only(orientations[env_id])
        world_directions[env_id, ray_id] = wp.quat_rotate(yaw, local_directions[env_id, ray_id])
