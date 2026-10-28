"""NavRL LiDAR: yaw rotates both starts and directions, independently of tilt."""

import warp as wp
from isaaclab_physx.sensors.ray_caster.ray_caster import RayCaster

from .lidar_kernels import rotate_directions_with_yaw


class NavRLRayCaster(RayCaster):
    """Preserve the customized Orbit sensor without modifying Isaac Lab."""

    def _update_ray_infos(self, env_mask):
        # The official yaw path already rotates starts and supplies the tracked
        # orientation. NavRL also rotates directions by that same yaw.
        super()._update_ray_infos(env_mask)
        wp.launch(
            rotate_directions_with_yaw,
            dim=(self._num_envs, self.num_rays),
            inputs=[self._data._quat_w, env_mask, self.ray_directions.warp],
            outputs=[self._ray_directions_w],
            device=self._device,
        )
