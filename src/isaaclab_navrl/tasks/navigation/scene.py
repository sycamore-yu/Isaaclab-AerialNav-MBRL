"""Reset independent aircraft without resetting the shared world."""

from isaaclab_experimental.envs.interactive_scene_warp import InteractiveSceneWarp


class SharedNavigationScene(InteractiveSceneWarp):
    def reset(self, env_ids=None):
        self.articulations["drone"].reset(env_ids)
        for sensor in self.sensors.values():
            sensor.reset(env_ids)
