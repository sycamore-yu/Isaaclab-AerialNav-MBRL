"""Reset independent aircraft without resetting the shared world."""

from isaaclab.scene import InteractiveScene


class SharedNavigationScene(InteractiveScene):
    def reset(self, env_ids=None):
        self.articulations["drone"].reset(env_ids)
        for sensor in self.sensors.values():
            sensor.reset(env_ids)
