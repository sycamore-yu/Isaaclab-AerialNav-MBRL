"""PhysX navigation LiDAR with yaw-only ray alignment."""

from isaaclab_physx.sensors.ray_caster.ray_caster import RayCaster

from .sensor_mixins import _NavRLRayCasterMixin


class NavRLRayCaster(_NavRLRayCasterMixin, RayCaster):
    """PhysX LiDAR with the navigation task's yaw-only ray alignment."""
