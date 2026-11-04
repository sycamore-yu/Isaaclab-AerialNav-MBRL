"""Native Isaac Lab domain presets for shared navigation observations and policy."""

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab_tasks.utils.hydra import PresetCfg, preset

from aerialnav.diffnav.config import DiffNavTrainCfg

from .config import PointNetTrainCfg


@configclass
class PerceptionCfg:
    mode: str = "depth"
    policy_outputs: int = 6
    point_lidar: RayCasterCfg | None = None


@configclass
class NavigationPerceptionPresets(PresetCfg):
    default = PerceptionCfg()
    pointnet = PerceptionCfg(
        mode="pointnet",
        policy_outputs=3,
        point_lidar=RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Hummingbird/base_link",
            spawn=None,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ray_alignment="base",
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=2.0, vertical_ray_angles=tuple(float(a) for a in range(-10, 50, 2))
            ),
            mesh_prim_paths=["/World/ground"],
            max_distance=20.0,
            update_period=0.05,
        ),
    )


def training_presets():
    return preset(default=DiffNavTrainCfg(), pointnet=PointNetTrainCfg())
