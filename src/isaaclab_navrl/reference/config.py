"""Configuration for the isolated historical DCOL comparator."""

from dataclasses import dataclass

from isaaclab_navrl.zhang.config import ZhangTrainCfg


@dataclass
class DcolReferenceTrainCfg(ZhangTrainCfg):
    collision_metric: str = "dcol"
    geometry_backend: str = "native"

    def validate_collision_metric(self):
        if self.collision_metric != "dcol":
            raise ValueError("The historical reference requires DCOL")
