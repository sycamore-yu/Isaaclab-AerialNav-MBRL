"""Configuration for the isolated historical DCOL comparator."""

from dataclasses import dataclass

from aerialnav.diffnav.config import DiffNavTrainCfg


@dataclass
class DcolReferenceTrainCfg(DiffNavTrainCfg):
    collision_metric: str = "dcol"
    geometry_backend: str = "native"

    def validate_collision_metric(self):
        if self.collision_metric != "dcol":
            raise ValueError("The historical reference requires DCOL")
