"""PointNet's body-mounted scan and world-to-body XYZ contracts."""

import pytest
import torch
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
from isaaclab_tasks.utils.hydra import resolve_presets

from aerialnav.physics import select_physics
from aerialnav.tasks.navigation.config.hummingbird.env_cfg import DiffNavEnvCfg
from aerialnav.tasks.navigation.pointcloud import pack_body_points


@pytest.mark.parametrize("physics", ["newton_mjwarp", "isaacsim_physx"])
def test_pointnet_scan_is_body_mounted_without_changing_navigation_lidar(physics):
    cfg = DiffNavEnvCfg()
    resolve_presets(cfg, selected=("pointnet", physics))
    select_physics(cfg, physics)
    for _ in range(2):
        cfg.configure_scene()
        assert cfg.scene.point_lidar.ray_alignment == "base"
        assert cfg.scene.lidar.ray_alignment == "yaw"
        starts, directions = cfg.scene.point_lidar.pattern_cfg.func(cfg.scene.point_lidar.pattern_cfg, "cpu")
        assert starts.shape == directions.shape == (5400, 3)
        assert cfg.scene.point_lidar.max_distance == 20.0


def test_pointnet_points_recover_body_xyz_under_roll_pitch_yaw_and_missing_returns():
    euler = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 1.2], [0.4, -0.5, 1.2]])
    quaternion = quat_from_euler_xyz(*euler.T)
    position = torch.tensor([1.0, -2.0, 3.0]).expand(len(euler), -1)
    body = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.5, 1.0]]).expand(len(euler), -1, -1).contiguous()
    rotations = quaternion[:, None].expand(-1, body.shape[1], -1).contiguous()
    hits = quat_apply(rotations, body) + position[:, None]
    hits = torch.cat((hits, torch.full((len(euler), 1, 3), float("inf"))), dim=1)

    packed = pack_body_points(hits, position, quaternion)
    assert packed.shape == (len(euler), 4, 3)
    torch.testing.assert_close(packed[:, :3, :2].transpose(1, 2), body, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(packed[:, 3, :2], torch.ones(len(euler), 2))
    torch.testing.assert_close(packed[:, :, -1], torch.zeros(len(euler), 4))
