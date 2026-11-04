"""A recording stores measured hits rather than filled miss-ray endpoints."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

pytest.importorskip("rerun")
from aerialnav.pointnet.recording import PointCloudRecording, terrain_mesh_parts  # noqa: E402


def test_recording_preserves_only_valid_world_and_body_hits(tmp_path):
    wp.init()
    mesh = wp.Mesh(
        points=wp.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=wp.vec3, device="cpu"),
        indices=wp.array([0, 1, 2], dtype=wp.int32, device="cpu"),
    )
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(mesh_prim_paths=["/World/ground"]),
        meshes={("/World/ground", "cpu"): mesh},
        data=SimpleNamespace(ray_hits_w=SimpleNamespace(torch=torch.tensor([[[1.0, 0.0, 0.0], [float("inf")] * 3]]))),
    )
    state = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene={"point_lidar": sensor},
        target_pos=torch.tensor([[4.0, 0.0, 1.0]]),
        root_state=lambda: state,
        nearest_obstacle_signed_distance=lambda p: torch.tensor([1.3]),
        diffnav_parameters={"margin": torch.tensor([0.3])},
        get_diffnav_observation=lambda: {
            "policy_points": torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]])
        },
    )
    record = PointCloudRecording(tmp_path, env)
    record.frame(0.0)
    record.close({"outcome": "fixture"})
    trace = np.load(tmp_path / "trace.npz")
    np.testing.assert_array_equal(trace["hits_world"], [[1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(trace["hits_body"], [[1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(trace["hit_offsets"], [0, 1])
    assert (tmp_path / "navigation.rrd").stat().st_size > 1000


def test_terrain_palette_matches_main_without_changing_any_triangle():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 2]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3]], dtype=np.int32)
    parts = list(terrain_mesh_parts(vertices, faces))
    assert [p[0] for p in parts] == ["ground", "static_obstacles"]
    assert [p[3] for p in parts] == [[58, 73, 86, 255], [215, 225, 235, 255]]
    restored = np.concatenate([points[triangles] for _, points, triangles, _ in parts])
    np.testing.assert_array_equal(restored, vertices[faces])
