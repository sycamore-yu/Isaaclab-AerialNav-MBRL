"""Compare batched full-resolution depth with Isaac Lab's native camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/smoke/camera_parity.json"))
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()
    import gymnasium as gym
    import torch
    from isaaclab.utils.math import quat_apply
    from isaaclab.utils.string import ResolvableString

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import ZhangNavRLEnvCfg
    from isaaclab_navrl.tasks.navigation.depth_processing import depth_features

    cfg = ZhangNavRLEnvCfg()
    cfg.scene.num_envs = 1
    cfg.evaluation = True
    cfg.sim.device = args.device
    cfg.configure_scene()
    cfg.scene.reference_depth = cfg.scene.depth_camera.copy()
    cfg.scene.reference_depth.class_type = ResolvableString("isaaclab.sensors:MultiMeshRayCasterCamera")
    with launch_simulation(cfg, args):
        env = gym.make("Isaac-NavRL-Zhang-Hummingbird-Direct-v0", cfg=cfg).unwrapped
        try:
            env.reset()
            camera, reference = env.scene["depth_camera"], env.scene["reference_depth"]
            pooled = camera.data.output["policy_depth"].torch.permute(0, 3, 1, 2).clone()
            raw = reference.data.output["distance_to_image_plane"].torch.permute(0, 3, 1, 2).clone()
            torch.testing.assert_close(camera.last_depth_batch[2], raw, atol=2e-5, rtol=1e-5)
            torch.testing.assert_close(pooled, depth_features(raw), atol=2e-5, rtol=1e-5)
            max_error = float((camera.last_depth_batch[2] - raw).abs().max())
            lidar_before = env.lidar.data.ray_hits_w.torch.clone()
            pose = env.dynamic_collection.data.body_pose_w.torch.clone()
            state = env.root_state()
            forward = quat_apply(state[:, 3:7], state.new_tensor([[1.0, 0.0, 0.0]]))
            pose[0, 0, :3] = state[0, :3] + forward[0]
            env.dynamic_collection.write_body_pose_to_sim_index(body_poses=pose)
            env.dynamic_collection.write_data_to_sim()
            camera.update(0.1)
            reference.update(0.1)
            env.lidar.update(0.1)
            moved = camera.data.output["policy_depth"].torch.permute(0, 3, 1, 2).clone()
            moved_raw = reference.data.output["distance_to_image_plane"].torch.permute(0, 3, 1, 2)
            torch.testing.assert_close(camera.last_depth_batch[2], moved_raw, atol=2e-5, rtol=1e-5)
            torch.testing.assert_close(moved, depth_features(moved_raw), atol=2e-5, rtol=1e-5)
            assert float((moved - pooled).abs().max()) > 0.1, "Dynamic obstacle did not appear in camera."
            torch.testing.assert_close(env.lidar.data.ray_hits_w.torch, lidar_before)
            result = {
                "full_resolution": [480, 640], "pooled_resolution": [12, 16],
                "max_depth_error_m": max_error,
                "dynamic_obstacles_visible_in_depth": True,
                "dynamic_obstacles_excluded_from_navrl_lidar": True,
                "intrinsics": camera.data.intrinsic_matrices.torch.cpu().tolist(),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("CAMERA_PARITY " + json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
