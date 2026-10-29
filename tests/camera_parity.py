"""Strict full-image parity against Isaac Lab's unchanged native depth camera."""

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
    from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
    from isaaclab.utils.string import ResolvableString

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import ZhangNavRLEnvCfg
    from isaaclab_navrl.tasks.navigation.depth_processing import depth_features

    cfg = ZhangNavRLEnvCfg()
    cfg.training_curriculum = False
    cfg.num_dynamic_obstacles = 100
    cfg.scene.num_envs = 17  # Full 16-camera batch plus a one-camera tail.
    cfg.evaluation = True
    cfg.sim.device = args.device
    cfg.configure_scene()
    cfg.scene.reference_depth = cfg.scene.depth_camera.copy()
    cfg.scene.reference_depth.class_type = ResolvableString("isaaclab.sensors:MultiMeshRayCasterCamera")
    result = {
        "status": "running", "num_envs": 17,
        "full_resolution": [480, 640], "pooled_resolution": [12, 16],
        "atol": 2e-5, "rtol": 1e-5, "stages": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def errors(actual, expected):
        absolute = (actual - expected).abs()
        bad = ~torch.isclose(actual, expected, atol=result["atol"], rtol=result["rtol"])
        coordinates = bad.nonzero()[:10]
        return {
            "elements": actual.numel(), "max_abs_error": float(absolute.max()),
            "mean_abs_error": float(absolute.mean()), "mismatches": int(bad.sum()),
            "examples": [
                {"index": index.tolist(), "actual": float(actual[tuple(index)]),
                 "native": float(expected[tuple(index)])}
                for index in coordinates
            ],
        }

    with launch_simulation(cfg, args):
        from isaaclab_navrl.tasks.navigation import depth_camera as depth_module

        env = gym.make("Isaac-NavRL-Zhang-Hummingbird-Direct-v0", cfg=cfg).unwrapped
        try:
            env.reset()
            camera, reference = env.scene["depth_camera"], env.scene["reference_depth"]
            camera.cfg.pixel_noise_std = 0.0
            assert camera.image_shape == (480, 640)
            assert camera._batch_size == 16
            result["world_mesh"] = camera.world_mesh_info
            result["intrinsics"] = camera.data.intrinsic_matrices.torch.cpu().tolist()
            initial_intrinsics = camera.data.intrinsic_matrices.torch.clone()

            def measure(name, *, selected=None, refresh=True):
                for begin, end in ((0, 16), (16, 17)):
                    cached = camera._local_rays(begin, end)
                    original = camera._compute_local_rays(begin, end)
                    for actual, expected in zip(cached, original, strict=True):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if refresh:
                    camera.update(0.1)
                    reference.update(0.1)
                raw_batches, noise_batches = [], []

                def capture(depth, noise=None):
                    # Observe each batch without changing the production sensor's
                    # storage or image transform. Clone because its buffer is reused.
                    raw_batches.append(depth.detach().clone())
                    noise_batches.append(None if noise is None else noise.detach().clone())
                    return depth_features(depth, noise)

                depth_module.depth_features = capture
                try:
                    pooled = camera.data.output["policy_depth"].torch.permute(0, 3, 1, 2).clone()
                finally:
                    depth_module.depth_features = depth_features
                assert [len(batch) for batch in raw_batches] == [16, 1], "Tail batch was not exercised."
                raw = torch.cat(raw_batches)
                native = reference.data.output["distance_to_image_plane"].torch.permute(0, 3, 1, 2).clone()
                noise = None if noise_batches[0] is None else torch.cat(noise_batches)
                expected = depth_features(native, noise)
                chosen = slice(None) if selected is None else selected
                details = {
                    "batch_sizes": [len(batch) for batch in raw_batches],
                    "selected_env_ids": list(range(17)) if selected is None else selected.cpu().tolist(),
                    "depth": errors(raw[chosen], native[chosen]),
                    "pooled": errors(pooled, expected),
                    "pixel_noise_std": camera.cfg.pixel_noise_std,
                }
                result["stages"][name] = details
                persist()
                if details["depth"]["mismatches"] or details["pooled"]["mismatches"]:
                    failure_file = args.output.with_suffix(".failure.pt")
                    torch.save({
                        "stage": name, "raw": raw.cpu(), "native": native.cpu(),
                        "pooled": pooled.cpu(), "expected_pooled": expected.cpu(),
                        "drone_state": env.root_state().cpu(),
                        "camera_position": camera.data.pos_w.torch.cpu(),
                        "camera_quaternion": camera.data.quat_w_world.torch.cpu(),
                        "intrinsics": camera.data.intrinsic_matrices.torch.cpu(),
                        "dynamic_poses": env.dynamic_collection.data.body_pose_w.torch.cpu(),
                        "selected_env_ids": details["selected_env_ids"],
                    }, failure_file)
                    result["failure_snapshot"] = str(failure_file.resolve())
                    persist()
                torch.testing.assert_close(raw[chosen], native[chosen], atol=2e-5, rtol=1e-5)
                torch.testing.assert_close(pooled, expected, atol=2e-5, rtol=1e-5)
                return raw, pooled

            measure("initial")
            state = env.root_state()
            phase = torch.linspace(0.2, 2.7, 17, device=env.device)
            roll = 0.23 * phase.sin()
            pitch = 0.19 * phase.cos()
            yaw = -torch.pi / 2 + 0.3 * phase.sin()
            quaternion = quat_from_euler_xyz(roll, pitch, yaw)
            env.drone.write_root_pose_to_sim_index(
                root_pose=torch.cat((state[:, :3], quaternion), dim=-1),
                env_ids=torch.arange(17, device=env.device, dtype=torch.int32),
            )
            tilted_raw, _ = measure("roll_pitch_yaw")
            env.lidar.update(0.1)
            lidar_before = env.lidar.data.ray_hits_w.torch.clone()

            poses = env.dynamic_collection.data.body_pose_w.torch.clone()
            obstacle_phase = torch.arange(poses.shape[1], device=env.device, dtype=torch.float32) + 0.3
            poses[0, :, 0] += 0.7 + 0.2 * obstacle_phase.sin()
            poses[0, :, 1] += 0.6 + 0.2 * obstacle_phase.cos()
            poses[0, :, 2] += 0.15
            zero = torch.zeros_like(obstacle_phase)
            poses[0, :, 3:7] = quat_from_euler_xyz(zero, zero, obstacle_phase * 0.07)
            state = env.root_state()
            forward = quat_apply(state[:, 3:7], state.new_tensor([[1.0, 0.0, 0.0]]).expand(17, -1))
            poses[0, 0, :3] = state[0, :3] + 2.0 * forward[0]
            env.dynamic_collection.write_body_pose_to_sim_index(body_poses=poses)
            env.dynamic_collection.write_data_to_sim()
            moved_raw, moved_pooled = measure("all_dynamic_bodies_moved")
            assert float((moved_raw - tilted_raw).abs().max()) > 0.1, "Moving geometry did not change camera depth."
            env.lidar.update(0.1)
            torch.testing.assert_close(env.lidar.data.ray_hits_w.torch, lidar_before)
            result["dynamic_obstacles_visible_in_depth"] = True
            result["dynamic_obstacles_excluded_from_navrl_lidar"] = True

            reset_ids = torch.tensor([0, 8, 16], device=env.device, dtype=torch.int32)
            keep = torch.ones(17, dtype=torch.bool, device=env.device)
            keep[reset_ids] = False
            frame_before = camera._frame.clone()
            env._reset_idx(reset_ids)
            _, reset_pooled = measure("partial_reset", selected=reset_ids, refresh=False)
            torch.testing.assert_close(reset_pooled[keep], moved_pooled[keep], atol=0, rtol=0)
            torch.testing.assert_close(camera._frame[keep], frame_before[keep], atol=0, rtol=0)
            torch.testing.assert_close(camera._frame[reset_ids], torch.ones_like(camera._frame[reset_ids]))
            result["partial_reset_preserves_other_images_and_frames"] = True

            camera.cfg.pixel_noise_std = 0.02
            camera._noise_rng.manual_seed(20261028)
            measure("pixel_noise_before_pool")
            matrices = camera.data.intrinsic_matrices.torch[reset_ids].clone()
            scale = torch.tensor([0.85, 1.1, 1.25], device=env.device)
            matrices[:, 0, 0] *= scale
            matrices[:, 1, 1] *= scale
            matrices[:, 0, 2] += torch.tensor([-4.0, 2.0, 5.0], device=env.device)
            camera.set_intrinsic_matrices(matrices, env_ids=reset_ids)
            reference.set_intrinsic_matrices(matrices, env_ids=reset_ids)
            measure("partial_intrinsics_update")
            camera.set_intrinsic_matrices(initial_intrinsics)
            reference.set_intrinsic_matrices(initial_intrinsics)
            measure("restored_shared_intrinsics")
            result["cached_local_rays_exact"] = True
            result["status"] = "completed"
            persist()
            print("CAMERA_PARITY " + json.dumps(result), flush=True)
        except BaseException as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
            persist()
            raise
        finally:
            depth_module.depth_features = depth_features
            env.close()


if __name__ == "__main__":
    main()
