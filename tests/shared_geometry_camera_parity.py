"""Verify global non-physics obstacles against the official native Newton camera."""

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_navrl.physics import parse_physics_args, select_physics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parse_physics_args(parser)
    if args.physics_preset != "newton_mjwarp":
        parser.error("This regression checks Newton geometry")

    import gymnasium as gym
    import torch
    import warp as wp
    from isaaclab.utils.math import quat_apply
    from isaaclab.utils.string import ResolvableString
    from pxr import UsdPhysics

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation import sensor_mixins
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import ZhangNavRLEnvCfg
    from isaaclab_navrl.tasks.navigation.depth_processing import depth_features
    from isaaclab_navrl.tasks.navigation.newton_sensors import _set_shared_obstacle_mesh_poses

    cfg = ZhangNavRLEnvCfg()
    cfg.training_curriculum = False
    cfg.num_dynamic_obstacles = 8
    cfg.scene.num_envs = 4
    cfg.evaluation = True
    cfg.sim.device = args.device
    cfg.depth_camera_batch_size = 3  # Includes one-image tail batch.
    select_physics(cfg, args.physics_preset)
    cfg.configure_scene()
    cfg.scene.reference_depth = cfg.scene.depth_camera.copy()
    cfg.scene.reference_depth.class_type = ResolvableString("isaaclab.sensors:MultiMeshRayCasterCamera")
    result = {"physics": args.physics_preset, "num_envs": 4, "dynamic_geometry": 8, "stages": {}}
    try:
        with launch_simulation(cfg, args):
            env = gym.make("Isaac-NavRL-Zhang-Hummingbird-Direct-v0", cfg=cfg).unwrapped
            try:
                env.reset()
                camera, reference = env.scene["depth_camera"], env.scene["reference_depth"]
                assert env.dynamic_collection is None
                for index in range(cfg.num_dynamic_obstacles):
                    path = f"/World/Dynamic/obstacle_{index:03d}"
                    prim = env.sim.stage.GetPrimAtPath(path)
                    assert prim.IsValid() and not prim.HasAPI(UsdPhysics.RigidBodyAPI), path
                    for child in prim.GetAllChildren():
                        assert not child.HasAPI(UsdPhysics.RigidBodyAPI), str(child.GetPath())
                result["rigid_obstacle_bodies"] = 0
                result["camera_meshes"] = camera._mesh_ids_wp.shape[1]
                assert result["camera_meshes"] == 10

                # The official camera remains unmodified. Supply its mesh-table
                # poses from the same global motion state that our camera reads.
                def update_reference_mesh_poses():
                    wp.launch(
                        _set_shared_obstacle_mesh_poses,
                        dim=(env.num_envs, cfg.num_dynamic_obstacles),
                        inputs=[
                            camera._shared_motion_positions,
                            camera._shared_active_indices,
                            2,
                            reference._mesh_positions_w,
                            reference._mesh_orientations_w,
                        ],
                        device=env.device,
                    )

                reference._update_mesh_transforms = update_reference_mesh_poses
                camera.cfg.pixel_noise_std = 0.0

                def measure(stage):
                    raw_batches = []

                    def capture(depth, noise=None):
                        raw_batches.append(depth.clone())
                        return depth_features(depth, noise)

                    sensor_mixins.depth_features = capture
                    try:
                        camera.update(0.1)
                        _ = camera.data.output["policy_depth"].torch
                    finally:
                        sensor_mixins.depth_features = depth_features
                    reference.update(0.1)
                    _ = reference.data.output["distance_to_image_plane"].torch
                    raw = torch.cat(raw_batches)
                    native = reference.data.output["distance_to_image_plane"].torch.permute(0, 3, 1, 2)
                    if raw.shape != native.shape:
                        raise AssertionError(f"Shape mismatch: {raw.shape}, {native.shape}")
                    absolute = (raw - native).abs()
                    torch.testing.assert_close(raw, native, atol=2e-5, rtol=1e-5)
                    result["stages"][stage] = {
                        "raw_max_abs_error_m": float(absolute.max()),
                        "raw_mean_abs_error_m": float(absolute.mean()),
                        "pixel_count": raw.numel(),
                        "dynamic_mesh_pose_max_error_m": float(
                            (wp.to_torch(camera._mesh_positions_w) - wp.to_torch(reference._mesh_positions_w))
                            .abs()
                            .max()
                        ),
                    }
                    return raw.clone()

                initial = measure("initial")
                env.motion.position[0] = (
                    env.root_state()[0, :3]
                    + 2.0 * quat_apply(env.root_state()[0:1, 3:7], env.root_state().new_tensor([[1.0, 0.0, 0.0]]))[0]
                )
                moved = measure("moved_dynamic_obstacle")
                assert float((moved - initial).abs().max()) > 0.1
                result["changed_depth_after_motion"] = True
                result["status"] = "passed"
            finally:
                env.close()
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
