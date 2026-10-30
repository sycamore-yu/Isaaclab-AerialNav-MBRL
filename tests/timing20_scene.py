"""Measure physics, action and sensor cadence in the real 20 Hz scene."""

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing", choices=("unified_20hz", "unified_50hz"), default="unified_20hz")
    parser.add_argument("--train_smoke", action="store_true")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()
    import gymnasium as gym
    import torch
    import warp as wp
    from isaaclab.utils.math import convert_camera_frame_orientation_convention, quat_apply, quat_mul

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import ZhangNavRLEnvCfg
    from isaaclab_navrl.zhang.config import ZhangTrainCfg
    from isaaclab_navrl.zhang.runner import ZhangTrainer

    cfg = ZhangNavRLEnvCfg()
    cfg.timing, cfg.variant = args.timing, "dcol"
    cfg.scene.num_envs, cfg.training_stage = 16, "finetune_s1"
    cfg.sim_to_real.enabled = True
    cfg.training_report_episodes = 1024
    cfg.num_static_obstacles, cfg.num_dynamic_obstacles = 300, 60
    cfg.sim.device = args.device
    cfg.configure_scene()
    dt = cfg.sim.dt
    result = {"status": "running", "training_acceptance": False}
    with launch_simulation(cfg, args):
        env = gym.make("Isaac-NavRL-Zhang-Hummingbird-Direct-v0", cfg=cfg).unwrapped
        try:
            env.reset()
            camera, lidar = env.scene["depth_camera"], env.scene["lidar"]
            env.get_zhang_observation()
            # The configured mount must match the actual sensor's optical frame.
            root = env.root_state()
            offset = camera.cfg.offset
            offset_q = convert_camera_frame_orientation_convention(
                root.new_tensor(offset.rot).expand(env.num_envs, -1), origin=offset.convention, target="ros"
            )
            actual_q = convert_camera_frame_orientation_convention(
                camera.data.quat_w_world.torch, origin="world", target="ros"
            )
            expected_q = quat_mul(root[:, 3:7], offset_q)
            torch.testing.assert_close((actual_q * expected_q).sum(-1).abs(), torch.ones_like(root[:, 0]))
            torch.testing.assert_close(
                camera.data.pos_w.torch,
                root[:, :3] + quat_apply(root[:, 3:7], root.new_tensor(offset.pos).expand(env.num_envs, -1)),
                atol=1e-5,
                rtol=1e-5,
            )
            env.perturbation.observation_buffer.set_time_lag(
                torch.arange(env.num_envs, device=env.device) % (env.perturbation.observation_buffer.history_length + 1)
            )
            times, lidar_times, physics_steps = [], [], []
            for _ in range(20):
                _, _, _, _, info = env.step(torch.zeros(16, 3, device=env.device))
                observation = env.get_zhang_observation()
                depth = observation["policy_depth_features"]
                assert torch.isfinite(depth).all() and torch.isfinite(env.root_state()).all()
                assert observation["depth_age_s"].min() >= 0 and observation["depth_age_s"].max() <= 0.100001
                assert (observation["depth_known_fraction"] > 0).all()
                assert (observation["depth_known_fraction"] <= 1).all()
                assert camera.data.output["distance_to_image_plane"].torch.shape == (16, 120, 160, 1)
                assert env._physical_substep == 1
                assert info["transition"]["dynamic_positions_by_substep"].shape == (1, 60, 3)
                times.append(wp.to_torch(camera._timestamp_last_update).clone())
                lidar_times.append(wp.to_torch(lidar._timestamp_last_update).clone())
                physics_steps.append(int(env._sim_step_counter))
            camera_delta = torch.stack(times).diff(dim=0)
            lidar_delta = torch.stack(lidar_times).diff(dim=0)
            # Compare lanes that were not reset; resets legitimately restart timestamps.
            valid_camera, valid_lidar = camera_delta >= 0, lidar_delta >= 0
            torch.testing.assert_close(
                camera_delta[valid_camera], torch.full_like(camera_delta[valid_camera], dt), atol=1e-6, rtol=0
            )
            torch.testing.assert_close(
                lidar_delta[valid_lidar], torch.full_like(lidar_delta[valid_lidar], dt), atol=1e-6, rtol=0
            )
            assert all(b - a == 1 for a, b in zip(physics_steps, physics_steps[1:], strict=False))
            assert env.physics_dt == env.step_dt == dt and env.curriculum.min_episodes == 1024
            tau = env.controller.rotor_parameters["tau_up"]
            torch.testing.assert_close(tau, torch.full_like(tau, 1 - 0.57 ** (dt / 0.01)))
            from isaaclab_navrl.zhang.losses import navrl_height_loss

            band = env.get_zhang_observation()["height_range"]
            point = env.root_state()[:, :3].detach().clone()
            point[:, 2] = band[:, 0] - 0.4
            point.requires_grad_()
            penalty = navrl_height_loss(point, band)
            gradient = torch.autograd.grad(penalty.sum(), point)[0]
            torch.testing.assert_close(penalty, torch.full_like(penalty, 0.04), atol=1e-6, rtol=0)
            torch.testing.assert_close(gradient[:, 2], torch.full_like(penalty, -0.4), atol=1e-6, rtol=0)
            result["navrl_height_real_episode_band_gradient"] = "passed"
            result.update(
                status="passed",
                physics_dt=env.physics_dt,
                policy_dt=env.step_dt,
                physics_steps=20,
                policy_steps=20,
                simulated_s=20 * dt,
                depth_delta_min_s=float(camera_delta[valid_camera].min()),
                depth_delta_max_s=float(camera_delta[valid_camera].max()),
                lidar_delta_min_s=float(lidar_delta[valid_lidar].min()),
                lidar_delta_max_s=float(lidar_delta[valid_lidar].max()),
                motor_alpha=float(tau[0, 0]),
                report_episodes=env.curriculum.min_episodes,
                depth_age_max_s=float(observation["depth_age_s"].max()),
                depth_known_fraction_mean=float(observation["depth_known_fraction"].mean()),
                action_delay_max_s=env.perturbation.action_buffer.history_length * dt,
            )
            if args.train_smoke:
                train_cfg = ZhangTrainCfg(
                    max_iterations=2,
                    timesteps=round(2 / dt) + 10,
                    physics_dt=dt,
                    policy_dt=dt,
                    gradient_diagnostic_updates=(1,),
                )
                trainer = ZhangTrainer(env, train_cfg, args.output.parent / "training")
                trainer.run()
                assert trainer.completed_updates == 2
                result["training_updates"] = trainer.completed_updates
        except BaseException as error:
            result.update(status="failed", error=repr(error))
            raise
        finally:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            env.close()


if __name__ == "__main__":
    main()
