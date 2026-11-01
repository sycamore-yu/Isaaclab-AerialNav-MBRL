"""Exercise real simulator reset/step and shared-world contracts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from aerialnav.physics import create_navigation_env, parse_physics_args, runtime_metadata, select_physics


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--method", choices=("ppo", "diffnav"), default="ppo")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--dynamic_obstacles", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("logs/smoke/result.json"))
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parse_physics_args(parser)

    import torch

    import aerialnav.tasks  # noqa: F401
    from aerialnav.tasks.navigation.config.hummingbird.env_cfg import DiffNavEnvCfg, NavigationEnvCfg

    cfg = (DiffNavEnvCfg if args.method == "diffnav" else NavigationEnvCfg)()
    cfg.training_curriculum = False
    cfg.scene.num_envs = args.num_envs
    cfg.num_dynamic_obstacles = args.dynamic_obstacles
    cfg.sim.device = args.device
    select_physics(cfg, args.physics_preset)
    cfg.configure_scene()
    task = "Isaac-AerialNav-Depth-v0" if args.method == "diffnav" else "Isaac-AerialNav-PPO-v0"
    with launch_simulation(cfg, args):
        env = create_navigation_env(task, cfg, args)
        try:
            observation, _ = env.reset()
            assert observation["policy"].shape == (args.num_envs, 202)
            assert torch.isfinite(observation["policy"]).all()
            torch.cuda.reset_peak_memory_stats()
            free_memory, total_memory = torch.cuda.mem_get_info()
            initial_height = env.root_state()[:, 2].clone()
            start = time.perf_counter()
            done_count, min_range = 0, 4.0
            hover_height_error, hover_done = 0.0, 0
            depth_shape, depth_change = None, None
            previous_depth = None
            for step in range(args.steps):
                action = torch.zeros(args.num_envs, 3, device=env.device)
                if args.method == "ppo" and step >= args.steps // 2:
                    action[:, 0] = 1.0
                observation, reward, terminated, timeout, info = env.step(action)
                assert torch.isfinite(env.root_state()).all(), f"Non-finite state at step {step}"
                assert torch.isfinite(observation["policy"]).all()
                assert torch.isfinite(reward).all()
                done_count += int((terminated | timeout).sum())
                if step < args.steps // 2:
                    hover_height_error = max(
                        hover_height_error, float((env.root_state()[:, 2] - initial_height).abs().max())
                    )
                    hover_done += int((terminated | timeout).sum())
                min_range = min(min_range, float(4 - observation["policy"][:, 8:152].max()))
                assert "transition" in info
                if args.method == "diffnav":
                    depth = env.get_diffnav_observation()["depth_features"]
                    assert torch.isfinite(depth).all()
                    depth_shape = list(depth.shape)
                    if previous_depth is not None:
                        depth_change = float((depth - previous_depth).abs().max())
                    previous_depth = depth.clone()
            elapsed = time.perf_counter() - start
            heading = env.target_dir[:, :2] / env.target_dir[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
            forward_speed = float((env.root_state()[:, 7:9] * heading).sum(-1).mean())
            if args.method == "ppo":
                assert hover_done == 0, "Zero velocity terminated an initially clear episode."
                assert hover_height_error < 0.2, "Hover height drift exceeds 0.2 m."
                if args.steps >= 128:
                    assert forward_speed > 0.2, "The commanded forward velocity did not produce forward motion."
            obstacle_position = env.motion.position.clone()
            obstacle_steps = env.motion.step_count
            env._reset_idx(torch.tensor([0], device=env.device, dtype=torch.int32))
            torch.testing.assert_close(env.motion.position, obstacle_position)
            assert env.motion.step_count == obstacle_steps
            result = {
                "method": args.method,
                "physics": args.physics_preset,
                **runtime_metadata(env),
                "num_envs": args.num_envs,
                "steps": args.steps,
                "static_obstacles": cfg.num_static_obstacles,
                "dynamic_obstacles": cfg.num_dynamic_obstacles,
                "physics_dt": env.physics_dt,
                "policy_dt": env.step_dt,
                "episode_steps": env.max_episode_length,
                "elapsed_s": elapsed,
                "transitions_per_s": args.num_envs * args.steps / elapsed,
                "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
                "device_free_after_reset_mb": free_memory / 2**20,
                "device_total_mb": total_memory / 2**20,
                "completed_episodes": done_count,
                "min_lidar_range_m": min_range,
                "hover_max_height_error_m": hover_height_error,
                "hover_completed_episodes": hover_done,
                "forward_speed_after_command_m_s": forward_speed,
                "depth_shape": depth_shape,
                "last_depth_change_m": depth_change,
                "shared_obstacles_survive_drone_reset": True,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print("SMOKE_RESULT " + json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
