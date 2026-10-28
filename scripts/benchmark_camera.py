"""Measure full-resolution depth and one complete Zhang policy transition."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def summarize(samples):
    ordered = sorted(samples)
    fractional_index = 0.95 * (len(ordered) - 1)
    lower = int(fractional_index)
    upper = min(lower + 1, len(ordered) - 1)
    percentile = ordered[lower] + (ordered[upper] - ordered[lower]) * (fractional_index - lower)
    return {
        "mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples),
        "p95_ms": percentile, "min_ms": min(samples), "max_ms": max(samples),
        "samples_ms": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--terrain_seed", type=int, default=0)
    parser.add_argument("--static_obstacles", type=int, default=350)
    parser.add_argument("--dynamic_obstacles", type=int, default=100)
    parser.add_argument("--pixel_noise_std", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=Path("logs/benchmark/camera_16.json"))
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 1 or args.num_envs < 1:
        parser.error("num_envs, samples, and warmup must be positive")

    import gymnasium as gym
    import torch

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import ZhangNavRLEnvCfg

    cfg = ZhangNavRLEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed, cfg.terrain_seed = args.seed, args.terrain_seed
    cfg.num_static_obstacles, cfg.num_dynamic_obstacles = args.static_obstacles, args.dynamic_obstacles
    cfg.evaluation = True  # Fixed starts, nominal dynamics, fixed mount and intrinsics.
    cfg.sim.device = args.device
    cfg.configure_scene()
    with launch_simulation(cfg, args):
        env = gym.make("Isaac-NavRL-Zhang-Hummingbird-Direct-v0", cfg=cfg).unwrapped
        try:
            env.reset()
            camera = env.scene["depth_camera"]
            # Keep the training image-noise cost while holding physical parameters fixed.
            camera.cfg.pixel_noise_std = args.pixel_noise_std
            assert camera.image_shape == (480, 640), "This benchmark must trace all original camera pixels"
            if not str(env.device).startswith("cuda"):
                raise ValueError("Use the actual CUDA simulation device for this performance measurement")
            action = torch.zeros(env.num_envs, 3, device=env.device)

            def refresh_camera():
                camera.update(env.step_dt)
                return camera.data.output["policy_depth"].torch

            def complete_policy_transition():
                env.step(action)
                return env.get_zhang_observation()["depth_features"]

            def measure(operation):
                for _ in range(args.warmup):
                    operation()
                torch.cuda.synchronize(env.device)
                torch.cuda.reset_peak_memory_stats(env.device)
                elapsed = []
                last = None
                for _ in range(args.samples):
                    torch.cuda.synchronize(env.device)
                    started = time.perf_counter()
                    last = operation()
                    # PhysX and Warp may use several streams; wall time waits for all of them.
                    torch.cuda.synchronize(env.device)
                    elapsed.append((time.perf_counter() - started) * 1000)
                assert torch.isfinite(last).all()
                stats = summarize(elapsed)
                stats["torch_peak_allocated_mb"] = torch.cuda.max_memory_allocated(env.device) / 2**20
                return stats

            camera_stats = measure(refresh_camera)
            # The second measurement includes all ten physical steps, dynamic motion,
            # collision checks, observations, and the synchronized depth-camera frame.
            env.reset()
            transition_stats = measure(complete_policy_transition)
            free_bytes, total_bytes = torch.cuda.mem_get_info(env.device)
            result = {
                "method": "zhang", "num_envs": env.num_envs,
                "seed": cfg.seed, "terrain_seed": cfg.terrain_seed,
                "static_obstacles": cfg.num_static_obstacles, "dynamic_obstacles": cfg.num_dynamic_obstacles,
                "physics_dt": env.physics_dt, "policy_dt": env.step_dt,
                "source_resolution": [480, 640], "pooled_resolution": [12, 16],
                "full_resolution_rays_per_frame": env.num_envs * 480 * 640,
                "camera_class": f"{type(camera).__module__}.{type(camera).__name__}",
                "camera_batch_size": camera.cfg.batch_size,
                "pixel_noise_std": camera.cfg.pixel_noise_std,
                "nominal_physics_parameters": True,
                "warmup": args.warmup, "samples": args.samples,
                "camera_fixed_scene": camera_stats,
                "env_step_with_depth": transition_stats,
                "policy_transitions_per_s": env.num_envs * 1000 / transition_stats["mean_ms"],
                "device_free_mb": free_bytes / 2**20, "device_total_mb": total_bytes / 2**20,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print("CAMERA_BENCHMARK " + json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
