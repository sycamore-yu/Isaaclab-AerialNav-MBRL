"""Evaluate first arrival on one independently launched NavRL difficulty."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--method", choices=("navrl", "zhang"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--terrain_seed", type=int, default=10000)
    parser.add_argument("--static_obstacles", type=int, default=350)
    parser.add_argument("--dynamic_obstacles", type=int, default=100)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.output.exists():
        parser.error(f"evaluation output already exists: {args.output}")

    import gymnasium as gym
    import torch

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.evaluation import EpisodeEvaluation
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import NavRLEnvCfg, ZhangNavRLEnvCfg

    cfg = (ZhangNavRLEnvCfg if args.method == "zhang" else NavRLEnvCfg)()
    cfg.scene.num_envs = args.episodes
    cfg.num_static_obstacles = args.static_obstacles
    cfg.num_dynamic_obstacles = args.dynamic_obstacles
    cfg.seed, cfg.terrain_seed = args.seed, args.terrain_seed
    cfg.evaluation = True
    cfg.sim.device = args.device
    cfg.configure_scene()
    task = "Isaac-NavRL-Zhang-Hummingbird-Direct-v0" if args.method == "zhang" else "Isaac-NavRL-Hummingbird-Direct-v0"
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            if args.method == "navrl":
                from isaaclab_navrl.agents.inference import load_navrl_policy

                policy = load_navrl_policy(env, args.checkpoint)
            else:
                from isaaclab_navrl.zhang.inference import load_zhang_policy

                policy = load_zhang_policy(env, args.checkpoint)
            observation, _ = env.reset()
            evaluation = EpisodeEvaluation(
                env.root_state()[:, :3].cpu().tolist(), env.target_pos.cpu().tolist(), args.seed
            )
            start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(env.max_episode_length):
                    action = policy(observation)
                    observation, _, _, _, info = env.step(action)
                    evaluation.update(info["transition"])
                    if evaluation.complete:
                        break
            result = {
                "method": args.method,
                "checkpoint": str(args.checkpoint.resolve()),
                "seed": args.seed,
                "terrain_seed": args.terrain_seed,
                "static_obstacles": args.static_obstacles,
                "dynamic_obstacles": args.dynamic_obstacles,
                "physics_dt": env.physics_dt,
                "policy_dt": env.step_dt,
                "episode_timeout_s": cfg.episode_length_s,
                "elapsed_s": time.perf_counter() - start,
                "summary": evaluation.summary(),
                "episodes": evaluation.rows,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print("EVALUATION_RESULT " + json.dumps(result["summary"], allow_nan=False), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
