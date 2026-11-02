"""Evaluate completed episodes on one independently launched Navigation difficulty."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from aerialnav.physics import create_navigation_env, parse_physics_args, runtime_metadata, select_physics


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--method", choices=("ppo", "diffnav"), default="diffnav")
    parser.add_argument("--geometry_backend", choices=("native", "curobo_v2"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--terrain_seed", type=int, default=10000)
    parser.add_argument("--static_obstacles", type=int, default=350)
    parser.add_argument("--dynamic_obstacles", type=int, default=100)
    parser.add_argument(
        "--stage", choices=("pretrain_s0", "pretrain_s1", "pretrain_s2", "finetune_s1"), default="finetune_s1"
    )
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parse_physics_args(parser)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.output.exists():
        parser.error(f"evaluation output already exists: {args.output}")

    import torch

    import aerialnav.tasks  # noqa: F401
    from aerialnav.evaluation import EpisodeEvaluation
    from aerialnav.tasks.navigation.config.hummingbird.env_cfg import DiffNavEnvCfg, NavigationEnvCfg
    from aerialnav.training import EPISODE_END_CONTRACT

    cfg = (DiffNavEnvCfg if args.method == "diffnav" else NavigationEnvCfg)()
    if args.geometry_backend is not None:
        if args.method != "diffnav":
            parser.error("--geometry_backend requires --method diffnav")
        cfg.geometry_backend = args.geometry_backend
    cfg.training_curriculum = False
    cfg.scene.num_envs = args.episodes
    cfg.num_static_obstacles = args.static_obstacles
    cfg.num_dynamic_obstacles = args.dynamic_obstacles
    cfg.seed, cfg.terrain_seed = args.seed, args.terrain_seed
    cfg.evaluation = args.stage is None
    if args.stage:
        cfg.apply_training_stage(args.stage)
    cfg.sim.device = args.device
    select_physics(cfg, args.physics_preset)
    cfg.configure_scene()
    task = "Isaac-AerialNav-Depth-v0" if args.method == "diffnav" else "Isaac-AerialNav-PPO-v0"
    with launch_simulation(cfg, args):
        env = create_navigation_env(task, cfg, args)
        try:
            if args.method == "ppo":
                from aerialnav.agents.inference import load_ppo_policy

                policy = load_ppo_policy(env, args.checkpoint)
            else:
                from aerialnav.diffnav.inference import load_diffnav_policy

                policy = load_diffnav_policy(env, args.checkpoint)
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
                "physics": args.physics_preset,
                **runtime_metadata(env),
                **({"recurrent_model": policy.cfg.recurrent_model} if args.method == "diffnav" else {}),
                "checkpoint": str(args.checkpoint.resolve()),
                "seed": args.seed,
                "terrain_seed": args.terrain_seed,
                "static_obstacles": cfg.num_static_obstacles,
                "dynamic_obstacles": cfg.num_dynamic_obstacles,
                "physics_dt": env.physics_dt,
                "policy_dt": env.step_dt,
                "episode_timeout_s": cfg.episode_length_s,
                "stage": args.stage,
                "randomization": cfg.sim_to_real.enabled,
                "episode_end_contract": EPISODE_END_CONTRACT,
                "wait_before_truncate_s": cfg.wait_before_truncate,
                "height_bounds_reset": getattr(cfg, "height_bounds_reset", True),
                **(
                    {
                        "training_collision_metric": policy.training_collision_metric,
                        "training_geometry_backend": policy.training_geometry_backend,
                        "evaluation_geometry_backend": cfg.geometry_backend,
                        "target_speed_range": list(cfg.target_speed_range or (1.0, 6.0)),
                        "position_integration": policy.cfg.position_integration,
                    }
                    if args.method == "diffnav"
                    else {}
                ),
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
