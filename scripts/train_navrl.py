"""Fresh NavRL training with the Isaac Lab 3.0 / RSL-RL 5.5.1 runner."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--terrain_seed", type=int, default=0)
    parser.add_argument("--static_obstacles", type=int, default=350)
    parser.add_argument("--dynamic_obstacles", type=int, default=100)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()
    if args.num_envs < 1 or args.max_iterations < 1:
        parser.error("num_envs and max_iterations must be positive")
    log_dir = args.log_dir.resolve()
    if log_dir.exists() and any(log_dir.iterdir()):
        parser.error(f"Fresh training requires an empty output directory: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)

    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.agents.rsl_rl_cfg import NavRLPpoRunnerCfg
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import NavRLEnvCfg

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = NavRLEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.terrain_seed = args.terrain_seed
    cfg.num_static_obstacles = args.static_obstacles
    cfg.num_dynamic_obstacles = args.dynamic_obstacles
    cfg.sim.device = args.device
    cfg.configure_scene()
    agent_cfg = NavRLPpoRunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.device = str(args.device)
    agent_cfg.max_iterations = args.max_iterations
    run_config = {
        "method": "navrl", "fresh": True, "num_envs": args.num_envs,
        "max_updates": args.max_iterations, "seed": args.seed, "terrain_seed": args.terrain_seed,
        "static_obstacles": args.static_obstacles, "dynamic_obstacles": args.dynamic_obstacles,
        "physics_dt": cfg.sim.dt, "decimation": cfg.decimation,
        "episode_length_s": cfg.episode_length_s, "training": agent_cfg.to_dict(),
        "task": "Isaac-NavRL-Hummingbird-Direct-v0",
        "optimizer_differences_from_original": [
            "Official PPO uses a single Adam optimizer and squared value loss without ValueNorm.",
            "Official actor clipping includes the shared encoder; critic clipping covers its value head.",
            "Official PPO surrogate has no extra multiplication by action dimension.",
        ],
    }
    (log_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    started = time.perf_counter()
    result = {
        "method": "navrl", "status": "running", "fresh": True,
        "num_envs": args.num_envs, "completed_updates": 0, "max_updates": args.max_iterations,
    }

    def write_result():
        result["elapsed_s"] = time.perf_counter() - started
        (log_dir / "training_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("TRAINING_RESULT " + json.dumps(result), flush=True)

    write_result()
    runner = None
    env = None
    try:
        with launch_simulation(cfg, args):
            try:
                env = gym.make(run_config["task"], cfg=cfg).unwrapped
                wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
                runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=str(log_dir), device=args.device)
                params = [p for group in runner.alg.optimizer.param_groups for p in group["params"]]
                if len(params) != len({id(p) for p in params}):
                    raise RuntimeError("Shared network parameters were registered twice in the optimizer.")
                runner.add_git_repo_to_log(__file__)
                runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
                if not all(torch.isfinite(p).all() for p in params):
                    raise FloatingPointError("Training produced non-finite model parameters.")
                checkpoint = log_dir / "checkpoint_final.pt"
                runner.save(str(checkpoint), infos={"fresh": True, "completed_updates": args.max_iterations})
                result.update(
                    status="completed", completed_updates=args.max_iterations,
                    checkpoint=str(checkpoint), nonfinite_events=int(env.nonfinite_events),
                )
            except BaseException as error:
                result.update(status="failed", error=f"{type(error).__name__}: {error}")
                if runner is not None:
                    result["last_iteration_index"] = runner.current_learning_iteration
                raise
            finally:
                # KitLauncher.close() can terminate Python during context exit.
                # Persist the outcome while still inside the launched runtime.
                try:
                    write_result()
                finally:
                    if env is not None:
                        env.close()
    except BaseException as error:
        # This covers launcher entry failures, before the inner block is reached.
        if result["status"] == "running":
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_result()
        raise


if __name__ == "__main__":
    main()
