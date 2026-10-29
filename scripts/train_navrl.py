"""Fresh NavRL training with the Isaac Lab 3.0 / RSL-RL 5.5.1 runner."""

from __future__ import annotations

import argparse
import json
import random
import time
from itertools import count
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=None, help="Optional limit for a smoke run; default unlimited")
    parser.add_argument("--stop_after_acceptance", action="store_true",
                        help="Finish after 100-obstacle acceptance and the first 120-obstacle training report")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--terrain_seed", type=int, default=0)
    parser.add_argument("--static_obstacles", type=int, default=350)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()
    if args.num_envs < 1 or (args.max_iterations is not None and args.max_iterations < 1):
        parser.error("num_envs and max_iterations must be positive")
    log_dir = args.log_dir.resolve()
    if log_dir.exists() and any(log_dir.iterdir()):
        parser.error(f"Fresh training requires an empty output directory: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)

    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.agents.continuous_runner import ContinuousNavRLRunner
    from isaaclab_navrl.agents.rsl_rl_cfg import NavRLPpoRunnerCfg
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import NavRLEnvCfg
    from isaaclab_navrl.training import record_curriculum, write_json

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = NavRLEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.terrain_seed = args.terrain_seed
    cfg.num_static_obstacles = args.static_obstacles
    cfg.training_curriculum = True
    cfg.num_dynamic_obstacles = 60
    cfg.sim.device = args.device
    cfg.configure_scene()
    agent_cfg = NavRLPpoRunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.device = str(args.device)
    agent_cfg.max_iterations = args.max_iterations
    run_config = {
        "method": "navrl", "fresh": True, "num_envs": args.num_envs,
        "max_updates": args.max_iterations, "seed": args.seed, "terrain_seed": args.terrain_seed,
        "stop_after_acceptance": args.stop_after_acceptance,
        "static_obstacles": args.static_obstacles, "dynamic_obstacles": 60,
        "curriculum": [60, 80, 100, 120], "independent_evaluation": False,
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

    def write_result(announce=False):
        result["elapsed_s"] = time.perf_counter() - started
        if env is not None:
            result["curriculum"] = env.curriculum.state_dict()
            result["acceptance_passed"] = (
                env.curriculum.acceptance_passed and args.num_envs == 1024 and args.static_obstacles == 350
            )
            result["nonfinite_events"] = int(env.nonfinite_events)
        write_json(log_dir / "training_result.json", result)
        if announce:
            print("TRAINING_RESULT " + json.dumps(result), flush=True)

    runner = None
    env = None
    write_result(announce=True)
    try:
        with launch_simulation(cfg, args):
            try:
                env = gym.make(run_config["task"], cfg=cfg).unwrapped
                wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
                runner = ContinuousNavRLRunner(wrapped, agent_cfg.to_dict(), log_dir=str(log_dir), device=args.device)
                params = [p for group in runner.alg.optimizer.param_groups for p in group["params"]]
                if len(params) != len({id(p) for p in params}):
                    raise RuntimeError("Shared network parameters were registered twice in the optimizer.")
                runner.add_git_repo_to_log(__file__)

                def save_checkpoint(path):
                    runner.save(str(path), infos={
                        "fresh": True, "completed_updates": result["completed_updates"],
                        "curriculum": env.curriculum.state_dict(),
                    })

                updates = count() if args.max_iterations is None else range(args.max_iterations)
                for update in updates:
                    runner.current_learning_iteration = update
                    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
                    result["completed_updates"] = update + 1
                    result["environment_steps"] = (update + 1) * agent_cfg.num_steps_per_env * args.num_envs
                    record_curriculum(env.curriculum, update + 1, log_dir, runner.logger.writer, save_checkpoint)
                    if (update + 1) % agent_cfg.save_interval == 0:
                        save_checkpoint(log_dir / "checkpoint_latest.pt")
                    if env.curriculum.ready_to_advance:
                        env.advance_curriculum()
                        runner.logger.cur_reward_sum.zero_()
                        runner.logger.cur_episode_length.zero_()
                        runner.logger.rewbuffer.clear()
                        runner.logger.lenbuffer.clear()
                        print(f"CURRICULUM -> {env.cfg.num_dynamic_obstacles} at update {update + 1}", flush=True)
                    write_result()
                    if (args.stop_after_acceptance and result["acceptance_passed"]
                            and 120 in env.curriculum.best_by_stage):
                        break
                if not all(torch.isfinite(p).all() for p in params):
                    raise FloatingPointError("Training produced non-finite model parameters.")
                checkpoint = log_dir / "checkpoint_final.pt"
                save_checkpoint(checkpoint)
                result.update(
                    status="completed",
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
                    write_result(announce=True)
                finally:
                    if runner is not None and runner.logger.writer is not None:
                        runner.logger.writer.close()
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
