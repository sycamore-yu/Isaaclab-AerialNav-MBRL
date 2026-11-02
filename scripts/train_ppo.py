"""Train one Navigation curriculum stage with Isaac Lab 3.0 and the RSL-RL runner."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from itertools import count
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import add_launcher_args, launch_simulation

from aerialnav.physics import create_navigation_env, parse_physics_args, runtime_metadata, select_physics
from aerialnav.source import verify_training_import
from aerialnav.training import write_json


def main():
    verify_training_import()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--report_episodes", type=int, default=None)
    parser.add_argument(
        "--max_iterations", type=int, default=None, help="Optional update limit for a smoke run; default unlimited"
    )
    parser.add_argument(
        "--stop_after_acceptance",
        action="store_true",
        help="Finish when this training stage reaches its success threshold",
    )
    parser.add_argument("--stage", default="pretrain_s0", choices=("pretrain_s0",))
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint from the preceding training stage")
    parser.add_argument("--lr", type=float, help="Override this stage's learning rate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--terrain_seed", type=int, default=0)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parse_physics_args(parser)
    if args.num_envs < 1 or (args.max_iterations is not None and args.max_iterations < 1):
        parser.error("num_envs and max_iterations must be positive")
    if args.lr is not None and not 0.0 < args.lr < float("inf"):
        parser.error("lr must be finite and positive")
    if args.report_episodes is not None and args.report_episodes < 1:
        parser.error("report_episodes must be positive")
    if args.stage == "pretrain_s0" and args.checkpoint is not None:
        parser.error("pretrain_s0 must start from random initialization, without a checkpoint")
    if args.stage != "pretrain_s0" and args.max_iterations is None and args.checkpoint is None:
        parser.error("Unlimited training after pretrain_s0 requires the preceding stage's checkpoint")
    log_dir = args.log_dir.resolve()
    if log_dir.exists() and any(log_dir.iterdir()):
        parser.error(f"Each training stage requires a new or empty output directory: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)
    parent_checkpoint = str(args.checkpoint.resolve()) if args.checkpoint is not None else None
    started = time.perf_counter()
    previous_elapsed_s = 0.0
    runner = None
    env = None
    result = {
        "method": "ppo",
        "physics": args.physics_preset,
        "frontend": args.frontend,
        "status": "running",
        "fresh": parent_checkpoint is None,
        "stage": args.stage,
        "parent_checkpoint": parent_checkpoint,
        "num_envs": args.num_envs,
        "completed_updates": 0,
        "stage_updates": 0,
        "environment_steps": 0,
        "max_updates": args.max_iterations,
        "stage_passed": False,
        "acceptance_passed": False,
        "stop_requested": None,
    }

    def request_stop(signum):
        result["stop_requested"] = str(signum)

    def elapsed_s():
        return previous_elapsed_s + time.perf_counter() - started

    def write_result(announce=False):
        result["elapsed_s"] = elapsed_s()
        if env is not None:
            result["curriculum"] = env.curriculum.state_dict()
            result["stage_passed"] = bool(env.curriculum.stage_passed)
            result["acceptance_passed"] = bool(
                args.stage == "finetune_s2"
                and env.curriculum.stage_passed
                and env.num_envs >= 1024
                and env.curriculum.min_episodes >= 1024
                and env.cfg.num_static_obstacles == 400
                and env.cfg.num_dynamic_obstacles == 100
                and env.cfg.sim_to_real.enabled
            )
            result["nonfinite_events"] = int(env.nonfinite_events)
        # This also records failures before the training helpers can be imported.
        write_json(log_dir / "training_result.json", result)
        if announce:
            print("TRAINING_RESULT " + json.dumps(result), flush=True)

    write_result(announce=True)
    try:
        import numpy as np
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        import aerialnav.tasks  # noqa: F401
        from aerialnav.agents.continuous_runner import ContinuousNavigationRunner
        from aerialnav.agents.rsl_rl_cfg import NavigationPpoRunnerCfg
        from aerialnav.tasks.navigation.config.hummingbird.env_cfg import NavigationEnvCfg
        from aerialnav.tasks.navigation.curriculum import STAGE_CONFIG
        from aerialnav.training import graceful_training_stop, record_curriculum

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        stage_config = STAGE_CONFIG[args.stage] | {
            "static_obstacles": 300,
            "dynamic_obstacles": 0,
            "randomization": False,
        }
        cfg = NavigationEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.seed = args.seed
        cfg.terrain_seed = args.terrain_seed
        stage_config = cfg.apply_training_stage(args.stage, stage_config=stage_config)
        cfg.training_report_episodes = args.report_episodes
        cfg.training_curriculum = True
        cfg.sim.device = args.device
        select_physics(cfg, args.physics_preset)
        cfg.configure_scene()
        from importlib.metadata import version

        from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(NavigationPpoRunnerCfg(), version("rsl-rl-lib"))
        agent_cfg.seed = args.seed
        agent_cfg.device = str(args.device)
        agent_cfg.max_iterations = args.max_iterations
        learning_rate = args.lr
        if learning_rate is None:
            learning_rate = 5e-5 if stage_config["randomization"] else agent_cfg.algorithm.learning_rate
        agent_cfg.algorithm.learning_rate = learning_rate

        if parent_checkpoint is not None:
            checkpoint_data = torch.load(parent_checkpoint, map_location="cpu", weights_only=True)
            parent_infos = checkpoint_data["infos"]
            predecessor = tuple(STAGE_CONFIG)[tuple(STAGE_CONFIG).index(args.stage) - 1]
            if parent_infos.get("method") != "ppo" or parent_infos.get("stage") != predecessor:
                raise ValueError(f"Expected a Navigation {predecessor} checkpoint, received {parent_checkpoint}")
            if args.max_iterations is None and not parent_infos["curriculum"]["stage_passed"]:
                raise ValueError(f"The parent checkpoint has not passed {predecessor}")
            del checkpoint_data, parent_infos

        initial_updates = 0
        initial_environment_steps = 0
        sim_to_real_config = asdict(cfg.sim_to_real)
        run_config = {
            "method": "ppo",
            "physics": args.physics_preset,
            "frontend": args.frontend,
            "fresh": parent_checkpoint is None,
            "stage": args.stage,
            "parent_checkpoint": parent_checkpoint,
            "num_envs": args.num_envs,
            "max_updates": args.max_iterations,
            "report_episodes": args.report_episodes or args.num_envs,
            "seed": args.seed,
            "terrain_seed": args.terrain_seed,
            "stop_after_acceptance": args.stop_after_acceptance,
            "static_obstacles": cfg.num_static_obstacles,
            "dynamic_obstacles": cfg.num_dynamic_obstacles,
            "curriculum": {"stage_config": stage_config},
            "sim_to_real": sim_to_real_config,
            "independent_evaluation": False,
            "physics_dt": cfg.sim.dt,
            "decimation": cfg.decimation,
            "episode_length_s": cfg.episode_length_s,
            "training": agent_cfg.to_dict(),
            "task": "Isaac-AerialNav-PPO-v0",
            "optimizer_differences_from_original": [
                "Official PPO uses a single Adam optimizer and squared value loss without ValueNorm.",
                "Official actor clipping includes the shared encoder; critic clipping covers its value head.",
                "Official PPO surrogate has no extra multiplication by action dimension.",
            ],
        }

        with launch_simulation(cfg, args), graceful_training_stop(SimpleNamespace(request_stop=request_stop)):
            try:
                env = create_navigation_env(run_config["task"], cfg, args)
                run_config.update(runtime_metadata(env))
                result.update(runtime_metadata(env))
                write_json(log_dir / "run_config.json", run_config)
                write_json(log_dir / "nominal_dynamics.json", env.nominal_dynamics)
                wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
                runner = ContinuousNavigationRunner(
                    wrapped, agent_cfg.to_dict(), log_dir=str(log_dir), device=args.device
                )
                if parent_checkpoint is not None:
                    loaded_infos = runner.load(parent_checkpoint, strict=True, map_location=args.device)
                    initial_updates = int(loaded_infos["completed_updates"])
                    previous_elapsed_s = float(loaded_infos["elapsed_s"])
                    initial_environment_steps = int(
                        loaded_infos.get(
                            "environment_steps", initial_updates * agent_cfg.num_steps_per_env * args.num_envs
                        )
                    )
                    result.update(completed_updates=initial_updates, environment_steps=initial_environment_steps)
                # Loading an optimizer restores its old rate; each stage uses its configured rate.
                runner.alg.learning_rate = learning_rate
                for group in runner.alg.optimizer.param_groups:
                    group["lr"] = learning_rate
                params = [p for group in runner.alg.optimizer.param_groups for p in group["params"]]
                if len(params) != len({id(p) for p in params}):
                    raise RuntimeError("Shared network parameters were registered twice in the optimizer.")
                runner.add_git_repo_to_log(__file__)

                def save_checkpoint(path):
                    path = Path(path)
                    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
                    try:
                        runner.save(
                            str(temporary),
                            infos={
                                "method": "ppo",
                                "physics": args.physics_preset,
                                **runtime_metadata(env),
                                "stage": args.stage,
                                "fresh": parent_checkpoint is None,
                                "parent_checkpoint": parent_checkpoint,
                                "completed_updates": result["completed_updates"],
                                "stage_updates": result["stage_updates"],
                                "environment_steps": result["environment_steps"],
                                "elapsed_s": elapsed_s(),
                                "curriculum": env.curriculum.state_dict(),
                                "sim_to_real": sim_to_real_config,
                                "stop_requested": result["stop_requested"],
                            },
                        )
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)

                write_result()
                updates = count() if args.max_iterations is None else range(args.max_iterations)
                for stage_update in updates:
                    update_started = time.perf_counter()
                    global_update = initial_updates + stage_update + 1
                    runner.current_learning_iteration = global_update - 1
                    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
                    result["completed_updates"] = global_update
                    result["stage_updates"] = stage_update + 1
                    result["environment_steps"] = (
                        initial_environment_steps + (stage_update + 1) * agent_cfg.num_steps_per_env * args.num_envs
                    )
                    with (log_dir / "updates.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "update": global_update,
                                    "environment_steps": result["environment_steps"],
                                    "rollout_steps": agent_cfg.num_steps_per_env,
                                    "elapsed_s": elapsed_s(),
                                    "update_s": time.perf_counter() - update_started,
                                },
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    record_curriculum(env.curriculum, global_update, log_dir, runner.logger.writer, save_checkpoint)
                    if global_update % agent_cfg.save_interval == 0:
                        save_checkpoint(log_dir / "checkpoint_latest.pt")
                    if env.curriculum.stage_passed and "stage_checkpoint" not in result:
                        stage_checkpoint = log_dir / "checkpoint_stage.pt"
                        save_checkpoint(stage_checkpoint)
                        result["stage_checkpoint"] = str(stage_checkpoint)
                    write_result()
                    if result["stop_requested"] is not None:
                        break
                    if args.stop_after_acceptance and env.curriculum.stage_passed:
                        break
                if not all(torch.isfinite(p).all() for p in params):
                    raise FloatingPointError("Training produced non-finite model parameters.")
                stopped = result["stop_requested"] is not None
                checkpoint = log_dir / ("checkpoint_stopped.pt" if stopped else "checkpoint_final.pt")
                save_checkpoint(checkpoint)
                # A stop can also arrive while writing the normal final checkpoint.
                if not stopped and result["stop_requested"] is not None:
                    stopped = True
                    checkpoint = log_dir / "checkpoint_stopped.pt"
                    save_checkpoint(checkpoint)
                result.update(
                    status="stopped" if stopped else "completed",
                    checkpoint=str(checkpoint),
                    nonfinite_events=int(env.nonfinite_events),
                )
            except BaseException as error:
                result.update(status="failed", error=f"{type(error).__name__}: {error}")
                if runner is not None:
                    result["last_iteration_index"] = runner.current_learning_iteration
                raise
            finally:
                # Kit can terminate Python on context exit, so persist the outcome first.
                try:
                    write_result(announce=True)
                finally:
                    if runner is not None and runner.logger.writer is not None:
                        runner.logger.writer.close()
                    if env is not None:
                        env.close()
    except BaseException as error:
        # Includes imports, checkpoint validation, configuration, and launcher entry.
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_result()
        raise


if __name__ == "__main__":
    main()
