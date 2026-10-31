"""Bounded checkpoint fork for diagnostics; never resumes a live experiment or grants acceptance."""

import argparse
import hashlib
import time
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_navrl.training import write_json


def main():
    process_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--updates", type=int, default=4)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()
    if min(args.num_envs, args.updates) < 1:
        parser.error("num_envs and updates must be positive")
    args.output.mkdir(parents=True, exist_ok=False)

    import gymnasium as gym
    import torch
    from profile_step import instrument_methods

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.reference.diagnostics import compare_dynamic_metrics
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import ZhangNavRLEnvCfg
    from isaaclab_navrl.zhang.actor_critic import CostActorCritic
    from isaaclab_navrl.zhang.config import ZhangTrainCfg
    from isaaclab_navrl.zhang.runner import ZhangTrainer

    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location=args.device, weights_only=True)
    if payload.get("format_version") != 3 or payload.get("method") != "zhang":
        raise ValueError("Diagnostic forks require a version-3 Zhang checkpoint")
    saved_env = payload["environment"]
    train = ZhangTrainCfg(**payload["train_cfg"])
    train.max_iterations, train.stop_after_acceptance = args.updates, False
    train.gradient_diagnostic_updates = tuple(
        range(payload["completed_updates"] + 1, payload["completed_updates"] + args.updates + 1)
    )
    cfg = ZhangNavRLEnvCfg()
    cfg.variant = train.variant
    cfg.timing = {(0.01, 0.1): "baseline", (0.05, 0.05): "unified_20hz", (0.02, 0.02): "unified_50hz"}[
        (train.physics_dt, train.policy_dt)
    ]
    cfg.scene.num_envs = args.num_envs
    cfg.seed, cfg.terrain_seed = saved_env["seed"], saved_env["terrain_seed"]
    cfg.height_target_range = train.height_target_range
    cfg.training_stage = saved_env["training_stage"]
    cfg.training_curriculum = True
    cfg.training_report_episodes = 1024
    cfg.num_static_obstacles, cfg.num_dynamic_obstacles = saved_env["static_obstacles"], saved_env["dynamic_obstacles"]
    for name, value in saved_env["sim_to_real"].items():
        setattr(cfg.sim_to_real, name, value)
    cfg.capture_terminal_observation = train.algorithm != "bptt"
    cfg.sim.device = args.device
    cfg.configure_scene()
    identity = {
        "purpose": "checkpoint_diagnostic_fork",
        "acceptance_passed": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_update": payload["completed_updates"],
        "num_envs": args.num_envs,
        "source_num_envs": payload["num_envs"],
        "train": asdict(train),
        "environment": saved_env,
        "initial_state": "fresh seeded environment; restored actor, optimizer, normalizer and critic; not trajectory replay",
        "timing": "CUDA-synchronized inclusive method times; instrumented diagnostics, not training throughput",
        "source_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ("src", "scripts")
            for p in (Path(__file__).resolve().parents[1] / directory).rglob("*.py")
        },
    }
    write_json(args.output / "diagnostic_identity.json", identity)
    torch.manual_seed(cfg.seed)
    with launch_simulation(cfg, args):
        env = gym.make("Isaac-NavRL-Zhang-Hummingbird-Direct-v0", cfg=cfg).unwrapped
        try:
            trainer = ZhangTrainer(env, train, args.output)
            trainer.process_started_at = process_started
            trainer.model.load_state_dict(payload["model_state_dict"], strict=True)
            trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
            trainer.normalizer.load_state_dict(payload["normalizer_state_dict"], strict=True)
            trainer._pending_critic_state = payload["actor_critic_state_dict"]
            trainer.completed_updates = payload["completed_updates"]
            trainer.schedule_updates = payload["schedule_updates"]
            trainer.lr_scale, trainer.lr_cap = payload["lr_scale"], payload["lr_cap"]
            trainer.parent_checkpoint = str(checkpoint)
            if trainer.scheduler is not None:
                trainer.scheduler.load_state_dict(payload["scheduler_state_dict"])
            if trainer.horizon_controller is not None:
                trainer.horizon_controller.load_state_dict(payload["horizon_controller_state_dict"])
            methods = [
                ("env.step", env, "step"),
                ("depth.observation", env, "get_zhang_observation"),
                ("static.distance", env.terrain_distance, "signed_distance"),
                ("policy.forward", trainer.model, "forward_checkpointed"),
                ("autograd.backward", torch.autograd, "backward"),
                ("actor.optimizer", trainer.optimizer, "step"),
                ("critic.fit", CostActorCritic, "fit"),
            ]

            def synchronize():
                if torch.device(env.device).type == "cuda":
                    torch.cuda.synchronize(env.device)

            with instrument_methods(methods, synchronize) as timing:
                result = trainer.run()
            write_json(args.output / "profile.json", timing)
            state = env.root_state().detach()
            lookahead = torch.linspace(0, env.step_dt, 10, device=env.device)[None, :, None]
            points = state[:, None, :3] + state[:, None, 7:10] * lookahead
            write_json(
                args.output / "dynamic_metrics.json",
                compare_dynamic_metrics(
                    points,
                    env.motion.position,
                    env.motion.size,
                ),
            )
            result.update(purpose="checkpoint_diagnostic_fork", acceptance_passed=False)
            write_json(args.output / "diagnostic_result.json", result)
            write_json(args.output / "training_result.json", result)
        finally:
            env.close()


if __name__ == "__main__":
    main()
