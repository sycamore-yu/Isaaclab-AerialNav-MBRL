"""Train one Zhang BPTT, SHAC, or AHAC curriculum stage on Isaac Lab 3.0 PhysX."""

import argparse
import os
import time
from dataclasses import asdict, fields
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_navrl.source import verify_training_import
from isaaclab_navrl.training import STAGE_CONFIG, STAGES, graceful_training_stop, write_json
from isaaclab_navrl.zhang.cli import add_train_args, parse_train_args


def main():
    process_started = float(os.environ.get("NAVRL_STAGE_STARTED_MONOTONIC", time.monotonic()))
    verify_training_import()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--report_episodes", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None, help="Optional smoke-run limit; default unlimited")
    parser.add_argument("--stop_after_acceptance", action="store_true", help="Finish when this stage reaches 80%")
    parser.add_argument("--stage", choices=STAGES, default="pretrain_s0")
    parser.add_argument("--checkpoint", type=Path, help="Previous stage checkpoint from this experiment")
    parser.add_argument(
        "--fresh_stage",
        action="store_true",
        help="Explicit direct-stage comparison: random initialization without an earlier stage",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--terrain_seed", type=int, default=0)
    add_train_args(parser)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()
    overrides = {
        key: getattr(args, key)
        for key in ("num_dynamic_obstacles", "target_speed_min", "target_speed_max")
        if getattr(args, key) is not None
    }
    try:
        parse_train_args([token for key, value in overrides.items() for token in ("--" + key, str(value))])
    except ValueError as exc:
        parser.error(str(exc))
    if args.velocity_loss_mode is None:
        args.velocity_loss_mode = "per_step" if args.algorithm in ("shac", "ahac") else "full_window"
    if args.num_envs < 1 or (args.max_iterations is not None and args.max_iterations < 1):
        parser.error("num_envs and max_iterations must be positive")
    if args.camera_batch_size < 1:
        parser.error("camera_batch_size must be positive")
    if args.report_episodes is not None and args.report_episodes < 1:
        parser.error("report_episodes must be positive")
    if args.algorithm == "bptt" and args.critic_value_transform != "linear":
        parser.error("--critic_value_transform softplus requires --algorithm shac or ahac")
    if args.stage == "pretrain_s0" and args.checkpoint is not None:
        parser.error("Pretraining S0 must start from random initialization")
    if args.fresh_stage and (args.stage == "pretrain_s0" or args.checkpoint is not None):
        parser.error("--fresh_stage requires a later stage without --checkpoint")
    if args.stage != "pretrain_s0" and args.checkpoint is None and args.max_iterations is None and not args.fresh_stage:
        parser.error("Later stages require the previous checkpoint or explicit --fresh_stage comparison")
    if args.log_dir.exists() and any(args.log_dir.iterdir()):
        parser.error("Each training stage requires a new or empty --log_dir")
    args.lr = (
        args.lr
        if args.lr is not None
        else (5e-5 if args.stage.startswith("finetune") and not args.fresh_stage else 0.001)
    )

    import gymnasium as gym
    import torch

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import ZhangNavRLEnvCfg
    from isaaclab_navrl.zhang.config import ZhangTrainCfg
    from isaaclab_navrl.zhang.runner import ZhangTrainer

    stage = STAGE_CONFIG[args.stage].copy()
    if args.num_dynamic_obstacles is not None:
        stage["dynamic_obstacles"] = args.num_dynamic_obstacles
    cfg = ZhangNavRLEnvCfg()
    cfg.variant = args.variant
    cfg.timing = args.timing
    cfg.training_report_episodes = args.report_episodes
    cfg.height_target_range = (args.height_target_min, args.height_target_max)
    cfg.capture_terminal_observation = args.algorithm in ("shac", "ahac")
    cfg.scene.num_envs = args.num_envs
    cfg.seed, cfg.terrain_seed = args.seed, args.terrain_seed
    cfg.num_static_obstacles = stage["static_obstacles"]
    cfg.num_dynamic_obstacles = stage["dynamic_obstacles"]
    cfg.training_stage = args.stage
    cfg.training_stage_config = stage
    if args.target_speed_min is not None:
        cfg.target_speed_range = (args.target_speed_min, args.target_speed_max)
    cfg.training_curriculum = True
    cfg.sim_to_real.enabled = stage["randomization"]
    cfg.sim.device = args.device
    cfg.depth_camera_batch_size = args.camera_batch_size
    cfg.configure_scene()
    train_values = {
        field.name: getattr(args, field.name) for field in fields(ZhangTrainCfg) if hasattr(args, field.name)
    }
    train_cfg = ZhangTrainCfg(
        **train_values,
        physics_dt=cfg.sim.dt,
        policy_dt=cfg.sim.dt * cfg.decimation,
        height_target_range=cfg.height_target_range,
    )
    train_cfg.validate()
    torch.manual_seed(args.seed)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.log_dir / "config.json",
        {
            "method": "zhang",
            "variant": args.variant,
            "algorithm": args.algorithm,
            "train": asdict(train_cfg),
            "resolved_velocity_loss_mode": train_cfg.resolved_velocity_loss_mode,
            "algorithm_label": "AHAC-derived metric-clearance and stability feedback"
            if args.algorithm == "ahac"
            else args.algorithm,
            "height_loss_mode": train_cfg.height_loss_mode,
            "static_obstacle_distance": "signed_heightfield_occupancy_exit_proxy",
            "static_projection": "float64_selected_face_refinement_float32_storage",
            "environment": {
                "num_envs": args.num_envs,
                "seed": args.seed,
                "terrain_seed": args.terrain_seed,
                "stage": args.stage,
                **stage,
                "curriculum": STAGE_CONFIG,
                "variant": args.variant,
                "drone_collision_response": args.variant != "continuous",
                "height_target_range": list(cfg.height_target_range),
                "target_speed_range": [args.target_speed_min, args.target_speed_max]
                if args.target_speed_min is not None
                else [2.25, 9.75],
                "height_target_source": "current_episode_navigation_goal_z",
                "capture_terminal_observation": cfg.capture_terminal_observation,
                "dynamic_obstacle_collision_response": False,
                "collision_terminates": args.variant != "continuous",
                "bounds_terminates": args.variant != "continuous",
                "physics_dt": cfg.sim.dt,
                "policy_dt": cfg.sim.dt * cfg.decimation,
                "timing": args.timing,
                "report_episodes": args.report_episodes or args.num_envs,
                "lidar_period_s": cfg.scene.lidar.update_period,
                "depth_period_s": cfg.scene.depth_camera.update_period,
                "depth_camera": {
                    "width": cfg.scene.depth_camera.pattern_cfg.width,
                    "height": cfg.scene.depth_camera.pattern_cfg.height,
                    "batch_size": cfg.depth_camera_batch_size,
                    "policy_shape": [12, 16],
                    "pixel_noise_std": cfg.scene.depth_camera.pixel_noise_std,
                },
                "sim_to_real": asdict(cfg.sim_to_real),
            },
            "parent_checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "fresh": args.checkpoint is None,
            "direct_stage_start": args.fresh_stage,
            "independent_evaluation": False,
        },
    )
    with launch_simulation(cfg, args):
        env = gym.make("Isaac-NavRL-Zhang-Hummingbird-Direct-v0", cfg=cfg).unwrapped
        try:
            write_json(args.log_dir / "nominal_dynamics.json", env.nominal_dynamics)
            trainer = ZhangTrainer(env, train_cfg, args.log_dir)
            trainer.process_started_at = process_started
            with graceful_training_stop(trainer):
                if args.checkpoint is not None:
                    trainer.load_stage_checkpoint(args.checkpoint)
                trainer.run()
        finally:
            env.close()


if __name__ == "__main__":
    main()
