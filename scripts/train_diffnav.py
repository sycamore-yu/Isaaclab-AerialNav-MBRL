"""Train one DiffNav BPTT, SHAC, or AHAC curriculum stage on Isaac Lab 3.0."""

import argparse
import hashlib
import os
import time
from dataclasses import asdict, fields
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from aerialnav.diffnav.cli import add_train_args, parse_train_args
from aerialnav.physics import create_navigation_env, parse_physics_args, runtime_metadata, select_physics
from aerialnav.source import verify_training_import
from aerialnav.training import (
    DEFAULT_STAGE,
    END_CLEARANCE,
    EPISODE_END_CONTRACT,
    MAP_SIZE,
    STAGE_CONFIG,
    STAGES,
    graceful_training_stop,
    write_json,
)


def main():
    process_started = float(os.environ.get("AERIALNAV_STAGE_STARTED_MONOTONIC", time.monotonic()))
    verify_training_import()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--height_bounds_reset", type=int, choices=(0, 1), default=1)
    parser.add_argument("--report_episodes", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=None, help="Optional smoke-run limit; default unlimited")
    parser.add_argument("--stop_after_acceptance", action="store_true", help="Finish when this stage reaches 80%")
    parser.add_argument("--stage", choices=STAGES, help="Default: fresh finetune_s1 (fS1/40)")
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--checkpoint", type=Path, help="Previous stage checkpoint from this experiment")
    checkpoints.add_argument("--resume_checkpoint", type=Path, help="Continue saved learning state in the same stage")
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
    args = parse_physics_args(parser)
    if args.stage is None:
        args.stage = DEFAULT_STAGE
        args.fresh_stage = args.fresh_stage or (args.checkpoint is None and args.resume_checkpoint is None)
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
    if args.fresh_stage and (
        args.stage == "pretrain_s0" or args.checkpoint is not None or args.resume_checkpoint is not None
    ):
        parser.error("--fresh_stage requires a later stage without --checkpoint")
    if (
        args.stage != "pretrain_s0"
        and args.checkpoint is None
        and args.resume_checkpoint is None
        and args.max_iterations is None
        and not args.fresh_stage
    ):
        parser.error("Later stages require the previous checkpoint or explicit --fresh_stage comparison")
    if args.log_dir.exists() and any(args.log_dir.iterdir()):
        parser.error("Each training stage requires a new or empty --log_dir")
    if args.resume_checkpoint is not None and args.lr is None:
        import torch

        args.lr = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)["train_cfg"]["lr"]
    args.lr = (
        args.lr
        if args.lr is not None
        else (5e-5 if args.stage.startswith("finetune") and not args.fresh_stage else 0.001)
    )

    import torch

    import aerialnav.tasks  # noqa: F401
    from aerialnav.diffnav.config import DiffNavTrainCfg
    from aerialnav.tasks.navigation.config.hummingbird.env_cfg import DiffNavEnvCfg
    from aerialnav.tasks.navigation.depth_history import OBSERVATION_CONTRACT

    stage = STAGE_CONFIG[args.stage].copy()
    if args.num_dynamic_obstacles is not None:
        stage["dynamic_obstacles"] = args.num_dynamic_obstacles
    cfg = DiffNavEnvCfg()
    cfg.variant = args.variant
    cfg.height_bounds_reset = bool(args.height_bounds_reset)
    cfg.geometry_backend = args.geometry_backend
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
    select_physics(cfg, args.physics_preset)
    cfg.configure_scene()
    train_values = {
        field.name: getattr(args, field.name) for field in fields(DiffNavTrainCfg) if hasattr(args, field.name)
    }
    train_cfg = DiffNavTrainCfg(
        **train_values,
        physics=args.physics_preset,
        physics_dt=cfg.sim.dt,
        policy_dt=cfg.sim.dt * cfg.decimation,
        height_target_range=cfg.height_target_range,
    )
    train_cfg.validate()
    torch.manual_seed(args.seed)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "method": "diffnav",
        "physics": args.physics_preset,
        "frontend": args.frontend,
        "variant": args.variant,
        "algorithm": args.algorithm,
        "train": asdict(train_cfg),
        "resolved_velocity_loss_mode": train_cfg.resolved_velocity_loss_mode,
        "algorithm_label": "AHAC-derived metric-clearance and stability feedback"
        if args.algorithm == "ahac"
        else args.algorithm,
        "height_loss_mode": train_cfg.height_loss_mode,
        "static_obstacle_distance": "curobo_v2_known_occupancy_pba_esdf"
        if cfg.geometry_backend == "curobo_v2"
        else "signed_heightfield_occupancy_exit_proxy",
        "static_projection": "fp16_esdf_trilinear_fp32_gradient"
        if cfg.geometry_backend == "curobo_v2"
        else "float64_selected_face_refinement_float32_storage",
        "environment": {
            "physics": args.physics_preset,
            "frontend": args.frontend,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "terrain_seed": args.terrain_seed,
            "stage": args.stage,
            **stage,
            "curriculum": STAGE_CONFIG,
            "variant": args.variant,
            "drone_collision_response": args.variant != "continuous",
            "height_target_range": list(cfg.height_target_range),
            "map_size_xy_m": list(MAP_SIZE),
            "end_clearance_m": END_CLEARANCE,
            "route_sampling": "XY_distance=18*target_speed; heading=+/-X +/-8deg; feasible_start_uniform",
            "episode_length_s": cfg.episode_length_s,
            "dynamic_speed_range_mps": list(cfg.dynamic_speed_range),
            "target_speed_range": list(cfg.target_speed_range or (1.0, 6.0)),
            "height_target_source": "current_episode_navigation_goal_z",
            "capture_terminal_observation": cfg.capture_terminal_observation,
            "dynamic_obstacle_collision_response": False,
            "collision_terminates": args.variant != "continuous",
            "height_bounds_reset": cfg.height_bounds_reset,
            "geometry_backend": cfg.geometry_backend,
            "perception": "depth",
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
            "observation_contract": OBSERVATION_CONTRACT,
            "episode_end_contract": EPISODE_END_CONTRACT,
            "wait_before_truncate_s": cfg.wait_before_truncate,
        },
        "parent_checkpoint": str((args.checkpoint or args.resume_checkpoint).resolve())
        if args.checkpoint or args.resume_checkpoint
        else None,
        "fresh": args.checkpoint is None and args.resume_checkpoint is None,
        "direct_stage_start": args.fresh_stage,
        "independent_evaluation": False,
    }
    write_json(args.log_dir / "config.json", run_config)
    with launch_simulation(cfg, args):
        env = create_navigation_env("Isaac-AerialNav-Depth-v0", cfg, args)
        try:
            # TensorBoard's native dependencies must load after the PhysX Kit app.
            from aerialnav.diffnav.runner import DiffNavTrainer

            run_config.update(runtime_metadata(env))
            run_config["environment"].update(runtime_metadata(env))
            write_json(args.log_dir / "config.json", run_config)
            write_json(args.log_dir / "nominal_dynamics.json", env.nominal_dynamics)
            trainer = DiffNavTrainer(env, train_cfg, args.log_dir)
            if cfg.geometry_backend == "curobo_v2":
                write_json(args.log_dir / "geometry_backend.json", env.terrain_distance.metadata())
            initialize_rollout = trainer._initialize_rollout

            def initial_rollout():
                values = initialize_rollout()
                receipt = args.log_dir / "initial_rollout.json"
                if not receipt.exists():
                    route = env.target_pos[:, :2] - env.root_state()[:, :2]
                    speed = env.diffnav_parameters["max_speed"].flatten()
                    distance = route.norm(dim=-1)
                    angle = torch.atan2(route[:, 1].abs(), route[:, 0].abs()) * (180 / torch.pi)
                    world = {
                        "root_state": env.root_state(),
                        "target": env.target_pos,
                        "obstacle_position": env.motion.position,
                        "obstacle_velocity": env.motion.velocity,
                        **{f"parameter/{key}": value for key, value in env.diffnav_parameters.items()},
                    }
                    hashes = {
                        key: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
                        for key, value in world.items()
                    }
                    write_json(
                        receipt,
                        {
                            "world_sha256": hashlib.sha256(str(sorted(hashes.items())).encode()).hexdigest(),
                            "fields": hashes,
                            "seed": args.seed,
                            "terrain_seed": args.terrain_seed,
                            "route_distance_min_m": float(distance.min()),
                            "route_distance_max_m": float(distance.max()),
                            "route_duration_max_error_s": float((distance / speed - 18).abs().max()),
                            "route_heading_max_abs_deg": float(angle.max()),
                            "route_endpoint_min_clearance_m": float(
                                env.nearest_obstacle_signed_distance(
                                    torch.cat((env.root_state()[:, :3], env.target_pos))
                                ).min()
                            ),
                            "target_speed_min_mps": float(speed.min()),
                            "target_speed_max_mps": float(speed.max()),
                        },
                    )
                return values

            trainer._initialize_rollout = initial_rollout
            trainer.process_started_at = process_started
            with graceful_training_stop(trainer):
                if args.checkpoint is not None:
                    trainer.load_stage_checkpoint(args.checkpoint)
                elif args.resume_checkpoint is not None:
                    trainer.load_stage_checkpoint(args.resume_checkpoint, resume=True)
                initial_hash = hashlib.sha256()
                for name, parameter in trainer.model.state_dict().items():
                    initial_hash.update(name.encode())
                    initial_hash.update(parameter.detach().cpu().numpy().tobytes())
                write_json(
                    args.log_dir / "initialization.json",
                    {
                        "model_sha256": initial_hash.hexdigest(),
                        "policy_class": type(trainer.model).__name__,
                        "perception": train_cfg.perception,
                        "fresh": args.checkpoint is None and args.resume_checkpoint is None,
                        "seed": args.seed,
                    },
                )
                trainer.run()
        finally:
            env.close()


if __name__ == "__main__":
    main()
