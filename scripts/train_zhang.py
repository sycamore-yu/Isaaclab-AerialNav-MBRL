"""Train one stage of the Zhang BPTT curriculum on Isaac Lab 3.0 PhysX."""

import argparse
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def parse_gradient_diagnostic_updates(raw: str) -> tuple[int, ...]:
    """Parse an opt-in list of unique positive global completed-update numbers."""
    if raw == "":
        return ()
    parts = raw.split(",")
    if any(not part.isascii() or not part.isdecimal() for part in parts):
        raise ValueError("gradient_diagnostic_updates requires comma-separated positive integers without empty entries")
    updates = tuple(int(part) for part in parts)
    if any(update < 1 for update in updates) or len(set(updates)) != len(updates):
        raise ValueError("gradient_diagnostic_updates must contain unique positive integers")
    return updates


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=None, help="Optional smoke-run limit; default unlimited")
    parser.add_argument("--stop_after_acceptance", action="store_true", help="Finish when this stage reaches 80%")
    parser.add_argument("--stage", choices=("pretrain_s1", "pretrain_s2", "finetune_s1", "finetune_s2"),
                        default="pretrain_s1")
    parser.add_argument("--checkpoint", type=Path, help="Previous stage checkpoint from this experiment")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--terrain_seed", type=int, default=0)
    parser.add_argument("--variant", choices=("step", "continuous", "dcol", "truncated_window"), default="step")
    parser.add_argument("--timesteps", type=int, default=150)
    parser.add_argument("--velocity_window_s", type=float, default=2.0)
    parser.add_argument("--camera_batch_size", type=int, default=16,
                        help="Depth raycast environment batch size; does not change image resolution")
    parser.add_argument("--lr", type=float, default=None, help="Default: 0.001 pretraining, 0.00005 fine-tuning")
    parser.add_argument("--cosine_period_updates", type=int, default=400)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--grad_decay", type=float, default=0.4)
    parser.add_argument("--coef_obj_avoidance", type=float, default=3.0)
    parser.add_argument("--coef_collide", type=float, default=7.5)
    parser.add_argument("--coef_d_acc", type=float, default=0.01)
    parser.add_argument("--coef_d_jerk", type=float, default=0.001)
    parser.add_argument("--coef_height_boundary", type=float, default=1.0)
    parser.add_argument("--boundary_margin", type=float, default=0.5)
    parser.add_argument("--boundary_scale", type=float, default=0.5)
    parser.add_argument("--gradient_diagnostic_updates", type=parse_gradient_diagnostic_updates, default=(),
                        help="Optional unique global completed-update numbers, e.g. 20,22,24,26; empty disables")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("num_envs must be positive")
    if args.camera_batch_size < 1:
        parser.error("camera_batch_size must be positive")
    if args.stage == "pretrain_s1" and args.checkpoint is not None:
        parser.error("Pretraining S1 must start from random initialization")
    if args.stage != "pretrain_s1" and args.checkpoint is None and args.max_iterations is None:
        parser.error("Later stages require the previous stage checkpoint; use a finite limit only for smoke tests")
    if args.log_dir.exists() and any(args.log_dir.iterdir()):
        parser.error("Each training stage requires a new or empty --log_dir")
    args.lr = args.lr if args.lr is not None else (5e-5 if args.stage.startswith("finetune") else 0.001)

    import gymnasium as gym
    import torch

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import ZhangNavRLEnvCfg
    from isaaclab_navrl.tasks.navigation.curriculum import STAGE_CONFIGS
    from isaaclab_navrl.training import write_json
    from isaaclab_navrl.zhang.config import ZhangTrainCfg
    from isaaclab_navrl.zhang.runner import ZhangTrainer

    stage = STAGE_CONFIGS[args.stage]
    cfg = ZhangNavRLEnvCfg()
    cfg.variant = args.variant
    cfg.scene.num_envs = args.num_envs
    cfg.seed, cfg.terrain_seed = args.seed, args.terrain_seed
    cfg.num_static_obstacles = stage["static_obstacles"]
    cfg.num_dynamic_obstacles = stage["dynamic_obstacles"]
    cfg.training_stage = args.stage
    cfg.training_curriculum = True
    cfg.sim_to_real.enabled = stage["randomization"]
    cfg.sim.device = args.device
    cfg.depth_camera_batch_size = args.camera_batch_size
    cfg.configure_scene()
    fields = ("variant", "timesteps", "max_iterations", "stop_after_acceptance", "velocity_window_s", "lr", "max_grad_norm",
              "grad_decay", "coef_obj_avoidance", "coef_collide", "coef_d_acc", "coef_d_jerk",
              "coef_height_boundary", "boundary_margin", "boundary_scale", "cosine_period_updates", "seed",
              "gradient_diagnostic_updates")
    train_cfg = ZhangTrainCfg(**{name: getattr(args, name) for name in fields})
    train_cfg.validate()
    torch.manual_seed(args.seed)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.log_dir / "config.json", {
        "method": "zhang", "variant": args.variant, "train": asdict(train_cfg),
        "environment": {"num_envs": args.num_envs, "seed": args.seed, "terrain_seed": args.terrain_seed,
                        "stage": args.stage, **stage, "curriculum": STAGE_CONFIGS,
                        "variant": args.variant, "drone_collision_response": args.variant != "continuous",
                        "dynamic_obstacle_collision_response": False,
                        "collision_terminates": args.variant != "continuous",
                        "physics_dt": cfg.sim.dt, "policy_dt": cfg.sim.dt * cfg.decimation,
                        "depth_camera": {"width": cfg.scene.depth_camera.pattern_cfg.width,
                                         "height": cfg.scene.depth_camera.pattern_cfg.height,
                                         "batch_size": cfg.depth_camera_batch_size,
                                         "policy_shape": [12, 16],
                                         "pixel_noise_std": cfg.scene.depth_camera.pixel_noise_std},
                        "sim_to_real": asdict(cfg.sim_to_real)},
        "parent_checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "fresh": args.checkpoint is None, "independent_evaluation": False,
    })
    with launch_simulation(cfg, args):
        env = gym.make("Isaac-NavRL-Zhang-Hummingbird-Direct-v0", cfg=cfg).unwrapped
        try:
            write_json(args.log_dir / "nominal_dynamics.json", env.nominal_dynamics)
            trainer = ZhangTrainer(env, train_cfg, args.log_dir)
            if args.checkpoint is not None:
                trainer.load_stage_checkpoint(args.checkpoint)
            trainer.run()
        finally:
            env.close()


if __name__ == "__main__":
    main()
