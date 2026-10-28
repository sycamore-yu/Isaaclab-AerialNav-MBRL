"""Fresh Zhang BPTT training on the official Isaac Lab 3.0 PhysX task."""

import argparse
import json
from dataclasses import asdict
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
    parser.add_argument("--timesteps", type=int, default=150)
    parser.add_argument("--velocity_window_s", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--grad_decay", type=float, default=0.4)
    parser.add_argument("--coef_obj_avoidance", type=float, default=3.0)
    parser.add_argument("--coef_collide", type=float, default=7.5)
    parser.add_argument("--coef_d_acc", type=float, default=0.01)
    parser.add_argument("--coef_d_jerk", type=float, default=0.001)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()
    if args.log_dir.exists() and any(args.log_dir.iterdir()):
        parser.error("Fresh training requires a new or empty --log_dir; resume/pretraining is not supported")

    import gymnasium as gym
    import torch

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import ZhangNavRLEnvCfg
    from isaaclab_navrl.zhang.config import ZhangTrainCfg
    from isaaclab_navrl.zhang.runner import ZhangTrainer

    cfg = ZhangNavRLEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed, cfg.terrain_seed = args.seed, args.terrain_seed
    cfg.num_static_obstacles, cfg.num_dynamic_obstacles = args.static_obstacles, args.dynamic_obstacles
    cfg.sim.device = args.device
    cfg.configure_scene()
    fields = ("timesteps", "max_iterations", "velocity_window_s", "lr", "max_grad_norm", "grad_decay",
              "coef_obj_avoidance", "coef_collide", "coef_d_acc", "coef_d_jerk", "seed")
    train_cfg = ZhangTrainCfg(**{name: getattr(args, name) for name in fields})
    train_cfg.validate()
    torch.manual_seed(args.seed)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / "config.json").write_text(json.dumps({"train": asdict(train_cfg),
        "environment": {"num_envs": args.num_envs, "seed": args.seed, "terrain_seed": args.terrain_seed,
                        "static_obstacles": args.static_obstacles, "dynamic_obstacles": args.dynamic_obstacles},
        "fresh": True}, indent=2) + "\n")
    with launch_simulation(cfg, args):
        env = gym.make("Isaac-NavRL-Zhang-Hummingbird-Direct-v0", cfg=cfg).unwrapped
        try:
            ZhangTrainer(env, train_cfg, args.log_dir).run()
        finally:
            env.close()


if __name__ == "__main__":
    main()
