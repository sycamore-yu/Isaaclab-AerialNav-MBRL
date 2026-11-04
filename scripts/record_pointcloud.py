"""Record one actual PointNet navigation episode with measured LiDAR hits."""

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from aerialnav.physics import create_navigation_env, parse_physics_args, resolve_navigation_config, runtime_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--terrain_seed", type=int, default=0)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parse_physics_args(parser, allow_presets=True)
    if args.output.exists():
        parser.error("Use a new output directory; existing recordings are never overwritten")
    import torch

    import aerialnav.tasks  # noqa: F401
    from aerialnav.pointnet.checkpoint import policy_contract
    from aerialnav.pointnet.inference import load_pointnet_policy
    from aerialnav.pointnet.recording import PointCloudRecording

    cfg, _ = resolve_navigation_config("Isaac-AerialNav-Depth-v0", args)
    if cfg.perception_cfg.mode != "pointnet":
        parser.error("Recording requires presets=pointnet")
    cfg.scene.num_envs = 1
    cfg.seed, cfg.terrain_seed = args.seed, args.terrain_seed
    cfg.variant = "step"
    cfg.apply_training_stage("pretrain_s0")
    cfg.training_curriculum = False
    cfg.sim.device = args.device
    cfg.configure_scene()
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True)
    with launch_simulation(cfg, args):
        env = create_navigation_env("Isaac-AerialNav-Depth-v0", cfg, args)
        recorder = None
        try:
            policy = load_pointnet_policy(env, args.checkpoint)
            # A single-vehicle recording does not need compiler warm-up or its cache.
            policy.cfg.point_encoder_backend = "torch"
            policy.model.configure_execution(policy.cfg)
            env.reset(seed=args.seed)
            recorder = PointCloudRecording(args.output, env)
            result = {"outcome": "incomplete"}
            with torch.inference_mode():
                for step in range(env.max_episode_length):
                    action = policy()
                    recorder.frame(step * env.step_dt)
                    _, _, terminated, timeout, info = env.step(action)
                    if bool((terminated | timeout)[0]):
                        code = int(info["transition"]["terminal_event_code"][0])
                        result = {
                            "outcome": {1: "success", 2: "collision", 3: "bounds", 4: "invalid", 5: "timeout"}[code],
                            "terminal_code": code,
                            "steps": step + 1,
                            "sim_time_s": (step + 1) * env.step_dt,
                        }
                        break
            result.update(
                checkpoint=str(args.checkpoint.resolve()),
                checkpoint_update=torch.load(args.checkpoint, map_location="cpu", weights_only=True)["completed_updates"],
                physics=args.physics_preset,
                seed=args.seed,
                terrain_seed=args.terrain_seed,
                static_obstacles=cfg.num_static_obstacles,
                dynamic_obstacles=cfg.num_dynamic_obstacles,
                **runtime_metadata(env),
                policy_contract=policy_contract(env),
                encoder_backend=policy.cfg.point_encoder_backend,
                point_precision=policy.cfg.point_precision,
                minimum_recorded_clearance_m=min(recorder.clearances),
            )
            recorder.close(result)
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
