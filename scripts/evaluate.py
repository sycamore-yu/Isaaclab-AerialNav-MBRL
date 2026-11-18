"""Evaluate completed episodes on one independently launched NavRL difficulty."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_navrl.physics import (
    create_navigation_env,
    parse_runtime_args,
    resolve_navigation_config,
    runtime_metadata,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--method", choices=("navrl", "zhang"), required=True)
    parser.add_argument("--geometry_backend", choices=("native", "curobo_v2"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--terrain_seed", type=int, default=10000)
    parser.add_argument("--static_obstacles", type=int, default=None)
    parser.add_argument("--dynamic_obstacles", type=int, default=None)
    parser.add_argument("--stage", choices=("pretrain_s0", "pretrain_s1", "finetune_s1"))
    parser.add_argument("--wait_before_truncate", type=float, default=None)
    parser.add_argument("--noise_scale", type=float, default=0.0, help="Multiply sim_to_real odometry/depth noise during eval")
    parser.add_argument("--trace_output", type=Path, default=None, help="Optional JSON file saving per-step episode traces")
    parser.add_argument("--trace_episodes", type=int, default=8)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parse_runtime_args(parser, allow_presets=True)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.output.exists():
        parser.error(f"evaluation output already exists: {args.output}")

    import torch

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.evaluation import EpisodeEvaluation
    from isaaclab_navrl.training import EPISODE_END_CONTRACT

    task = "Isaac-NavRL-Zhang-Hummingbird-Direct-v0" if args.method == "zhang" else "Isaac-NavRL-Hummingbird-Direct-v0"
    cfg, _ = resolve_navigation_config(task, args)
    pointnet = args.method == "zhang" and cfg.perception_cfg.mode == "pointnet"
    if pointnet:
        cfg.variant = "step"
    if args.geometry_backend is not None:
        if args.method != "zhang":
            parser.error("--geometry_backend requires --method zhang")
        cfg.geometry_backend = args.geometry_backend
    if args.method == "zhang" and not pointnet:
        # Configure event acquisition before the sensor and task are created.
        saved_train = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["train_cfg"]
        cfg.depth_event_channel = bool(saved_train.get("depth_event_channel", False))
        cfg.depth_event_kinematic_field = bool(
            saved_train.get("depth_event_kinematic_field", False) or cfg.depth_event_channel
        )
        cfg.depth_event_history_frames = int(saved_train.get("depth_event_history_frames", 1))
        cfg.depth_event_pose_source = saved_train.get("depth_event_pose_source", "estimated")
        cfg.active_yaw_alignment = float(saved_train.get("active_yaw_alignment", 0.0))
    cfg.training_curriculum = False
    cfg.scene.num_envs = args.episodes
    cfg.seed, cfg.terrain_seed = args.seed, args.terrain_seed
    cfg.evaluation = args.stage is None and args.noise_scale <= 0.0
    if args.stage:
        cfg.apply_training_stage(args.stage)
    if args.noise_scale > 0.0:
        cfg.evaluation = False
        cfg.sim_to_real.enabled = True
        cfg.sim_to_real.observation_noise = True
        cfg.sim_to_real.input_delay = False
        cfg.sim_to_real.action_delay = False
        cfg.sim_to_real.control_response = False
        cfg.sim_to_real.action_difference = False
        cfg.sim_to_real.mass = False
        cfg.sim_to_real.inertia = False
        cfg.sim_to_real.thrust = False
        cfg.sim_to_real.position_std *= args.noise_scale
        cfg.sim_to_real.velocity_std *= args.noise_scale
        cfg.sim_to_real.attitude_std_deg *= args.noise_scale
        cfg.sim_to_real.depth_std *= args.noise_scale
    cfg.num_static_obstacles = (
        args.static_obstacles if args.static_obstacles is not None else (cfg.num_static_obstacles if args.stage else 350)
    )
    cfg.num_dynamic_obstacles = (
        args.dynamic_obstacles if args.dynamic_obstacles is not None else (cfg.num_dynamic_obstacles if args.stage else 100)
    )
    if args.wait_before_truncate is not None:
        cfg.wait_before_truncate = args.wait_before_truncate
    cfg.sim.device = args.device
    cfg.configure_scene()
    task = "Isaac-NavRL-Zhang-Hummingbird-Direct-v0" if args.method == "zhang" else "Isaac-NavRL-Hummingbird-Direct-v0"
    with launch_simulation(cfg, args):
        env = create_navigation_env(task, cfg, args)
        try:
            if args.method == "navrl":
                from isaaclab_navrl.agents.inference import load_navrl_policy

                policy = load_navrl_policy(env, args.checkpoint)
            else:
                from isaaclab_navrl.zhang.inference import load_zhang_policy

                if pointnet:
                    from isaaclab_navrl.pointnet.inference import load_pointnet_policy

                    policy = load_pointnet_policy(env, args.checkpoint)
                else:
                    policy = load_zhang_policy(env, args.checkpoint)
            observation, _ = env.reset()
            evaluation = EpisodeEvaluation(
                env.root_state()[:, :3].cpu().tolist(), env.target_pos.cpu().tolist(), args.seed
            )
            # Classification uses simulator ground truth only for evaluation;
            # neither the detector nor the policy receives obstacle labels.
            from isaaclab_navrl.tasks.navigation.geometry import dynamic_signed_distance
            from isaaclab_navrl.zhang.aux import forward_closing_risk
            from isaaclab_navrl.zhang.model import quaternion_matrix

            trace_count = min(args.trace_episodes, args.episodes) if args.trace_output is not None else 0
            traces = [[] for _ in range(trace_count)]
            trace_done = [False] * trace_count
            start = time.perf_counter()
            with torch.inference_mode():
                for step_idx in range(env.max_episode_length):
                    root = env.root_state()
                    pre_pos = root[:, :3].clone()
                    pre_vel = root[:, 7:10].clone()
                    pre_quat = root[:, 3:7].clone()
                    if trace_count:
                        rot_m = quaternion_matrix(pre_quat)
                        fwd = rot_m[:, :, 0].clone()
                        fwd[:, 2] = 0.0
                        fwd = torch.nn.functional.normalize(fwd, dim=-1)
                        up = torch.zeros_like(fwd)
                        up[:, 2] = 1.0
                        hd = torch.stack((fwd, torch.cross(up, fwd, dim=-1), up), dim=-1)
                        r_close = forward_closing_risk(
                            pre_pos[:trace_count],
                            pre_vel[:trace_count],
                            hd[:trace_count],
                            env.motion.position,
                            env.motion.velocity,
                            env.motion.size,
                        )
                    action = policy(observation)
                    observation, _, _, _, info = env.step(action)
                    transition = info["transition"]
                    if trace_count and len(env.motion.position):
                        dyn_pos_w = env.motion.position
                        dyn_vel_w = env.motion.velocity
                        dyn_rad = 0.5 * env.motion.size[:, :2].amax(dim=-1)
                        for ei in range(trace_count):
                            if trace_done[ei]:
                                continue
                            rel_w = dyn_pos_w - pre_pos[ei : ei + 1]
                            rel_e = rel_w @ hd[ei]
                            in_fov = (rel_e[:, 0] > 0.0) & (torch.atan2(rel_e[:, 1], rel_e[:, 0]).abs() <= 0.5236)
                            norm_r = rel_w.norm(dim=-1).clamp_min(1e-3)
                            v_close = (-(rel_w * (dyn_vel_w - pre_vel[ei : ei + 1])).sum(dim=-1) / norm_r)
                            gap = (rel_e[:, :2].norm(dim=-1) - dyn_rad).clamp_min(0.05)
                            ttc = torch.where(in_fov & (v_close > 0.1), gap / v_close.clamp_min(1e-3), torch.full_like(gap, 99.0))
                            near_mask = rel_w[:, :2].norm(dim=-1) < 8.0
                            traces[ei].append(
                                {
                                    "t": float(step_idx * env.step_dt),
                                    "pos": pre_pos[ei].cpu().tolist(),
                                    "vel": pre_vel[ei].cpu().tolist(),
                                    "speed": float(pre_vel[ei].norm()),
                                    "r_closing": float(r_close[ei]),
                                    "min_fov_ttc_s": float(ttc.min()),
                                    "nearby_obs_xy": dyn_pos_w[near_mask, :2].cpu().tolist(),
                                }
                            )
                    evaluation.update(transition)
                    if trace_count:
                        for ei in range(trace_count):
                            if evaluation.rows[ei] is not None or bool(
                                transition["terminated"][ei] or transition["timeout"][ei]
                            ):
                                trace_done[ei] = True
                    collisions = transition["collision"] & transition["terminated"]
                    ids = collisions.nonzero(as_tuple=False).flatten()
                    if len(ids):
                        positions = transition["position"][ids]
                        velocities = transition.get("collision_velocity", pre_vel)[ids]
                        rot_c = quaternion_matrix(pre_quat[ids])
                        fwd_c = rot_c[:, :, 0].clone()
                        fwd_c[:, 2] = 0.0
                        fwd_c = torch.nn.functional.normalize(fwd_c, dim=-1)
                        up_c = torch.zeros_like(fwd_c)
                        up_c[:, 2] = 1.0
                        hd_c = torch.stack((fwd_c, torch.cross(up_c, fwd_c, dim=-1), up_c), dim=-1)

                        moving = dynamic_signed_distance(positions, env.motion.position, env.motion.size)
                        fixed = torch.minimum(
                            env.terrain_distance.signed_distance(positions), positions[:, 2] - 0.01
                        )
                        classification = ((moving <= 0.3) & (moving <= fixed + 1e-4)).cpu().tolist()
                        dyn_pos = env.motion.position
                        dyn_vel = env.motion.velocity
                        for k_idx, (i, dynamic) in enumerate(zip(ids.cpu().tolist(), classification, strict=True)):
                            row = evaluation.rows[i]
                            if row is None or row["outcome"] != "collision" or "collision_source" in row:
                                continue
                            row["collision_source"] = "dynamic" if dynamic else "static"
                            if not dynamic or len(dyn_pos) == 0:
                                row["collision_mode"] = "static"
                                continue
                            # Identify closest dynamic obstacle and decompose ego azimuth & relative velocity.
                            d_vec = dyn_pos - positions[k_idx : k_idx + 1]
                            nearest_idx = int((d_vec[:, :2].norm(dim=-1) - 0.5 * env.motion.size[:, :2].amax(dim=-1)).argmin())
                            rel_p_ego = torch.einsum("i,ij->j", dyn_pos[nearest_idx] - positions[k_idx], hd_c[k_idx])
                            rel_v_ego = torch.einsum("i,ij->j", dyn_vel[nearest_idx] - velocities[k_idx], hd_c[k_idx])
                            az_deg = float(torch.rad2deg(torch.atan2(rel_p_ego[1], rel_p_ego[0])))
                            row["collision_azimuth_deg"] = az_deg
                            row["collision_rel_speed_mps"] = float(rel_v_ego.norm())
                            if abs(az_deg) > 110.0:
                                row["collision_mode"] = "rear_tail"
                            elif abs(az_deg) > 45.0 or float(rel_v_ego[1].abs()) > float(rel_v_ego[0].abs()):
                                row["collision_mode"] = "cross_traffic"
                            else:
                                row["collision_mode"] = "head_on"
                    if evaluation.complete:
                        break
            if args.trace_output is not None:
                args.trace_output.parent.mkdir(parents=True, exist_ok=True)
                args.trace_output.write_text(json.dumps({"episodes": traces, "outcomes": evaluation.rows[:trace_count]}, indent=2) + "\n")
            mode_counts = {
                m: sum(row.get("collision_mode") == m for row in evaluation.rows)
                for m in ("head_on", "cross_traffic", "rear_tail", "static")
            }
            result = {
                "method": "pointnet" if pointnet else args.method,
                "preset": "pointnet" if pointnet else "default",
                "variant": getattr(cfg, "variant", None),
                **runtime_metadata(env),
                **({"recurrent_model": policy.cfg.recurrent_model} if args.method == "zhang" else {}),
                "checkpoint": str(args.checkpoint.resolve()),
                "seed": args.seed,
                "terrain_seed": args.terrain_seed,
                "noise_scale": args.noise_scale,
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
                    if args.method == "zhang"
                    else {}
                ),
                "elapsed_s": time.perf_counter() - start,
                "summary": evaluation.summary()
                | {
                    "dynamic_collision_count": sum(
                        row.get("collision_source") == "dynamic" for row in evaluation.rows
                    ),
                    "static_collision_count": sum(
                        row.get("collision_source") == "static" for row in evaluation.rows
                    ),
                    "collision_modes": mode_counts,
                },
                "episodes": evaluation.rows,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print("EVALUATION_RESULT " + json.dumps(result["summary"], allow_nan=False), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
