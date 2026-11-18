"""Record a Rerun (.rrd) replay of one evaluation episode with 3D world, dynamic traffic, event fields, and yielding metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import warp as wp
from isaaclab.app import add_launcher_args, launch_simulation
from pxr import Usd, UsdGeom

from isaaclab_navrl.physics import create_navigation_env, parse_runtime_args, resolve_navigation_config
from isaaclab_navrl.pointnet.recording import terrain_mesh_parts


def v4_recording_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Tabs(
                rrb.Spatial3DView(
                    origin="world/follow",
                    contents=["/world/**", "- /world/trajectory"],
                    name="Flight · follow · dynamic traffic",
                    line_grid=False,
                    eye_controls=rrb.EyeControls3D(
                        position=(-2.5, -3.5, 2.2),
                        look_target=(2.0, 0.0, 0.0),
                        eye_up=(0, 0, 1),
                    ),
                ),
                rrb.Spatial3DView(
                    origin="world",
                    name="Overview · full corridor",
                    eye_controls=rrb.EyeControls3D(position=(0.0, -75.0, 65.0), look_target=(0.0, 0.0, 1.0)),
                ),
                active_tab=0,
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="camera/depth", name="Depth (m)"),
                    rrb.Spatial2DView(origin="camera/kinematic_events", name="Kinematic Event Field (R:Loom G:App B:Dis)"),
                ),
                rrb.TimeSeriesView(origin="metrics/speed_mps", name="Drone Speed ||v|| (m/s)"),
                rrb.TimeSeriesView(origin="metrics/risk_and_ttc", name="Closing Risk R & Min FOV TTC (s)"),
                rrb.TextDocumentView(origin="status", name="Episode status"),
                row_shares=[2, 1, 1, 1],
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(timeline="sim_time", expanded=True),
        collapse_panels=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env_idx", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11000)
    parser.add_argument("--terrain_seed", type=int, default=11000)
    parser.add_argument("--static_obstacles", type=int, default=240)
    parser.add_argument("--dynamic_obstacles", type=int, default=120)
    parser.add_argument("--stage", choices=("pretrain_s0", "pretrain_s1", "finetune_s1"), default="pretrain_s1")
    parser.add_argument("--geometry_backend", choices=("native", "curobo_v2"), default="curobo_v2")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parse_runtime_args(parser, allow_presets=True)

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not 0 <= args.env_idx < args.episodes:
        parser.error("--env_idx must satisfy 0 <= env_idx < episodes")

    import torch

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.evaluation import EpisodeEvaluation
    from isaaclab_navrl.zhang.aux import forward_closing_risk
    from isaaclab_navrl.zhang.inference import load_zhang_policy
    from isaaclab_navrl.zhang.model import quaternion_matrix

    task = "Isaac-NavRL-Zhang-Hummingbird-Direct-v0"
    cfg, _ = resolve_navigation_config(task, args)
    cfg.geometry_backend = args.geometry_backend
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
    cfg.evaluation = True
    cfg.apply_training_stage(args.stage)
    cfg.num_static_obstacles = args.static_obstacles
    cfg.num_dynamic_obstacles = args.dynamic_obstacles
    cfg.sim.device = args.device
    cfg.configure_scene()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, args):
        env = create_navigation_env(task, cfg, args)
        try:
            policy = load_zhang_policy(env, args.checkpoint)
            observation, _ = env.reset()
            evaluation = EpisodeEvaluation(
                env.root_state()[:, :3].cpu().tolist(), env.target_pos.cpu().tolist(), args.seed
            )

            ei = args.env_idx
            stream = rr.RecordingStream(f"DiffNav v4.1 · {args.stage} · Ep {ei}")
            stream.save(args.output, default_blueprint=v4_recording_blueprint())
            stream.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

            # Static terrain & obstacle meshes.
            lidar = env.scene["lidar"]
            mesh = lidar.meshes[(lidar.cfg.mesh_prim_paths[0], env.device)]
            vertices = wp.to_torch(mesh.points).cpu().numpy()
            faces = wp.to_torch(mesh.indices).cpu().numpy().reshape(-1, 3)
            for name, points, triangles, color in terrain_mesh_parts(vertices, faces):
                stream.log(
                    "world/terrain/" + name,
                    rr.Mesh3D(vertex_positions=points, triangle_indices=triangles, albedo_factor=color),
                    static=True,
                )
            for x, label, color in [(60, "map_bounds", [150, 150, 150]), (54, "safe_end_boundary", [235, 200, 65])]:
                stream.log(
                    "world/" + label,
                    rr.LineStrips3D(
                        [[[-x, -20, 0.03], [x, -20, 0.03], [x, 20, 0.03], [-x, 20, 0.03], [-x, -20, 0.03]]],
                        colors=[color],
                        radii=0.03,
                    ),
                    static=True,
                )
            goal_np = env.target_pos[ei : ei + 1].cpu().numpy()
            stream.log("world/goal", rr.Points3D(goal_np, radii=0.6, colors=[250, 220, 60], labels=["Goal"]), static=True)

            # Vehicle USD mesh.
            source = Path(__file__).resolve().parents[1] / "src/isaaclab_navrl/assets/hummingbird.usd"
            usd_stage = Usd.Stage.Open(str(source))
            cache = UsdGeom.XformCache()
            for prim in usd_stage.Traverse():
                if not prim.IsA(UsdGeom.Mesh) or prim.GetName() != "visuals":
                    continue
                visual = UsdGeom.Mesh(prim)
                v_faces = np.asarray(visual.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
                v_points = np.asarray(visual.GetPointsAttr().Get())
                v_verts = np.column_stack((v_points, np.ones(len(v_points)))) @ np.asarray(
                    cache.GetLocalToWorldTransform(prim)
                )
                path = str(prim.GetPath())
                if "/base_link/" in path:
                    color = [65, 185, 245]
                else:
                    rotor = int(path.split("/rotor_")[1].split("/")[0])
                    color = [250, 98, 72] if rotor == 0 else [245, 220, 100] if rotor == 2 else [232, 238, 242]
                stream.log(
                    "world/vehicle/" + str(prim.GetPath()).strip("/"),
                    rr.Mesh3D(vertex_positions=v_verts[:, :3], triangle_indices=v_faces, albedo_factor=color),
                    static=True,
                )

            positions = []
            peak_risk = -1.0
            peak_risk_t = 0.0
            min_speed = 999.0
            max_speed = 0.0
            with torch.inference_mode():
                for step_idx in range(env.max_episode_length):
                    root = env.root_state()
                    pre_pos = root[ei : ei + 1, :3].clone()
                    pre_vel = root[ei : ei + 1, 7:10].clone()
                    pre_quat = root[ei : ei + 1, 3:7].clone()
                    rot_m = quaternion_matrix(pre_quat)
                    fwd = rot_m[:, :, 0].clone()
                    fwd[:, 2] = 0.0
                    fwd = torch.nn.functional.normalize(fwd, dim=-1)
                    up = torch.zeros_like(fwd)
                    up[:, 2] = 1.0
                    hd = torch.stack((fwd, torch.cross(up, fwd, dim=-1), up), dim=-1)

                    r_close = float(
                        forward_closing_risk(
                            pre_pos, pre_vel, hd, env.motion.position, env.motion.velocity, env.motion.size
                        )[0]
                    )
                    dyn_pos = env.motion.position
                    dyn_vel = env.motion.velocity
                    dyn_size = env.motion.size
                    dyn_rad = 0.5 * dyn_size[:, :2].amax(dim=-1)
                    rel_w = dyn_pos - pre_pos
                    rel_e = rel_w @ hd[0]
                    in_fov = (rel_e[:, 0] > 0.0) & (torch.atan2(rel_e[:, 1], rel_e[:, 0]).abs() <= 0.5236)
                    norm_r = rel_w.norm(dim=-1).clamp_min(1e-3)
                    v_close = -(rel_w * (dyn_vel - pre_vel)).sum(dim=-1) / norm_r
                    gap = (rel_e[:, :2].norm(dim=-1) - dyn_rad).clamp_min(0.05)
                    ttc = torch.where(
                        in_fov & (v_close > 0.1), gap / v_close.clamp_min(1e-3), torch.full_like(gap, 10.0)
                    )
                    min_ttc = float(ttc.min().clamp(0.0, 10.0))

                    # Fetch observation & kinematic event channels before step.
                    zobs = env.get_zhang_observation()
                    depth_raw = zobs.privileged["measured_depth"][ei].cpu().numpy()
                    kin_ws = getattr(env.depth_event_detector, "_kinematic_workspace", None)

                    pos_np = pre_pos[0].cpu().numpy()
                    quat_np = pre_quat[0].cpu().numpy()
                    speed = float(pre_vel[0].norm())
                    t_s = float(step_idx * env.step_dt)
                    positions.append(pos_np.copy())
                    if r_close > peak_risk:
                        peak_risk = r_close
                        peak_risk_t = t_s
                    min_speed = min(min_speed, speed)
                    max_speed = max(max_speed, speed)

                    stream.set_time("sim_time", duration=t_s)
                    stream.log("world/follow", rr.Transform3D(translation=pos_np))
                    stream.log(
                        "world/vehicle",
                        rr.Transform3D(translation=pos_np, rotation=rr.Quaternion(xyzw=quat_np)),
                    )
                    stream.log(
                        "world/trajectory",
                        rr.LineStrips3D([np.asarray(positions)], colors=[255, 184, 90], radii=0.06),
                    )
                    if len(dyn_pos):
                        d_pos_np = dyn_pos.cpu().numpy()
                        d_vel_np = dyn_vel.cpu().numpy()
                        d_half_np = (0.5 * dyn_size).cpu().numpy()
                        near = np.linalg.norm(d_pos_np[:, :2] - pos_np[:2], axis=-1) < 12.0
                        colors = np.where(near[:, None], [235, 75, 65], [160, 95, 95]).astype(np.uint8)
                        stream.log(
                            "world/dynamic_obstacles",
                            rr.Boxes3D(centers=d_pos_np, half_sizes=d_half_np, colors=colors),
                        )
                        stream.log(
                            "world/dynamic_velocities",
                            rr.Arrows3D(
                                origins=d_pos_np + np.array([[0.0, 0.0, 0.2]]),
                                vectors=d_vel_np,
                                colors=[255, 140, 80],
                            ),
                        )

                    stream.log("camera/depth", rr.DepthImage(depth_raw, meter=1.0, colormap="turbo"))
                    if kin_ws is not None:
                        kin_hwc = kin_ws.kinematic[ei].permute(1, 2, 0).cpu().numpy()
                        # R=radial_looming, G=wake_appear, B=wake_disappear.
                        rgb = np.clip(kin_hwc[:, :, [2, 0, 1]] * 255.0, 0.0, 255.0).astype(np.uint8)
                        stream.log("camera/kinematic_events", rr.Image(rgb))

                    stream.log("metrics/speed_mps", rr.Scalars(speed))
                    stream.log("metrics/risk_and_ttc/closing_risk_R", rr.Scalars(r_close))
                    stream.log("metrics/risk_and_ttc/min_fov_ttc_s", rr.Scalars(min_ttc))
                    stream.log(
                        "status",
                        rr.TextDocument(
                            f"DiffNav v4.1 · {args.stage} ({args.static_obstacles} static + {args.dynamic_obstacles} dyn)\n"
                            f"Episode {ei} · t={t_s:.2f} s\n"
                            f"Speed: {speed:.2f} m/s · R_closing: {r_close:.2f} · Min FOV TTC: {min_ttc:.2f} s"
                        ),
                    )

                    action = policy(observation)
                    observation, _, _, _, info = env.step(action)
                    evaluation.update(info["transition"])
                    if evaluation.rows[ei] is not None or bool(
                        info["transition"]["terminated"][ei] or info["transition"]["timeout"][ei]
                    ):
                        break

            outcome = evaluation.rows[ei]
            stream.log(
                "status",
                rr.TextDocument(
                    f"Episode {ei} finished: {outcome['outcome']} (t={outcome['time_s']:.2f} s, steps={outcome['step']})\n"
                    f"Speed range: {min_speed:.2f} .. {max_speed:.2f} m/s\n"
                    f"Peak closing risk R={peak_risk:.2f} at t={peak_risk_t:.2f} s"
                ),
            )
            stream.flush()
            stream.disconnect()
            print(
                f"RECORDED_RRD path={args.output} ep={ei} outcome={outcome['outcome']} "
                f"steps={outcome['step']} peak_risk={peak_risk:.3f} at t={peak_risk_t:.2f}s"
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
