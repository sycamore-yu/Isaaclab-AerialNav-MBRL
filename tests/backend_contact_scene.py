"""Verify forest free space and real static contacts with uncontrolled free fall.

Run this GPU integration check directly after installing the project::

    .venv/bin/python tests/backend_contact_scene.py physics=newton_mjwarp
    .venv/bin/python tests/backend_contact_scene.py physics=isaacsim_physx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_navrl.physics import parse_physics_args, select_physics


def choose_drop_locations(env):
    """Locate an interior corridor and a broad obstacle using the actual map SDF."""
    import torch

    searches = []
    for spacing in (0.2, 0.1, 0.05):
        axis = torch.arange(-18.0, 18.0 + spacing / 2, spacing, device=env.device)
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        points = torch.stack((x.flatten(), y.flatten(), torch.full_like(x.flatten(), 2.0)), dim=-1)
        distances = torch.cat(
            [env.terrain_distance.signed_distance(chunk).reshape(-1) for chunk in points.split(8192)]
        )
        assert torch.isfinite(distances).all(), "Terrain SDF returned non-finite distances."
        free_index, solid_index = distances.argmax(), distances.argmin()
        free_distance, solid_distance = float(distances[free_index]), float(distances[solid_index])
        searches.append(
            {"spacing_m": spacing, "points": len(points), "max_sdf_m": free_distance, "min_sdf_m": solid_distance}
        )
        if free_distance > 0.6 and solid_distance < -0.35:
            selected = points[torch.stack((free_index, solid_index))].clone()
            return selected, {
                "searches": searches,
                "free_sdf_m": free_distance,
                "solid_sdf_m": solid_distance,
                "sdf_query_height_m": 2.0,
            }
    raise AssertionError(f"No sufficiently clear corridor and solid obstacle were found: {searches}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--output", type=Path)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parse_physics_args(parser)
    if not 200 <= args.steps <= 300:
        parser.error("--steps must be between 200 and 300 for the free-fall contact check")
    output = args.output or Path(f"logs/tests/backend_contact_{args.physics_preset}.json")

    import gymnasium as gym
    import torch

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import NavRLEnvCfg

    cfg = NavRLEnvCfg()
    cfg.scene.num_envs = 2
    cfg.training_curriculum = False
    cfg.num_dynamic_obstacles = 0
    cfg.evaluation = True
    cfg.sim_to_real.enabled = False
    cfg.sim.device = args.device
    select_physics(cfg, args.physics_preset)
    cfg.configure_scene()

    result = {
        "physics": args.physics_preset,
        "task": "Isaac-NavRL-Hummingbird-Direct-v0",
        "num_envs": 2,
        "static_obstacles": cfg.num_static_obstacles,
        "dynamic_obstacles": 0,
        "steps_requested": args.steps,
        "passed": False,
    }
    trajectory = []
    persisted = False

    def persist():
        nonlocal persisted
        result["steps_completed"] = len(trajectory)
        result["trajectory_state_xyzw"] = trajectory
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        summary = {key: value for key, value in result.items() if key != "trajectory_state_xyzw"}
        print("BACKEND_CONTACT_RESULT " + json.dumps(summary), flush=True)
        persisted = True

    try:
        with launch_simulation(cfg, args):
            env = gym.make(result["task"], cfg=cfg).unwrapped
            try:
                env.reset()
                result["physics_backend"] = env.drone.__backend_name__
                with torch.no_grad():
                    locations, selection = choose_drop_locations(env)
                    result["location_selection"] = selection
                    locations[:, 2] = torch.tensor([2.0, 6.0], device=env.device)
                    pose = torch.zeros(2, 7, device=env.device)
                    pose[:, :3] = locations
                    pose[:, 6] = 1.0  # Isaac Lab 3.0 quaternion order is XYZW.
                    env.drone.write_root_pose_to_sim_index(root_pose=pose)
                    env.drone.write_root_velocity_to_sim_index(root_velocity=torch.zeros(2, 6, device=env.device))
                    env.drone.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(env.drone.data.joint_vel.torch))

                    # Clear every body: only gravity and physical contacts may act.
                    body_ids = list(range(len(env.drone.body_names)))
                    zeros = torch.zeros(2, len(body_ids), 3, device=env.device)
                    env.drone.permanent_wrench_composer.set_forces_and_torques_index(
                        forces=zeros, torques=zeros, body_ids=body_ids, is_global=False
                    )
                    result["initial_position_m"] = locations.cpu().tolist()
                    result["physics_dt_s"] = env.physics_dt

                    # Avoid env.step: task collisions, bounds, and automatic reset
                    # would hide the solver's response to the drop experiment.
                    for step in range(args.steps):
                        env.scene.write_data_to_sim()
                        env.sim.step(render=False)
                        env.scene.update(env.physics_dt)
                        state = env.root_state().detach().clone()
                        assert torch.isfinite(state).all(), f"Non-finite rigid-body state at step {step + 1}"
                        trajectory.append(state.cpu().tolist())

                    final_state = env.root_state()
                    free_height, solid_height = (float(value) for value in final_state[:, 2])
                    result["final_position_m"] = final_state[:, :3].cpu().tolist()
                    result["final_linear_velocity_m_s"] = final_state[:, 7:10].cpu().tolist()
                    result["free_height_m"] = free_height
                    result["solid_height_m"] = solid_height
                    result["solid_drop_m"] = 6.0 - solid_height
                    # A narrow obstacle can tip the aircraft onto the ground.
                    # Detect the airborne contact impulse instead of requiring
                    # the uncontrolled aircraft to balance on its top surface.
                    trace = torch.tensor(trajectory, dtype=torch.float64)
                    gravity = torch.tensor(env.cfg.sim.gravity, dtype=torch.float64)
                    impulse = trace[1:, :, 7:10] - trace[:-1, :, 7:10] - gravity * env.physics_dt
                    height = torch.minimum(trace[1:, :, 2], trace[:-1, :, 2])
                    impulse_norm = impulse.norm(dim=-1)
                    free_airborne = height[:, 0] > 1.0
                    solid_impact = (height[:, 1] > 1.0) & (trace[1:, 1, 2] < 5.5) & (impulse[:, 1, 2] > 0.5)
                    result["free_airborne_max_impulse_m_s"] = float(impulse_norm[free_airborne, 0].max())
                    impact_indices = solid_impact.nonzero().flatten()
                    result["airborne_obstacle_impacts"] = [
                        {
                            "step": int(index) + 2,
                            "position_m": trace[index + 1, 1, :3].tolist(),
                            "delta_velocity_minus_gravity_m_s": impulse[index, 1].tolist(),
                        }
                        for index in impact_indices
                    ]
                    assert -0.1 <= free_height <= 0.6, (
                        f"The free corridor did not reach the ground: z={free_height:.4f} m. "
                        "Check for a forest-wide convex hull or missing ground contacts."
                    )
                    assert result["free_airborne_max_impulse_m_s"] < 0.1, (
                        "The clear corridor produced an unexpected airborne contact impulse."
                    )
                    assert len(impact_indices) > 0, (
                        "No airborne obstacle contact interrupted the solid-location free fall."
                    )
                    assert 6.0 - solid_height >= 1.0, (
                        f"The obstacle drop descended only {6.0 - solid_height:.4f} m. "
                        "Check gravity, residual forces, or an invalid initial collision."
                    )
                    result["passed"] = True
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                # Kit may terminate the process on shutdown; save before closing.
                persist()
                env.close()
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if not persisted:
            persist()


if __name__ == "__main__":
    main()
