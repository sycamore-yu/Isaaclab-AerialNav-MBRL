"""Check shared perturbations against the selected backend and partial resets."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def drone_collision_attributes(env):
    """Inspect collision flags for every aircraft in the active representation."""
    from pxr import Usd, UsdPhysics

    attributes = {}
    continuous = env.cfg.method == "zhang" and env.cfg.variant == "continuous"
    if env.drone.__backend_name__ == "newton":
        from isaaclab_newton.physics import NewtonManager
        from newton import ShapeFlags

        model = NewtonManager.get_model()
        bodies, flags = model.shape_body.numpy(), model.shape_flags.numpy()
        for env_path in env.scene.env_prim_paths:
            prefix = f"{env_path}/Hummingbird/"
            enabled = []
            for index, (body, flag) in enumerate(zip(bodies, flags, strict=True)):
                if body >= 0 and model.body_label[body].startswith(prefix):
                    value = bool(flag & int(ShapeFlags.COLLIDE_SHAPES))
                    attributes[f"{model.shape_label[index]}.newton:collide_shapes"] = value
                    enabled.append(value)
            assert enabled, f"No Newton drone shapes found under {prefix}"
            assert not any(enabled) if continuous else any(enabled)
        return attributes

    for env_path in env.scene.env_prim_paths:
        drone = env.sim.stage.GetPrimAtPath(f"{env_path}/Hummingbird")
        enabled = []
        for prim in Usd.PrimRange(drone, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                attribute = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
                value = attribute.Get()
                assert isinstance(value, bool)
                attributes[str(attribute.GetPath())] = value
                enabled.append(value)
        assert enabled, f"No drone collision attributes found under {drone.GetPath()}"
        assert not any(enabled) if continuous else any(enabled)
    return attributes


def check_zhang_collision_mode(env):
    """Inject detector events into real steps; these are not training results."""
    import torch

    env.reset()
    moving_ids = torch.tensor([0, 1], device=env.device, dtype=torch.int32)
    initial_velocity = torch.zeros(2, 6, device=env.device)
    initial_velocity[:, 0] = 0.75
    env.drone.write_root_velocity_to_sim_index(root_velocity=initial_velocity, env_ids=moving_ids)
    original_terms = env._compute_terms
    original_events = env._compute_first_events
    injected = {0: "collision", 1: "reached"}
    observed_collisions = {}
    observed_dynamic_positions = {}
    observed_velocities = {}
    observed_dynamic_velocities = {}

    def inject(original):
        def compute():
            terms = original()
            for index, event in injected.items():
                for name in ("collision", "reached", "bounds", "invalid"):
                    terms[name][index] = name == event
                if event == "collision" and env._physical_substep > 0 and index not in observed_collisions:
                    observed_collisions[index] = (env._physical_substep, env.root_state()[index, :3].clone())
                    observed_velocities[index] = env.root_state()[index, 7:10].clone()
                    if env.cfg.variant == "dcol":
                        observed_dynamic_positions[index] = env.motion.position.clone()
                        observed_dynamic_velocities[index] = env.motion.velocity.clone()
            return terms

        return compute

    env._compute_terms = inject(original_terms)
    env._compute_first_events = inject(original_events)
    action = torch.zeros(env.num_envs, 3, device=env.device)
    continuous = env.cfg.variant == "continuous"

    def step():
        observed_collisions.clear()
        observed_dynamic_positions.clear()
        observed_velocities.clear()
        observed_dynamic_velocities.clear()
        result = env.step(action)
        transition = result[-1]["transition"]
        done = result[2] | result[3]
        if bool(done.any()):
            # Same-step resets must retain the completed transition's metrics.
            expected_logs = {
                "Episode/return": transition["episode_return"],
                "Episode/length": transition["step"],
                "Episode/reach_goal": transition["reached"],
                "Episode/collision": transition["collision"],
            }
            for name, values in expected_logs.items():
                torch.testing.assert_close(result[-1]["log"][name], values[done].float().mean())
        positions, substeps = transition["collision_position"], transition["collision_substep"]
        assert positions.shape == (env.num_envs, 3) and torch.isfinite(positions).all()
        assert substeps.shape == (env.num_envs,) and substeps.dtype == torch.long
        assert bool(((substeps >= 0) & (substeps <= env.cfg.decimation)).all())
        torch.testing.assert_close(substeps > 0, transition["collision"], rtol=0, atol=0)
        for index, event in injected.items():
            if event == "collision":
                observed_substep, observed_position = observed_collisions[index]
                assert observed_substep == 1 and int(substeps[index]) == observed_substep
                torch.testing.assert_close(positions[index], observed_position, rtol=0, atol=0)
                torch.testing.assert_close(
                    transition["collision_velocity"][index], observed_velocities[index], rtol=0, atol=0
                )
            else:
                assert int(substeps[index]) == 0
                torch.testing.assert_close(positions[index], action[index], rtol=0, atol=0)
        if env.cfg.variant == "dcol":
            dynamic = transition["dynamic_positions_by_substep"]
            assert dynamic.shape == (env.cfg.decimation, len(env.motion.position), 3)
            assert torch.isfinite(dynamic).all()
            torch.testing.assert_close(dynamic[-1], env.motion.position, rtol=0, atol=0)
            for index, observed in observed_dynamic_positions.items():
                observed_substep = observed_collisions[index][0]
                torch.testing.assert_close(dynamic[observed_substep - 1], observed, rtol=0, atol=0)
                torch.testing.assert_close(
                    transition["dynamic_velocities_by_substep"][observed_substep - 1],
                    observed_dynamic_velocities[index],
                    rtol=0,
                    atol=0,
                )
        else:
            assert "dynamic_positions_by_substep" not in transition
            assert env._dynamic_positions_by_substep is None
        return result

    try:
        episode_id = env.episode_id.clone()
        _, _, terminated, _, info = step()
        transition = info["transition"]
        first_transition = transition
        first_position = transition["collision_position"][0].clone()
        first_terminal_position = transition["position"][0].clone()
        terminal_displacement = float((first_terminal_position - first_position).norm())
        assert terminal_displacement > 1e-5, "The first-substep snapshot must precede the moving terminal pose"
        first_dynamic = transition["dynamic_positions_by_substep"].clone() if env.cfg.variant == "dcol" else None
        assert bool(transition["collision"][0])
        assert bool(terminated[0]) == (not continuous)
        assert bool(transition["terminated"][0]) == (not continuous)
        assert int(transition["first_event_code"][0]) == 2
        assert int(transition["first_event_code"][1]) == 1
        first_times = transition["first_event_time_s"][:2].clone()
        if continuous:
            torch.testing.assert_close(env.episode_id[:2], episode_id[:2])
            injected.update({0: "reached", 1: "collision"})
            _, _, terminated, _, info = step()
            transition = info["transition"]
            assert not bool(terminated[:2].any())
            assert bool(transition["reached"][0]) and not bool(transition["collision"][0])
            torch.testing.assert_close(transition["first_event_code"][:2], action.new_tensor([2, 1], dtype=torch.int8))
            torch.testing.assert_close(transition["first_event_time_s"][:2], first_times, rtol=0, atol=0)
            assert int(env.first_reach_step[0]) == 0
            assert int(env.first_reach_step[1]) > 0
            kept_position = env._collision_position[1].clone()
            kept_substep = env._collision_substep[1].clone()
            kept_logs = {name: value.clone() for name, value in env.extras.get("log", {}).items()}
            env._reset_idx(torch.tensor([0], device=env.device, dtype=torch.int32))
            assert int(env._collision_substep[0]) == 0
            torch.testing.assert_close(env._collision_position[0], action[0], rtol=0, atol=0)
            torch.testing.assert_close(env._collision_position[1], kept_position, rtol=0, atol=0)
            torch.testing.assert_close(env._collision_substep[1], kept_substep, rtol=0, atol=0)
            assert int(env._event_code[0]) == 0 and float(env._event_time[0]) == 0.0
            assert int(env.first_reach_step[0]) == 0
            assert int(env._event_code[1]) == 1
            torch.testing.assert_close(env._event_time[1], first_times[1], rtol=0, atol=0)
            # Clear a selected nonzero collision buffer without changing the
            # other aircraft's freshly reset first-outcome state.
            env._reset_idx(torch.tensor([1], device=env.device, dtype=torch.int32))
            assert int(env._collision_substep[1]) == 0 and int(env._event_code[0]) == 0
            torch.testing.assert_close(env._collision_position[1], action[1], rtol=0, atol=0)
            # Manual partial resets must not manufacture or replace episode logs.
            current_logs = env.extras.get("log", {})
            assert current_logs.keys() == kept_logs.keys()
            for name, value in kept_logs.items():
                torch.testing.assert_close(current_logs[name], value, rtol=0, atol=0)
            injected.update({0: "reached", 1: None})
            _, _, _, _, info = step()
            assert int(info["transition"]["first_event_code"][0]) == 1
            assert int(env.first_reach_step[0]) > 0
            injected[0] = "bounds"
            before_bounds = env.episode_id[0].clone()
            _, _, terminated, timeout, info = step()
            assert not bool(terminated[0] | timeout[0])
            torch.testing.assert_close(env.episode_id[0], before_bounds, rtol=0, atol=0)
            assert int(info["transition"]["first_event_code"][0]) == 1
            injected[0] = "invalid"
            _, _, terminated, _, _ = step()
            assert bool(terminated[0])
            assert int(env.episode_id[0]) == int(before_bounds) + 1
            injected[0] = None
            env.episode_length_buf[0] = env.max_episode_length - 1
            _, _, terminated, timeout, info = step()
            assert bool(timeout[0]) and not bool(terminated[0])
            assert int(info["transition"]["first_event_code"][0]) == 5
        else:
            assert int(env.episode_id[0]) == int(episode_id[0]) + 1
            assert int(env._event_code[0]) == 0 and float(env._event_time[0]) == 0.0
            assert int(env._collision_substep[0]) == 0
            torch.testing.assert_close(env._collision_position[0], action[0], rtol=0, atol=0)
            injected.update({0: None, 1: "collision"})
            _, _, terminated, _, info = step()
            assert bool(terminated[1])
            assert int(info["transition"]["first_event_code"][1]) == 1
            assert int(env.episode_id[1]) == int(episode_id[1]) + 1
            assert int(env._collision_substep[1]) == 0
            torch.testing.assert_close(env._collision_position[1], action[1], rtol=0, atol=0)
        # Later steps and resets must not mutate an already returned transition.
        torch.testing.assert_close(first_transition["collision_position"][0], first_position, rtol=0, atol=0)
        assert int(first_transition["collision_substep"][0]) == 1
        if first_dynamic is not None:
            torch.testing.assert_close(first_transition["dynamic_positions_by_substep"], first_dynamic, rtol=0, atol=0)
        return {
            "variant": env.cfg.variant,
            "collision_terminates": not continuous,
            "first_outcome_preserved": True,
            "reset_clears_outcome": True,
            "synthetic_detector_events": True,
            "policy_steps": 6 if continuous else 2,
            "collision_capture": {
                "first_substep": 1,
                "first_time_s": env.physics_dt,
                "first_position_m": first_position.tolist(),
                "terminal_position_m": first_terminal_position.tolist(),
                "terminal_displacement_m": terminal_displacement,
                "collision_after_success_recorded": True,
                "no_collision_uses_zero": True,
                "snapshot_survives_reset": True,
                "partial_reset_tested": continuous,
                "autoreset_tested": not continuous,
                "dynamic_positions_by_substep_shape": list(first_dynamic.shape) if first_dynamic is not None else None,
            },
        }
    finally:
        env._compute_terms = original_terms
        env._compute_first_events = original_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("navrl", "zhang"), required=True)
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--variant", choices=("step", "continuous", "dcol"), default="step")
    parser.add_argument("--backend", choices=("isaacsim_physx", "newton_mjwarp"), default="isaacsim_physx")
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()
    if args.method != "zhang" and args.variant != "step":
        parser.error("--variant applies only to the Zhang task")

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.hydra import resolve_presets

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import NavRLEnvCfg, ZhangNavRLEnvCfg
    from isaaclab_navrl.tasks.navigation.randomization import DifferentiableDelay

    cfg = (ZhangNavRLEnvCfg if args.method == "zhang" else NavRLEnvCfg)()
    resolve_presets(cfg, selected=(args.backend,))
    cfg.scene.num_envs = 16
    cfg.num_static_obstacles, cfg.num_dynamic_obstacles = 300, 60
    cfg.training_curriculum = False
    cfg.training_stage = "finetune_s1" if args.enabled else "pretrain_s1"
    cfg.sim_to_real.enabled = args.enabled
    cfg.sim.device = args.device
    if args.method == "zhang":
        cfg.timing = "baseline"  # Multiple physics substeps are required by the first-collision snapshot checks.
        # Exercise repeated configuration and restore the requested variant before launch.
        cfg.variant = "continuous"
        cfg.configure_scene()
        assert cfg.scene.drone.spawn.collision_props.collision_enabled is False
        cfg.variant = args.variant
    cfg.sim_to_real.action_delay_range_s = (0, 2 * cfg.sim.dt * cfg.decimation)
    cfg.configure_scene()
    if args.method == "zhang" and args.variant != "continuous":
        assert cfg.scene.drone.spawn.collision_props is None
    task = "Isaac-NavRL-Zhang-Hummingbird-Direct-v0" if args.method == "zhang" else "Isaac-NavRL-Hummingbird-Direct-v0"
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            observation, _ = env.reset()
            assert observation["policy"].shape == (env.num_envs, 202)
            camera = env.scene["depth_camera"]
            assert camera.image_shape == (120, 160)
            assert abs(camera.cfg.update_period - env.step_dt) < 1e-9
            collision_attributes = drone_collision_attributes(env)
            actual_mass = env.drone.data.body_mass.torch
            mass_ratio = actual_mass / env.nominal_body_mass
            inertia = env.drone.data.body_inertia.torch
            nominal_diag = env.nominal_body_inertia[..., [0, 4, 8]]
            inertia_ratio = inertia[..., [0, 4, 8]] / nominal_diag
            bounds = (0.9, 1.1) if args.enabled else (1.0, 1.0)
            for ratio in (mass_ratio, inertia_ratio):
                assert bool(((ratio >= bounds[0] - 1e-6) & (ratio <= bounds[1] + 1e-6)).all())
            lags = torch.arange(16, device=env.device, dtype=torch.int) % 3
            env.perturbation.action_buffer.set_time_lag(lags)
            history = DifferentiableDelay(2)
            max_execution_error = 0.0
            for step in range(12):
                command = torch.full((16, 3), 0.02 * (step + 1), device=env.device, requires_grad=True)
                expected = history.push(
                    command, env.perturbation.action_buffer.time_lags.clone(), zero_until_ready=True
                )
                observation, _, terminated, timeout, info = env.step(command.detach())
                assert observation["policy"].shape == (env.num_envs, 202)
                assert torch.isfinite(observation["policy"]).all()
                policy_depth = camera.data.output["policy_depth"].torch
                assert policy_depth.shape == (env.num_envs, 12, 16, 1)
                assert torch.isfinite(policy_depth).all()
                if env.dynamic_collection is not None:
                    obstacle_poses = env.dynamic_collection.data.body_pose_w.torch
                    torch.testing.assert_close(
                        obstacle_poses, obstacle_poses[:1].expand_as(obstacle_poses), atol=1e-6, rtol=0
                    )
                    torch.testing.assert_close(
                        obstacle_poses[0, env._active_obstacle_ids, :3], env.motion.position, atol=1e-6, rtol=0
                    )
                else:
                    import warp as wp

                    camera._update_mesh_transforms()
                    obstacle_poses = wp.to_torch(camera._mesh_positions_w)
                    torch.testing.assert_close(
                        obstacle_poses, obstacle_poses[:1].expand_as(obstacle_poses), atol=1e-6, rtol=0
                    )
                    torch.testing.assert_close(
                        obstacle_poses[0, env._active_obstacle_ids + 2], env.motion.position, atol=1e-6, rtol=0
                    )
                begin, end, raw_depth = camera.last_depth_batch
                assert raw_depth.shape == (end - begin, 1, 120, 160)
                assert torch.isfinite(raw_depth).all()
                actual = info["transition"]["executed_action"]
                torch.testing.assert_close(actual, expected.detach(), rtol=0, atol=0)
                max_execution_error = max(max_execution_error, float((actual - expected.detach()).abs().max()))
                done = terminated | timeout
                history.reset(done)
                if args.method == "zhang":
                    clean = env.root_state().clone()
                    first = env.get_zhang_observation()
                    pushes = env.perturbation.observation_buffer.num_pushes.clone()
                    second = env.get_zhang_observation()
                    torch.testing.assert_close(
                        first.policy["depth_features"], second.policy["depth_features"], rtol=0, atol=0
                    )
                    torch.testing.assert_close(env.perturbation.observation_buffer.num_pushes, pushes)
                    torch.testing.assert_close(env.root_state(), clean, rtol=0, atol=0)
                    for key, source in (
                        ("mass_scale", env.mass_scale),
                        ("inertia_scale", env.inertia_scale),
                        ("thrust_scale", env.perturbation.thrust_scale),
                        ("control_gain_scale", env.perturbation.control_gain_scale),
                    ):
                        torch.testing.assert_close(env.zhang_parameters[key], source)
                if step == 5:
                    env._reset_idx(torch.tensor([0], device=env.device, dtype=torch.int32))
                    mask = torch.zeros(16, dtype=torch.bool, device=env.device)
                    mask[0] = True
                    history.reset(mask)
                    env.perturbation.action_buffer.set_time_lag(2, [0])
                assert torch.isfinite(env.root_state()).all()
            collision_check = check_zhang_collision_mode(env) if args.method == "zhang" else None
            policy_steps = 12 + (collision_check["policy_steps"] if collision_check is not None else 0)
            result = {
                "physics": args.backend,
                "method": args.method,
                "enabled": args.enabled,
                "num_envs": 16,
                "variant": args.variant if args.method == "zhang" else None,
                "physics_steps": policy_steps * cfg.decimation,
                "policy_steps": policy_steps,
                "nominal_dynamics": env.nominal_dynamics,
                "sim_to_real": asdict(cfg.sim_to_real),
                "mass_scale_min": float(mass_ratio.min()),
                "mass_scale_max": float(mass_ratio.max()),
                "inertia_scale_min": float(inertia_ratio.min()),
                "inertia_scale_max": float(inertia_ratio.max()),
                "executed_action_max_error": max_execution_error,
                "partial_reset_passed": True,
                "obstacle_instances_follow_shared_motion": True,
                "nonfinite_events": int(env.nonfinite_events),
                "drone_collision_attributes": collision_attributes,
                "collision_mode_check": collision_check,
                "policy_observation_shape": list(observation["policy"].shape),
                "depth_camera": {
                    "source_resolution": list(camera.image_shape),
                    "policy_depth_shape": list(policy_depth.shape),
                    "last_raw_batch_shape": list(raw_depth.shape),
                    "update_period_s": camera.cfg.update_period,
                    "pixel_noise_std": camera.cfg.pixel_noise_std,
                    "finite_outputs": True,
                },
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("SIM_TO_REAL_CHECK " + json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
