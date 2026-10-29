"""Check S0 and Zhang failure continuation in a real 64-aircraft PhysX scene.

Bounds use actual written poses and unchanged detectors. Collision/invalid
injections exercise reset plumbing explicitly; no NaN enters PhysX. A separate
terrain-selected case attempts a real LiDAR collision. This is an environment
integration check, not a policy success-rate or training acceptance result.
"""

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def write_result(output, result, *, emit_marker=True):
    """Persist before Kit shutdown, which may end Python inside context exit."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if emit_marker:
        print("NEW_MECHANISMS_SCENE " + json.dumps(result, allow_nan=False), flush=True)


def write_pose(env, ids, positions, velocities, yaw=None):
    """Use the same indexed official asset writers as the environment reset."""
    import torch
    from isaaclab.utils.math import quat_from_euler_xyz

    ids = torch.as_tensor(ids, device=env.device, dtype=torch.int32)
    positions = torch.as_tensor(positions, device=env.device, dtype=torch.float32)
    velocities = torch.as_tensor(velocities, device=env.device, dtype=torch.float32)
    zero = positions.new_zeros(len(ids))
    yaw = zero if yaw is None else torch.as_tensor(yaw, device=env.device).reshape(-1)
    quaternion = quat_from_euler_xyz(zero, zero, yaw)
    env.drone.write_root_pose_to_sim_index(root_pose=torch.cat((positions, quaternion), dim=-1), env_ids=ids)
    root_velocity = positions.new_zeros(len(ids), 6)
    root_velocity[:, :3] = velocities
    env.drone.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=ids)
    env._terms = None


def check_perception(env):
    import torch

    camera = env.scene["depth_camera"]
    observation = env.get_zhang_observation()
    policy = observation["depth_features"]
    assert camera.image_shape == (120, 160)
    assert policy.shape == (64, 1, 12, 16) and bool(torch.isfinite(policy).all())
    stored = camera.data.output["policy_depth"].torch
    assert stored.shape == (64, 12, 16, 1) and bool(torch.isfinite(stored).all())
    begin, end, raw = camera.last_depth_batch
    assert raw.shape == (end - begin, 1, 120, 160) and bool(torch.isfinite(raw).all())
    assert bool(torch.isfinite(env.root_state()).all())
    assert camera._shared_mesh_tracking_ready and camera._shared_mesh_count == 0
    assert camera._shared_mesh_view is None
    assert not any(target.track_mesh_transforms for target in camera._raycast_targets_cfg)
    return {
        "source_resolution_hw": list(camera.image_shape), "policy_shape": list(policy.shape),
        "last_raw_batch_shape": list(raw.shape), "finite": True,
        "tracked_dynamic_bodies": camera._shared_mesh_count,
        "mesh_transform_info": camera.mesh_transform_info,
    }


def real_static_collision(env, action):
    """Probe a few actual vertical terrain surfaces; do not invent a collision."""
    import torch

    axis = torch.arange(-19.8, 20.0, 0.2, device=env.device)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    query = torch.stack((x.flatten(), y.flatten(), torch.full_like(x.flatten(), 2.0)), dim=-1)
    query.requires_grad_()
    with torch.enable_grad():
        distance = env.terrain_distance.signed_distance(query)
        normal = torch.autograd.grad(distance.sum(), query)[0]
    query, distance = query.detach(), distance.detach()
    delta = -distance[:, None] * normal
    closest = query + delta
    deep = distance < -0.05
    assert bool(deep.any()), "Actual terrain must supply interior samples"
    escaped = env.terrain_distance.signed_distance(query[deep] + 0.001 * normal[deep])
    assert bool((escaped > distance[deep]).all()), "Actual terrain interior gradients must escape"
    eligible = ((distance > 0.12) & (distance < 0.28) & torch.isfinite(distance)
                & (normal[:, 2].abs() < 0.02))
    candidates = eligible.nonzero().flatten()
    if not len(candidates):
        return {"status": "no_eligible_vertical_surface", "attempts": [], "synthetic_detector_events": False}
    # Spread candidates across the map instead of probing eight copies of one face.
    selected = candidates[torch.linspace(0, len(candidates) - 1, min(8, len(candidates)),
                                         device=env.device).long()]
    attempts = []
    lane = 6
    ids = torch.tensor([lane], dtype=torch.int32, device=env.device)
    for index in selected.cpu().tolist():
        env._reset_idx(ids)
        position = query[index:index + 1]
        toward = delta[index, :2]
        unit = toward / toward.norm()
        velocity = position.new_zeros(1, 3)
        velocity[0, :2] = torch.stack((-unit[1], unit[0])) * 0.5
        write_pose(env, ids, position, velocity, torch.atan2(toward[1], toward[0])[None])
        target = position.clone()
        target[0, :2] -= 10 * unit
        env.target_pos[lane] = target[0]
        env.target_dir[lane] = target[0] - position[0]
        episode = int(env.episode_id[lane])
        _, _, terminated, timeout, info = env.step(action)
        transition = info["transition"]
        record = {
            "position_m": position[0].tolist(), "nearest_surface_m": closest[index].tolist(),
            "distance_m": float(distance[index]), "collision": bool(transition["collision"][lane]),
            "bounds": bool(transition["bounds"][lane]), "terminated": bool(terminated[lane]),
            "timeout": bool(timeout[lane]), "first_event_code": int(transition["first_event_code"][lane]),
        }
        attempts.append(record)
        if not record["collision"]:
            continue
        assert record["first_event_code"] == 2 and not record["timeout"]
        continuous = env.cfg.variant == "continuous"
        assert record["terminated"] is (not continuous)
        assert int(env.episode_id[lane]) == episode + int(not continuous)
        if continuous:
            before = transition["position"][lane].clone()
            _, _, next_terminated, next_timeout, next_info = env.step(action)
            following = next_info["transition"]
            displacement = float((following["position"][lane] - before).norm())
            assert displacement > 1e-5
            assert not bool(next_terminated[lane] | next_timeout[lane])
            assert int(env.episode_id[lane]) == episode
            assert int(following["first_event_code"][lane]) == 2
            torch.testing.assert_close(following["first_event_time_s"][lane],
                                       transition["first_event_time_s"][lane], rtol=0, atol=0)
            record["continued_displacement_m"] = displacement
        return {"status": "observed", "attempts": attempts, "synthetic_detector_events": False}
    # Report an unobserved optional geometry probe honestly. The explicit event
    # checks below still establish reset semantics, not a contact stress result.
    return {"status": "not_observed", "attempts": attempts, "synthetic_detector_events": False}


def check_failures(env, action):
    import torch

    continuous = env.cfg.variant == "continuous"
    positions = action.new_zeros(env.num_envs, 3)
    positions[:, 0] = torch.linspace(-9.0, 9.0, env.num_envs, device=env.device)
    positions[:, 1], positions[:, 2] = -23.0, 2.0  # The terrain's flat five-metre border.
    positions[0, 2], positions[1, 2] = 0.15, 4.3
    velocity = torch.zeros_like(positions)
    velocity[:, 0] = 0.75
    write_pose(env, torch.arange(env.num_envs, device=env.device), positions, velocity)
    env.target_pos.copy_(positions)
    env.target_pos[:, 1], env.target_pos[:, 2] = 23.0, 2.0
    env.target_dir.copy_(env.target_pos - positions)
    env.height_range[:] = action.new_tensor((0.5, 2.5))
    original_terms, original_events = env._compute_terms, env._compute_first_events
    overrides = {2: "collision"}

    def inject(original):
        def compute():
            terms = original()
            for lane, event in overrides.items():
                for name in ("collision", "bounds", "invalid", "reached"):
                    terms[name][lane] = name == event
            return terms
        return compute

    env._compute_terms = inject(original_terms)
    env._compute_first_events = inject(original_events)
    result = {"synthetic_detector_events": {"collision": [2], "invalid": [3]},
              "real_pose_bounds": {"lower_lane": 0, "lower_z_m": 0.15, "upper_lane": 1, "upper_z_m": 4.3}}
    try:
        initial_episodes = env.episode_id.clone()
        _, _, terminated, timeout, info = env.step(action)
        transition = info["transition"]
        expected = action.new_tensor([2, 3, 2], dtype=torch.int8)
        assert bool(transition["bounds"][:2].all())
        assert bool(transition["collision"][0]) and not bool(transition["collision"][1])
        assert bool(transition["collision"][2]) and not bool(timeout[:3].any())
        torch.testing.assert_close(transition["first_event_code"][:3], expected, rtol=0, atol=0)
        torch.testing.assert_close(terminated[:3], torch.full_like(terminated[:3], not continuous), rtol=0, atol=0)
        torch.testing.assert_close(env.episode_id[:3], initial_episodes[:3] + int(not continuous), rtol=0, atol=0)
        first_times = transition["first_event_time_s"][:3].clone()
        first_position = transition["position"][:3].clone()
        displacement = first_position[:, 0] - positions[:3, 0]
        if continuous:
            assert bool((displacement > 1e-4).all())
        result["first_failure"] = {
            "episode_id_before": initial_episodes[:3].tolist(), "episode_id_after": env.episode_id[:3].tolist(),
            "first_event_codes": transition["first_event_code"][:3].tolist(),
            "first_event_times_s": first_times.tolist(), "forward_displacement_m": displacement.tolist(),
            "terminated": terminated[:3].tolist(),
        }
        if env.cfg.variant == "dcol":
            dynamic = transition["dynamic_positions_by_substep"]
            assert dynamic.shape == (env.cfg.decimation, 0, 3) and bool(torch.isfinite(dynamic).all())
            result["empty_dynamic_substep_shape"] = list(dynamic.shape)

        # Force a recovery and a genuine geometric arrival in the same episode
        # for continuous; dcol has already started a new episode at this point.
        overrides.clear()
        recovered = first_position.clone()
        recovered[:, 2] = 2.0
        write_pose(env, [0, 1, 2], recovered, velocity[:3])
        env.target_pos[:3] = recovered
        env.target_dir[:3] = action.new_tensor((1.0, 0.0, 0.0))
        before_recovery = env.episode_id[:3].clone()
        _, _, terminated, timeout, recovery_info = env.step(action)
        recovered_transition = recovery_info["transition"]
        assert bool(recovered_transition["reached"][:3].all())
        assert not bool((terminated[:3] | timeout[:3]).any())
        torch.testing.assert_close(env.episode_id[:3], before_recovery, rtol=0, atol=0)
        recovery_codes = expected if continuous else torch.ones_like(expected)
        torch.testing.assert_close(recovered_transition["first_event_code"][:3], recovery_codes, rtol=0, atol=0)
        if continuous:
            torch.testing.assert_close(recovered_transition["first_event_time_s"][:3], first_times, rtol=0, atol=0)
            assert bool((env.first_reach_step[:3] == 0).all())
        else:
            assert bool((env.first_reach_step[:3] > 0).all())
        result["recovery"] = {"actual_reached": True, "episode_id_unchanged": True,
                              "first_event_codes": recovery_codes.tolist(),
                              "prior_failure_not_overwritten": continuous}

        # Exercise invalid without writing NaNs into the physics engine. The
        # timeout lane has never reached or failed, so its first code must be 5.
        overrides[3] = "invalid"
        before_reset = env.episode_id.clone()
        env.episode_length_buf[4] = env.max_episode_length - 1
        _, _, terminated, timeout, reset_info = env.step(action)
        reset_transition = reset_info["transition"]
        assert bool(terminated[3]) and not bool(timeout[3])
        assert bool(timeout[4]) and not bool(terminated[4])
        assert int(reset_transition["first_event_code"][3]) == 4
        assert int(reset_transition["first_event_code"][4]) == 5
        torch.testing.assert_close(env.episode_id[3:5], before_reset[3:5] + 1, rtol=0, atol=0)
        assert int(env.episode_id[5]) == int(before_reset[5])
        assert bool((env._event_code[3:5] == 0).all()) and bool((env._event_time[3:5] == 0).all())
        overrides.clear()
        before_timeout = env.episode_id[:3].clone()
        env.episode_length_buf[:3] = env.max_episode_length - 1
        _, _, terminated, timeout, final_info = env.step(action)
        assert bool(timeout[:3].all()) and not bool(terminated[:3].any())
        torch.testing.assert_close(final_info["transition"]["first_event_code"][:3], recovery_codes, rtol=0, atol=0)
        torch.testing.assert_close(env.episode_id[:3], before_timeout + 1, rtol=0, atol=0)
        assert bool((env._event_code[:3] == 0).all())
        # Same-step resets and subsequent simulation must not mutate snapshots.
        torch.testing.assert_close(transition["first_event_code"][:3], expected, rtol=0, atol=0)
        torch.testing.assert_close(transition["position"][:3], first_position, rtol=0, atol=0)
        result["reset_checks"] = {"invalid_resets": True, "timeout_resets": True,
                                  "unaffected_lane_keeps_episode": True, "first_outcome_survives_to_timeout": True,
                                  "reset_clears_first_event": True, "old_snapshot_immutable": True}
        return result
    finally:
        env._compute_terms = original_terms
        env._compute_first_events = original_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("continuous", "dcol"), default="continuous")
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()

    import gymnasium as gym
    import torch
    from sim_to_real_scene import drone_collision_attributes

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import ZhangNavRLEnvCfg

    result = {"test": "new_mechanisms_scene", "variant": args.variant, "stage": "pretrain_s0",
              "num_envs": 64, "static_obstacles": 300, "dynamic_obstacles": 0,
              "randomization": False, "policy_acceptance_test": False, "training_success_statistics": False,
              "status": "running", "current_check": "configuration"}

    def progress(name):
        result["current_check"] = name
        write_result(args.output, result, emit_marker=False)
        print("NEW_MECHANISMS_CHECK " + name, flush=True)

    try:
        progress("configuration")
        cfg = ZhangNavRLEnvCfg()
        cfg.scene.num_envs = 64
        cfg.num_static_obstacles, cfg.num_dynamic_obstacles = 300, 0
        cfg.training_stage, cfg.training_curriculum = "pretrain_s0", True
        cfg.variant, cfg.sim.device = args.variant, args.device
        cfg.sim_to_real.enabled = False
        cfg.configure_scene()
        progress("launch_simulation")
        with launch_simulation(cfg, args):
            env = None
            try:
                progress("create_environment")
                env = gym.make("Isaac-NavRL-Zhang-Hummingbird-Direct-v0", cfg=cfg).unwrapped
                progress("reset_and_empty_dynamic_scene")
                observation, _ = env.reset()
                initial_step = int(env.common_step_counter)
                assert observation["policy"].shape == (64, 202) and bool(torch.isfinite(observation["policy"]).all())
                assert env.dynamic_collection is None and env.cfg.dynamic_capacity() == 0
                assert env.motion.position.shape == env.motion.size.shape == (0, 3)
                assert env.curriculum.active_obstacles == 0
                assert not env.sim.stage.GetPrimAtPath("/World/Dynamic/obstacle_000").IsValid()
                env.set_active_obstacles(0)  # Also exercise the previously unsafe missing-pose-buffer branch.
                progress("usd_collision_attributes")
                collision_attributes = drone_collision_attributes(env)
                result["drone_collision_attributes"] = collision_attributes
                result["physx_drone_contacts_disabled"] = not any(collision_attributes.values())
                assert result["physx_drone_contacts_disabled"] is (args.variant == "continuous")
                action = torch.zeros(64, 3, device=env.device)
                progress("bounds_collision_recovery_and_resets")
                result["failure_checks"] = check_failures(env, action)
                progress("real_static_collision")
                result["real_static_collision"] = real_static_collision(env, action)
                progress("depth_camera")
                result["depth_camera"] = check_perception(env)
                progress("static_only_distance_gradient")
                query = env.root_state()[:, None, :3].detach().requires_grad_()
                vectors = env.obstacle_signed_distance(query)
                vectors.square().sum().backward()
                assert bool(torch.isfinite(vectors).all() & torch.isfinite(query.grad).all())
                result["static_only_distance_gradient_finite"] = True
                result["nonfinite_event_count_includes_injection"] = int(env.nonfinite_events)
                result["nominal_dynamics"] = env.nominal_dynamics
                result["policy_steps"] = int(env.common_step_counter) - initial_step
                result["physics_steps"] = result["policy_steps"] * env.cfg.decimation
                result.update(status="passed", current_check="complete")
                write_result(args.output, result)
            except BaseException as error:
                result.update(status="failed", error=f"{type(error).__name__}: {error}",
                              traceback=traceback.format_exc())
                write_result(args.output, result)
                raise
            finally:
                if env is not None:
                    env.close()
    except BaseException as error:
        # Best effort for configuration/launcher errors before entering Kit.
        # In-context assertion failures were already persisted before close.
        if result["status"] != "failed":
            result.update(status="failed", error=f"{type(error).__name__}: {error}",
                          traceback=traceback.format_exc())
            write_result(args.output, result)
        raise


if __name__ == "__main__":
    main()
