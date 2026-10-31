"""Real Newton/Warp navigation contract checks; run each method in a process.

Examples:
    .venv/bin/python tests/warp_environment_contract.py --method aerialnav
    .venv/bin/python tests/warp_environment_contract.py --method diffnav --variant dcol
    .venv/bin/python tests/warp_environment_contract.py --method diffnav --variant step
    .venv/bin/python tests/warp_environment_contract.py --method diffnav --variant continuous

The fixed one-tick delays exercise delivery deterministically. The bounds test
writes a real valid pose; the invalid test injects a detector flag, never NaNs
into the simulator. This script tests environment semantics, not training quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from aerialnav.physics import create_navigation_env, parse_physics_args, runtime_metadata, select_physics


def run_contract(env, steps):
    import torch
    from isaaclab_experimental.envs import DirectRLEnvWarp

    assert isinstance(env, DirectRLEnvWarp)
    assert env.drone.__backend_name__ == "newton"
    assert env.num_envs == 4 and len(env.motion.position) == 8
    assert env.cfg.sim_to_real.enabled and not env.cfg.evaluation and env.cfg.training_stage is None
    diffnav = env.cfg.method == "diffnav"
    summary = {"runtime": runtime_metadata(env), "checks": {}, "policy_steps": 0}
    initial_observation, _ = env.reset()
    assert initial_observation["policy"].shape == (4, 202)
    assert bool(torch.isfinite(initial_observation["policy"]).all())
    torch.testing.assert_close(env.torch_obs_buf, initial_observation["policy"], rtol=0, atol=0)

    def dynamics():
        return {
            "mass": env.drone.data.body_mass.torch.clone(),
            "inertia": env.drone.data.body_inertia.torch.clone(),
            "velocity_gain": env.controller.velocity_gain.clone(),
            "KF": env.controller.rotor_parameters["KF"].clone(),
            "KM": env.controller.rotor_parameters["KM"].clone(),
            "control_scale": env.perturbation.control_gain_scale.clone(),
            "thrust_scale": env.perturbation.thrust_scale.clone(),
        }

    def check_randomization():
        values = dynamics()
        ranges = {}
        for name, nominal, configured in (
            ("mass", env.nominal_body_mass, env.cfg.sim_to_real.mass_scale_range),
            ("inertia", env.nominal_body_inertia, env.cfg.sim_to_real.inertia_scale_range),
        ):
            mask = nominal.abs() > 1.0e-12
            ratio = values[name][mask] / nominal[mask]
            assert bool(torch.isfinite(ratio).all())
            assert bool(((ratio >= configured[0] - 2.0e-5) & (ratio <= configured[1] + 2.0e-5)).all()), name
            assert bool((ratio - 1.0).abs().gt(1.0e-5).any()), f"{name} remained nominal"
            ranges[name] = [float(ratio.min()), float(ratio.max())]
        torch.testing.assert_close(
            values["velocity_gain"],
            env.controller.nominal_velocity_gain * env.perturbation.control_gain_scale,
        )
        for name in ("KF", "KM"):
            torch.testing.assert_close(
                values[name],
                getattr(env.controller.rotor_group, name) * env.perturbation.thrust_scale,
            )
        for name in ("control_scale", "thrust_scale"):
            assert bool((values[name] - 1.0).abs().gt(1.0e-5).any()), f"{name} remained nominal"
        if diffnav:
            for key, value in (
                ("mass_scale", env.mass_scale),
                ("inertia_scale", env.inertia_scale),
                ("thrust_scale", env.perturbation.thrust_scale),
                ("control_gain_scale", env.perturbation.control_gain_scale),
            ):
                torch.testing.assert_close(env.diffnav_parameters[key], value)
        return ranges

    summary["randomized_scale_ranges"] = check_randomization()
    retained = []
    last_command = torch.zeros((4, 3), device=env.device)
    previous_payload = None
    previous_payload_episode = None
    interval_updates = 0
    delayed_samples = 0
    expected_reward = None
    original_get_rewards = env._get_rewards

    def capture_reward():
        nonlocal expected_reward
        result = original_get_rewards()
        # Observe the existing boundary before later observation graphs run.
        expected_reward = result.detach().clone()
        return result

    env._get_rewards = capture_reward

    def read_delayed_observation():
        nonlocal previous_payload, previous_payload_episode, delayed_samples
        if diffnav:
            observation = env.get_diffnav_observation()
            for name in ("depth_features", "policy_depth_features"):
                assert observation[name].shape == (4, 1, 12, 16)
                assert bool(torch.isfinite(observation[name]).all()), name
            assert bool(((observation["depth_known_fraction"] >= 0) & (observation["depth_known_fraction"] <= 1)).all())
            pushes = env.perturbation.observation_buffer.num_pushes.clone()
            again = env.get_diffnav_observation()
            torch.testing.assert_close(env.perturbation.observation_buffer.num_pushes, pushes, rtol=0, atol=0)
            torch.testing.assert_close(again["depth_age_s"], observation["depth_age_s"], rtol=0, atol=0)
        perturbation = env.perturbation
        assert bool((perturbation.observation_buffer.time_lags == 1).all())
        expected_age = (perturbation.observation_buffer.num_pushes - 1).clamp(0, 1).float() * env.step_dt
        torch.testing.assert_close(perturbation.observation_age_s, expected_age, rtol=0, atol=1.0e-6)
        if previous_payload is not None:
            same_episode = env.episode_id == previous_payload_episode
            delayed = same_episode & (perturbation.observation_age_s > 0)
            if bool(delayed.any()):
                # Compare raw delivered storage; depth reprojection intentionally
                # changes the separately returned compensated policy features.
                torch.testing.assert_close(
                    perturbation._observation[delayed],
                    previous_payload[delayed],
                    rtol=0,
                    atol=0,
                )
                delayed_samples += int(delayed.sum())
        previous_payload = perturbation.latest_observation.clone()
        previous_payload_episode = env.episode_id.clone()

    read_delayed_observation()

    def step(action, *, manual_reset=False):
        nonlocal expected_reward, retained, interval_updates
        expected_reward = None
        ids_before = env.episode_id.clone()
        dg_before = env.diffnav_parameters["dg"].clone() if diffnav else None
        lags = env.perturbation.action_buffer.time_lags.clone()
        assert bool((lags == 1).all())
        ready = env.perturbation.action_buffer.num_pushes >= 1
        expected_action = torch.where(ready[:, None], last_command, torch.zeros_like(last_command))
        result = env.step(action)
        observation, reward, terminated, timeout, info = result
        assert observation["policy"].shape == (4, 202)
        for name, value in (("policy", observation["policy"]), ("reward", reward), ("root", env.root_state())):
            assert bool(torch.isfinite(value).all()), name
        assert expected_reward is not None
        torch.testing.assert_close(reward, expected_reward, rtol=0, atol=0)
        transition = info["transition"]
        for key in (
            "position",
            "quaternion",
            "velocity",
            "angular_velocity",
            "target",
            "episode_id",
            "step",
            "time_s",
            "reached",
            "collision",
            "bounds",
            "invalid",
            "timeout",
            "terminated",
            "executed_action",
            "action_delay_steps",
            "episode_return",
        ):
            assert key in transition, key
            assert len(transition[key]) == 4, key
            assert bool(torch.isfinite(transition[key]).all()), key
        torch.testing.assert_close(transition["episode_id"], ids_before, rtol=0, atol=0)
        torch.testing.assert_close(transition["executed_action"], expected_action, rtol=0, atol=0)
        torch.testing.assert_close(transition["action_delay_steps"], lags, rtol=0, atol=0)
        # UI resets mark the returned done flags after the physical transition
        # snapshot, so that snapshot intentionally keeps its original flags.
        if not manual_reset:
            torch.testing.assert_close(transition["terminated"], terminated, rtol=0, atol=0)
        torch.testing.assert_close(transition["timeout"], timeout, rtol=0, atol=0)
        if diffnav:
            for key in (
                "first_event_code",
                "first_event_time_s",
                "collision_position",
                "collision_velocity",
                "collision_substep",
                "depth_features",
                "critic_parameters",
            ):
                assert key in transition, key
            assert transition["depth_features"].shape == (4, 1, 12, 16)
            assert bool(torch.isfinite(transition["depth_features"]).all())
            stable = env.episode_id == ids_before
            if bool(stable.any()) and bool((env.diffnav_parameters["dg"][stable] != dg_before[stable]).any()):
                interval_updates += 1
            if env.cfg.variant == "dcol":
                trace = transition["dynamic_positions_by_substep"]
                assert trace.shape == (env.cfg.decimation, 8, 3)
                assert bool(torch.isfinite(trace).all())
                torch.testing.assert_close(trace[-1], env.motion.position, rtol=0, atol=0)
        done = terminated | timeout
        # With two reset batches, the final log belongs to the second batch.
        if bool(done.any()) and not manual_reset:
            for key, values in (
                ("Episode/return", transition["episode_return"]),
                ("Episode/length", transition["step"]),
                ("Episode/reach_goal", transition["reached"]),
                ("Episode/collision", transition["collision"]),
            ):
                torch.testing.assert_close(info["log"][key], values[done].float().mean())
        for reference, snapshot in retained:
            torch.testing.assert_close(reference, snapshot, rtol=0, atol=0)
        retained = [(observation["policy"], observation["policy"].clone())]
        for value in transition.values():
            values = value.values() if isinstance(value, dict) else (value,)
            retained.extend((item, item.clone()) for item in values if isinstance(item, torch.Tensor))
        last_command.copy_(action)
        read_delayed_observation()
        summary["policy_steps"] += 1
        assert not env._stepping, "step() leaked its in-step reset/logging state"
        return result

    try:
        initial_motion = env.motion.position.clone()
        for index in range(steps):
            command = torch.zeros((4, 3), device=env.device)
            command[:, 0] = 0.1 if index % 2 else -0.1
            command[:, 1] = 0.03 * (index % 3)
            step(command)
        assert delayed_samples > 0, "No one-tick observation/depth delivery was exercised"
        summary["checks"]["delays"] = {"ticks": 1, "verified_samples": delayed_samples}
        if diffnav:
            assert interval_updates > 0, "Active DiffNav disturbance never updated within an unchanged episode"
            summary["checks"]["interval_disturbance"] = {"updated_steps": interval_updates}
        else:
            summary["checks"]["interval_disturbance"] = "not_configured_for_aerialnav"
        assert not torch.equal(initial_motion, env.motion.position), "Shared obstacle process did not advance"

        # Consecutive manual partial resets also exercise the _stepping lifetime.
        for selected in (1, 2):
            reset_ids = torch.tensor([selected], dtype=torch.int32, device=env.device)
            kept = torch.arange(4, device=env.device) != selected
            before_ids, before_state = env.episode_id.clone(), env.root_state().clone()
            before = dynamics()
            world = {name: getattr(env.motion, name).clone() for name in ("position", "velocity", "goal")}
            world_step = env.motion.step_count
            parameters = {key: value.clone() for key, value in env.diffnav_parameters.items()} if diffnav else {}
            env._reset_idx(reset_ids)
            expected_ids = before_ids.clone()
            expected_ids[selected] += 1
            torch.testing.assert_close(env.episode_id, expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(env.root_state()[kept], before_state[kept], rtol=0, atol=1.0e-7)
            assert env.motion.step_count == world_step
            for name, value in world.items():
                torch.testing.assert_close(getattr(env.motion, name), value, rtol=0, atol=0)
            after = dynamics()
            for name in before:
                torch.testing.assert_close(after[name][kept], before[name][kept], rtol=0, atol=0)
            for name in ("mass", "inertia", "control_scale", "thrust_scale"):
                assert not torch.equal(before[name][selected], after[name][selected]), (
                    f"Partial reset did not resample {name}"
                )
            for name, value in parameters.items():
                torch.testing.assert_close(env.diffnav_parameters[name][kept], value[kept], rtol=0, atol=0)
        check_randomization()
        summary["checks"]["partial_reset"] = {"selected": [1, 2], "shared_world_preserved": True}

        env.reset()
        # Keep the forced timeout independent of a coincidental nearby goal.
        env.target_pos[0] = env.root_state()[0, :3] + env.target_pos.new_tensor((10.0, 0.0, 0.0))
        env.target_dir[0] = env.target_pos[0] - env.root_state()[0, :3]
        target_before = env.target_pos.clone()
        ids_before = env.episode_id.clone()
        env.episode_length_buf[0] = env.max_episode_length - 1
        _, _, terminated, timeout, info = step(torch.zeros_like(last_command))
        transition = info["transition"]
        assert bool(timeout[0]) and not bool(terminated[0])
        assert int(env.episode_id[0]) == int(ids_before[0]) + 1
        assert int(transition["episode_id"][0]) == int(ids_before[0])
        assert int(transition["step"][0]) == env.max_episode_length
        assert int(env.episode_length_buf[0]) == 0
        torch.testing.assert_close(transition["target"], target_before, rtol=0, atol=0)
        assert not torch.equal(transition["target"][0], env.target_pos[0])
        if diffnav:
            assert int(transition["first_event_code"][0]) == 5
        summary["checks"]["timeout_autoreset"] = "snapshot_and_episode_logs_preserved"

        env.reset()
        ids_before = env.episode_id.clone()
        env.episode_length_buf[0] = env.max_episode_length - 1
        original_consume_reset = env.sim.consume_reset_request
        request_count = 0

        def consume_once():
            nonlocal request_count
            request_count += 1
            return request_count == 1

        env.sim.consume_reset_request = consume_once
        try:
            observation, _, terminated, timeout, info = step(torch.zeros_like(last_command), manual_reset=True)
        finally:
            env.sim.consume_reset_request = original_consume_reset
        assert request_count == 1
        assert bool(timeout[0]) and not bool(timeout[1:].any())
        assert bool((terminated | timeout).all())
        torch.testing.assert_close(env.episode_id, ids_before + 1, rtol=0, atol=0)
        torch.testing.assert_close(info["transition"]["episode_id"], ids_before, rtol=0, atol=0)
        assert bool((env.episode_length_buf == 0).all())
        torch.testing.assert_close(env.torch_obs_buf, observation["policy"], rtol=0, atol=0)
        summary["checks"]["mixed_timeout_and_ui_reset"] = "each_episode_reset_once_with_original_snapshot_ids"

        env.reset()
        ids_before = env.episode_id.clone()
        pose = env.root_state()[0:1, :7].clone()
        pose[:, 2] = 4.6
        reset_ids = torch.tensor([0], device=env.device, dtype=torch.int32)
        env.drone.write_root_pose_to_sim_index(root_pose=pose, env_ids=reset_ids)
        env.drone.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((1, 6), device=env.device), env_ids=reset_ids
        )
        _, _, terminated, timeout, info = step(torch.zeros_like(last_command))
        transition = info["transition"]
        continuous = diffnav and env.cfg.variant == "continuous"
        assert bool(transition["bounds"][0]) and not bool(transition["invalid"][0])
        assert not bool(timeout[0])
        assert bool(terminated[0]) == (not continuous)
        assert int(env.episode_id[0]) == int(ids_before[0]) + int(not continuous)
        assert float(transition["position"][0, 2]) > 4.0
        if diffnav:
            assert int(transition["first_event_code"][0]) == 3
        summary["checks"]["bounds"] = {"real_pose_z": 4.6, "terminates": not continuous}

        env.reset()
        event_episode = int(env.episode_id[0])
        original_events = env._compute_first_events

        def inject_invalid_detector():
            terms = original_events()
            if env._stepping and int(env.episode_id[0]) == event_episode:
                terms = dict(terms)
                terms["invalid"] = terms["invalid"].clone()
                terms["invalid"][0] = True
            return terms

        env._compute_first_events = inject_invalid_detector
        try:
            _, _, terminated, timeout, info = step(torch.zeros_like(last_command))
        finally:
            env._compute_first_events = original_events
        assert bool(terminated[0]) and not bool(timeout[0])
        assert bool(info["transition"]["invalid"][0])
        assert int(env.episode_id[0]) == event_episode + 1
        if diffnav:
            assert int(info["transition"]["first_event_code"][0]) == 4
        summary["checks"]["invalid"] = "detector_injection_terminates_with_finite_simulator_state"
        assert env._graph_cache._graphs, "No environment Warp graph was captured"
        assert env.controller.graphs._graphs, "No control Warp graph was captured"
        summary["graphs"] = sorted(env._graph_cache._graphs)
        summary["controller_graphs"] = sorted(env.controller.graphs._graphs)
        summary["checks"]["retained_outputs"] = "policy_and_transition_snapshots_immutable"
        summary["checks"]["reward_copy"] = "preserved_before_observation_replay"
        return summary
    finally:
        env._get_rewards = original_get_rewards


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--method", choices=("ppo", "diffnav"), default="ppo")
    parser.add_argument("--variant", choices=("dcol", "step", "continuous"), default="dcol")
    parser.add_argument("--timing", choices=("baseline", "unified_20hz", "unified_50hz"), default="baseline")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    tokens = list(sys.argv[1:])
    if not any(token.startswith("physics=") for token in tokens):
        tokens.append("physics=newton_mjwarp")
    if not any(token == "--frontend" or token.startswith("--frontend=") for token in tokens):
        tokens.extend(("--frontend", "warp"))
    args = parse_physics_args(parser, tokens)
    if args.physics_preset != "newton_mjwarp" or args.frontend != "warp":
        parser.error("This contract requires physics=newton_mjwarp --frontend warp")
    if args.steps < 4:
        parser.error("At least four steps are needed to exercise graph replay, interval events, and delays")

    import aerialnav.tasks  # noqa: F401
    from aerialnav.tasks.navigation.config.hummingbird.env_cfg import DiffNavEnvCfg, NavigationEnvCfg

    cfg = (DiffNavEnvCfg if args.method == "diffnav" else NavigationEnvCfg)()
    cfg.scene.num_envs = 4
    cfg.num_static_obstacles, cfg.num_dynamic_obstacles = 300, 8
    cfg.depth_camera_batch_size = 4
    cfg.training_curriculum, cfg.training_stage, cfg.evaluation = False, None, False
    cfg.sim_to_real.enabled = True
    cfg.seed, cfg.sim.device = args.seed, args.device
    if args.method == "diffnav":
        cfg.variant, cfg.timing = args.variant, args.timing
        cfg.capture_terminal_observation = True
    select_physics(cfg, args.physics_preset)
    cfg.configure_scene()
    policy_dt = cfg.sim.dt * cfg.decimation
    cfg.sim_to_real.action_delay_range_s = (policy_dt, policy_dt)
    cfg.sim_to_real.input_delay_range_s = (policy_dt, policy_dt)
    cfg.episode_length_s = max(4.0, (args.steps + 4) * policy_dt)
    suffix = args.method + (f"_{args.variant}" if args.method == "diffnav" else "")
    output = args.output or Path(f"logs/warp_environment_contract/{suffix}.json")
    result = {
        "method": args.method,
        "variant": args.variant if args.method == "diffnav" else None,
        "physics": args.physics_preset,
        "frontend": args.frontend,
        "num_envs": 4,
        "static_obstacles": 300,
        "dynamic_obstacles": 8,
        "seed": args.seed,
    }
    task = "Isaac-AerialNav-Static-Depth-v0" if args.method == "diffnav" else "Isaac-AerialNav-Static-PPO-v0"
    try:
        with launch_simulation(cfg, args):
            env = create_navigation_env(task, cfg, args)
            try:
                result.update(run_contract(env, args.steps))
            finally:
                env.close()
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
