"""Check shared perturbations against real PhysX execution and partial resets."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("navrl", "zhang"), required=True)
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()

    import gymnasium as gym
    import torch

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import NavRLEnvCfg, ZhangNavRLEnvCfg
    from isaaclab_navrl.tasks.navigation.randomization import DifferentiableDelay

    cfg = (ZhangNavRLEnvCfg if args.method == "zhang" else NavRLEnvCfg)()
    cfg.scene.num_envs = 16
    cfg.num_static_obstacles, cfg.num_dynamic_obstacles = 300, 60
    cfg.training_curriculum = False
    cfg.training_stage = "finetune_s1" if args.enabled else "pretrain_s1"
    cfg.sim_to_real.enabled = args.enabled
    cfg.sim.device = args.device
    cfg.configure_scene()
    task = "Isaac-NavRL-Zhang-Hummingbird-Direct-v0" if args.method == "zhang" else "Isaac-NavRL-Hummingbird-Direct-v0"
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset()
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
                expected = history.push(command, env.perturbation.action_buffer.time_lags.clone(), zero_until_ready=True)
                _, _, terminated, timeout, info = env.step(command.detach())
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
                    torch.testing.assert_close(first["policy_depth_features"], second["policy_depth_features"], rtol=0, atol=0)
                    torch.testing.assert_close(env.perturbation.observation_buffer.num_pushes, pushes)
                    torch.testing.assert_close(env.root_state(), clean, rtol=0, atol=0)
                    for key, source in (("mass_scale", env.mass_scale), ("inertia_scale", env.inertia_scale),
                                        ("thrust_scale", env.perturbation.thrust_scale),
                                        ("control_gain_scale", env.perturbation.control_gain_scale)):
                        torch.testing.assert_close(env.zhang_parameters[key], source)
                if step == 5:
                    env._reset_idx(torch.tensor([0], device=env.device, dtype=torch.int32))
                    mask = torch.zeros(16, dtype=torch.bool, device=env.device)
                    mask[0] = True
                    history.reset(mask)
                    env.perturbation.action_buffer.set_time_lag(2, [0])
                assert torch.isfinite(env.root_state()).all()
            result = {
                "method": args.method, "enabled": args.enabled, "num_envs": 16,
                "physics_steps": 12 * cfg.decimation, "policy_steps": 12,
                "nominal_dynamics": env.nominal_dynamics, "sim_to_real": asdict(cfg.sim_to_real),
                "mass_scale_min": float(mass_ratio.min()), "mass_scale_max": float(mass_ratio.max()),
                "inertia_scale_min": float(inertia_ratio.min()), "inertia_scale_max": float(inertia_ratio.max()),
                "executed_action_max_error": max_execution_error, "partial_reset_passed": True,
                "nonfinite_events": int(env.nonfinite_events),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("SIM_TO_REAL_CHECK " + json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
