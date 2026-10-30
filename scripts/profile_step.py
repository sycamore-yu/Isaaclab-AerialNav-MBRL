"""Measure real environment phases and an uninstrumented step baseline.

The phase run synchronizes CUDA around each measured method. These are
instrumented wall times, not training throughput or isolated GPU kernel times.
Both runs include Zhang depth sampling and use zero commands without a policy
network, backward pass, or optimizer. Production source is not modified.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from contextlib import contextmanager
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


@contextmanager
def instrument_methods(methods, synchronize):
    """Temporarily time bound methods and restore the exact instance attributes."""
    totals = {label: {"calls": 0, "wall_s": 0.0} for label, _, _ in methods}
    restore = []

    def wrapper(original, label):
        @functools.wraps(original)
        def timed(*args, **kwargs):
            synchronize()
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                synchronize()
                totals[label]["wall_s"] += time.perf_counter() - started
                totals[label]["calls"] += 1

        return timed

    try:
        for label, target, name in methods:
            original = getattr(target, name)
            had_instance_attribute = name in vars(target)
            restore.append((target, name, original, had_instance_attribute))
            setattr(target, name, wrapper(original, label))
        yield totals
    finally:
        for target, name, original, had_instance_attribute in reversed(restore):
            if had_instance_attribute:
                setattr(target, name, original)
            else:
                delattr(target, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--method", choices=("navrl", "zhang"), default="navrl")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measured", type=int, default=10)
    parser.add_argument("--camera_batch_size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("logs/profile/step.json"))
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()
    if args.num_envs < 1 or args.warmup < 0 or args.measured < 1 or args.camera_batch_size < 1:
        parser.error("num_envs, measured and camera_batch_size must be positive; warmup must be nonnegative")

    import gymnasium as gym
    import torch

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import NavRLEnvCfg, ZhangNavRLEnvCfg

    cfg = (ZhangNavRLEnvCfg if args.method == "zhang" else NavRLEnvCfg)()
    cfg.training_curriculum = False
    cfg.num_dynamic_obstacles = 100
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.configure_scene()
    if args.method == "zhang":
        cfg.scene.depth_camera.batch_size = args.camera_batch_size
    task = "Isaac-NavRL-Zhang-Hummingbird-Direct-v0" if args.method == "zhang" else "Isaac-NavRL-Hummingbird-Direct-v0"
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset()
            command = torch.zeros(args.num_envs, 3, device=env.device)

            def synchronize():
                if torch.device(env.device).type == "cuda":
                    torch.cuda.synchronize(env.device)

            def step():
                env.step(command)
                if args.method == "zhang":
                    env.get_zhang_observation()

            def measure_steps():
                synchronize()
                started = time.perf_counter()
                for _ in range(args.measured):
                    step()
                synchronize()
                return time.perf_counter() - started

            with torch.no_grad():
                for _ in range(args.warmup):
                    step()
                baseline_s = measure_steps()
                methods = [
                    ("controller.compute", env.controller, "compute"),
                    ("controller.apply", env.controller, "apply"),
                    ("env.compute_terms", env, "_compute_terms"),
                    ("env.dynamic_obstacles", env, "_update_dynamic_obstacles"),
                    ("sim.step", env.sim, "step"),
                    ("scene.write_data_to_sim", env.scene, "write_data_to_sim"),
                    ("scene.update", env.scene, "update"),
                ]
                if args.method == "zhang":
                    methods.append(("env.first_events", env, "_compute_first_events"))
                    methods.append(("depth.sample", env, "get_zhang_observation"))
                with instrument_methods(methods, synchronize) as totals:
                    instrumented_s = measure_steps()

            phases = {}
            for label, values in totals.items():
                calls, wall_s = values["calls"], values["wall_s"]
                phases[label] = {
                    "calls": calls,
                    "wall_s": wall_s,
                    "mean_ms_per_call": wall_s * 1000 / calls if calls else None,
                    "ms_per_policy_step": wall_s * 1000 / args.measured,
                }
            result = {
                "method": args.method,
                "num_envs": args.num_envs,
                "device": str(env.device),
                "warmup_policy_steps": args.warmup,
                "measured_policy_steps_per_run": args.measured,
                "physics_dt": env.physics_dt,
                "policy_dt": env.step_dt,
                "decimation": cfg.decimation,
                "camera_batch_size": args.camera_batch_size if args.method == "zhang" else None,
                "controller_capture_setup_wall_s": getattr(env.controller, "capture_setup_wall_s", 0.0),
                "static_obstacles": cfg.num_static_obstacles,
                "dynamic_obstacles": cfg.num_dynamic_obstacles,
                "workload": "zero command environment steps, including depth sampling for Zhang; no policy or BPTT",
                "runs": "warmup, uninstrumented baseline, then instrumented phases in the same shared world",
                "baseline": {
                    "wall_s": baseline_s,
                    "ms_per_policy_step": baseline_s * 1000 / args.measured,
                    "ms_per_physics_step": baseline_s * 1000 / (args.measured * cfg.decimation),
                    "environment_transitions_per_s": args.num_envs * args.measured / baseline_s,
                    "cuda_sync": "run boundaries only",
                },
                "instrumented": {
                    "wall_s": instrumented_s,
                    "ms_per_policy_step": instrumented_s * 1000 / args.measured,
                    "cuda_sync": "before and after each phase call; changes scheduling and is not real throughput",
                    "phase_times_are_inclusive": True,
                    "phases": phases,
                },
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print("STEP_PROFILE " + json.dumps(result, allow_nan=False), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
