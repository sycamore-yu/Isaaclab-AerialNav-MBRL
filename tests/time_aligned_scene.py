"""Bounded PhysX checks of the six comparison configurations, not convergence."""

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()
    import gymnasium as gym
    import torch

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import ZhangNavRLEnvCfg
    from isaaclab_navrl.tasks.navigation.curriculum import TrainingCurriculum
    from isaaclab_navrl.training import STAGE_CONFIG, write_json
    from isaaclab_navrl.zhang.cli import parse_train_args
    from isaaclab_navrl.zhang.config import ZhangTrainCfg
    from isaaclab_navrl.zhang.runner import ZhangTrainer

    plan = json.loads((Path(__file__).resolve().parents[1] / "configs/experiments/zhang_finetune_s1.json").read_text())
    cfg = ZhangNavRLEnvCfg()
    cfg.scene.num_envs = 32
    cfg.training_stage = "finetune_s1"
    cfg.training_curriculum = True
    cfg.sim_to_real.enabled = True
    cfg.num_static_obstacles = STAGE_CONFIG[cfg.training_stage]["static_obstacles"]
    cfg.num_dynamic_obstacles = STAGE_CONFIG[cfg.training_stage]["dynamic_obstacles"]
    cfg.capture_terminal_observation = True
    cfg.sim.device = args.device
    cfg.configure_scene()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"policy_acceptance": False, "num_envs": 32, "stage": cfg.training_stage, "runs": {}}
    with launch_simulation(cfg, args):
        env = gym.make("Isaac-NavRL-Zhang-Hummingbird-Direct-v0", cfg=cfg).unwrapped
        original_step = env.step
        latest = {}

        def step(command):
            result = original_step(command)
            latest.update(result[-1]["transition"])
            return result

        env.step = step
        try:
            for run in plan["runs"]:
                tokens = [
                    token
                    for key, value in run["train_args"].items()
                    for token in ("--" + key, str(int(value)) if isinstance(value, bool) else str(value))
                ]
                values = parse_train_args(tokens)
                values.pop("timing")
                values.update(
                    timesteps=min(values["timesteps"], 48), max_iterations=2, gradient_diagnostic_updates=(1,)
                )
                torch.manual_seed(plan["seed"])
                env.curriculum = TrainingCurriculum(32, env.device, stage=cfg.training_stage)
                (args.output / run["name"]).mkdir()
                trainer = ZhangTrainer(env, ZhangTrainCfg(**values), args.output / run["name"])
                original_gap = trainer._collision_gap
                checks = {"intermediate_samples": 0, "terminal_samples": 0, "maximum_motion_m": 0.0}

                def gap(
                    query, radius, positions, *, _gap=original_gap, _checks=checks, _dt=trainer.cfg.policy_dt, **kwargs
                ):
                    if query.ndim == 3:
                        times = torch.linspace(0, _dt, 10, device=env.device)[None, 1:-1, None, None]
                        expected = env.motion.position + env.motion.velocity * times
                        torch.testing.assert_close(positions, expected, rtol=0, atol=0)
                        _checks["maximum_motion_m"] = max(
                            _checks["maximum_motion_m"], float((positions - env.motion.position).norm(dim=-1).max())
                        )
                        _checks["intermediate_samples"] += query.shape[0] * query.shape[1]
                    else:
                        substep = latest["collision_substep"]
                        index = torch.where(
                            substep > 0, substep - 1, latest["dynamic_positions_by_substep"].shape[0] - 1
                        )
                        torch.testing.assert_close(
                            positions, latest["dynamic_positions_by_substep"][index], rtol=0, atol=0
                        )
                        _checks["terminal_samples"] += len(query)
                    return _gap(query, radius, positions, **kwargs)

                trainer._collision_gap = gap
                result = trainer.run()
                rows = [json.loads(line) for line in (trainer.log_dir / "updates.jsonl").read_text().splitlines()]
                assert result["completed_updates"] == 2
                assert result["nonfinite_grad_events"] == result["nonfinite_events"] == 0
                assert all(row["grad_norm"] > 0 and row["executed_action_max_error"] == 0 for row in rows)
                assert sum(row["velocity_valid_samples"] for row in rows) > 0
                assert checks["intermediate_samples"] == checks["terminal_samples"] * 8
                report["runs"][run["name"]] = {
                    "status": "passed",
                    "checks": checks,
                    "training": result,
                    "updates": rows,
                }
                write_json(args.output / "report.json", report)
                del trainer
        finally:
            env.close()


if __name__ == "__main__":
    main()
