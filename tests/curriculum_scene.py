"""Exercise all four curriculum stages inside one real shared PhysX scene.

Promotion uses labelled synthetic outcomes solely to test stage transitions.
This result is an integration check, not a policy success-rate measurement.
"""

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
    from isaaclab_navrl.tasks.navigation.task_logic import dynamic_layout

    cfg = ZhangNavRLEnvCfg()
    cfg.scene.num_envs = 4
    cfg.training_curriculum = True
    cfg.sim.device = args.device
    cfg.configure_scene()
    with launch_simulation(cfg, args):
        env = gym.make("Isaac-NavRL-Zhang-Hummingbird-Direct-v0", cfg=cfg).unwrapped
        try:
            env.reset()
            pool = env.dynamic_collection
            camera = env.scene["depth_camera"]
            stages = []
            for difficulty in (60, 80, 100, 120):
                assert env.curriculum.active_obstacles == difficulty
                assert env.dynamic_collection is pool and env.scene["depth_camera"] is camera
                assert len(env.motion.position) == difficulty
                _, sizes = dynamic_layout(difficulty, cfg.terrain_seed)
                torch.testing.assert_close(env.motion.size.cpu(), torch.from_numpy(sizes))
                pose = pool.data.body_pose_w.torch
                assert pose.shape == (1, 120, 7)
                active = env._active_obstacle_ids
                inactive = torch.ones(120, dtype=torch.bool, device=env.device)
                inactive[active] = False
                torch.testing.assert_close(pose[0, active, :3], env.motion.position)
                assert (pose[0, inactive, 2] == -1000).all()
                before = env.motion.position.clone()
                env._reset_idx(torch.tensor([0], dtype=torch.int32, device=env.device))
                torch.testing.assert_close(env.motion.position, before, rtol=0, atol=0)
                for _ in range(2):
                    observation, reward, _, _, _ = env.step(torch.zeros(4, 3, device=env.device))
                    assert torch.isfinite(observation["policy"]).all() and torch.isfinite(reward).all()
                    depth = env.get_zhang_observation()["depth_features"]
                    assert depth.shape == (4, 1, 12, 16) and torch.isfinite(depth).all()
                query = env.root_state()[:, None, :3].detach().requires_grad_()
                vectors = env.obstacle_signed_distance(query)
                vectors.square().sum().backward()
                assert torch.isfinite(query.grad).all()
                stages.append({"active_obstacles": difficulty, "allocated_obstacles": 120,
                               "depth_finite": True, "distance_gradient_finite": True})
                if difficulty != 120:
                    env.curriculum.counts.zero_()
                    env.curriculum.time_sum.zero_()
                    env.curriculum.return_sum.zero_()
                    env.curriculum.length_sum.zero_()
                    env.curriculum.reset_episodes()
                    env.curriculum.observe({
                        "first_event_code": torch.ones(4, dtype=torch.long, device=env.device),
                        "first_event_time_s": torch.ones(4, device=env.device),
                        "terminated": torch.ones(4, dtype=torch.bool, device=env.device),
                        "timeout": torch.zeros(4, dtype=torch.bool, device=env.device),
                        "episode_return": torch.zeros(4, device=env.device),
                        "step": torch.ones(4, dtype=torch.long, device=env.device),
                    })
                    env.curriculum.pop_report()
                    ids = env.episode_id.clone()
                    env.advance_curriculum()
                    assert torch.equal(env.episode_id, ids + 1)
            result = {"test": "curriculum_scene", "synthetic_promotion_outcomes": True,
                      "policy_acceptance_test": False, "status": "passed", "stages": stages}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("CURRICULUM_SCENE " + json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
