"""Coincident clones must fly independently while seeing one shared world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/smoke/scene_isolation.json"))
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()
    import gymnasium as gym
    import torch

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import NavRLEnvCfg

    cfg = NavRLEnvCfg()
    cfg.training_curriculum = False
    cfg.num_dynamic_obstacles = 100
    cfg.scene.num_envs = 2
    cfg.sim.device = args.device
    cfg.configure_scene()
    with launch_simulation(cfg, args):
        env = gym.make("Isaac-NavRL-Hummingbird-Direct-v0", cfg=cfg).unwrapped
        try:
            env.reset()
            pose = torch.tensor([[-24.0, -24.0, 2.0, 0.0, 0.0, 0.0, 1.0]], device=env.device).repeat(2, 1)
            ids = torch.arange(2, device=env.device, dtype=torch.int32)
            env.drone.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            env.drone.write_root_velocity_to_sim_index(root_velocity=pose.new_zeros(2, 6), env_ids=ids)
            env.target_pos[:] = pose.new_tensor((24.0, -24.0, 2.0))
            env.target_dir[:] = pose.new_tensor((48.0, 0.0, 0.0))
            env.height_range[:] = 2.0
            env.previous_velocity.zero_()
            env.sim.forward()
            env.scene.update(0.0)
            max_difference = 0.0
            for _ in range(64):
                observation, _, terminated, timeout, _ = env.step(pose.new_zeros(2, 3))
                assert not (terminated | timeout).any(), "Coincident drones collided or lost stable flight."
                state = env.root_state()
                max_difference = max(max_difference, float((state[0] - state[1]).abs().max()))
                torch.testing.assert_close(state[0], state[1], atol=2e-5, rtol=1e-5)
                torch.testing.assert_close(observation["policy"][0], observation["policy"][1], atol=2e-5, rtol=1e-5)
            result = {
                "num_envs": 2,
                "steps": 64,
                "initial_positions_identical": True,
                "cross_environment_collisions_filtered": True,
                "same_world_observations_identical": True,
                "max_state_difference": max_difference,
                "shared_dynamic_collection_shape": list(env.dynamic_collection.data.body_pose_w.torch.shape),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("SCENE_ISOLATION " + json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
