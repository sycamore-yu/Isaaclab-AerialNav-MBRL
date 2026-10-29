"""Real PhysX SHAC terminal/timeout smoke; forced cases are not policy successes."""
import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--algorithm', choices=('shac', 'ahac'), default='shac')
    add_launcher_args(parser)
    parser.set_defaults(visualizer=['none'])
    args = parser.parse_args()
    import gymnasium as gym
    import torch

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import ZhangNavRLEnvCfg
    from isaaclab_navrl.zhang.config import ZhangTrainCfg
    from isaaclab_navrl.zhang.runner import ZhangTrainer

    cfg = ZhangNavRLEnvCfg()
    cfg.scene.num_envs = 16
    cfg.training_stage = 'pretrain_s1'
    cfg.num_static_obstacles, cfg.num_dynamic_obstacles = 300, 60
    cfg.capture_terminal_observation = True
    cfg.episode_length_s = .2
    cfg.sim.device = args.device
    cfg.configure_scene()
    result = dict(status='running', policy_acceptance=False)
    with launch_simulation(cfg, args):
        env = gym.make('Isaac-NavRL-Zhang-Hummingbird-Direct-v0', cfg=cfg).unwrapped
        try:
            reset = env.reset

            def injected_reset(*a, **kw):
                reset(*a, **kw)
                pose = env.root_state()[:2, :7].clone()
                pose[:, 2] = torch.tensor([4.3, .1], device=env.device)
                env.drone.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([0, 1], device=env.device, dtype=torch.int32))
                return env.get_zhang_observation(), {}

            env.reset = injected_reset
            trainer = ZhangTrainer(env, ZhangTrainCfg(algorithm=args.algorithm, timesteps=8, max_iterations=2,
                ahac_initial_horizon=8, ahac_min_horizon=4, ahac_max_horizon=16,
                critic_iterations=1, critic_max_iterations=1), args.output.parent/(args.output.stem+'_training'))
            d = trainer.run()
            rows = [json.loads(s) for s in (trainer.log_dir/'updates.jsonl').read_text().splitlines()]
            absorbing = sum(r['ActorCritic/absorbing_terminal_count'] for r in rows)
            timeouts = sum(r['ActorCritic/pre_reset_timeout_count'] for r in rows)
            assert absorbing > 0 and timeouts > 0
            assert all(r['ActorCritic/absorbing_bootstrap_abs_sum'] == 0 for r in rows)
            assert all(r['velocity_valid_samples'] == 16*r['rollout_steps'] for r in rows)
            assert d['nonfinite_grad_events'] == d['nonfinite_events'] == 0
            assert all(r['executed_action_max_error'] == 0 for r in rows)
            result.update(status='passed', updates=d['completed_updates'], environment_steps=d['environment_steps'],
                          algorithm=args.algorithm,
                          maximum_critic_optimizer_steps=max(r['ActorCritic/critic_optimizer_steps'] for r in rows),
                          absorbing_terminals=absorbing, pre_reset_timeouts=timeouts,
                          absorbing_bootstrap_abs_sum=0, velocity_loss_mode=trainer.cfg.resolved_velocity_loss_mode,
                          bootstrap_semantics=d['bootstrap_semantics'], nonfinite_grad_events=0)
        except BaseException as error:
            result.update(status='failed', error=repr(error))
            raise
        finally:
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            env.close()


if __name__ == '__main__':
    main()
