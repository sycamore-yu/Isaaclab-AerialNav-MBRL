"""Repeated upstream PPO updates must retain optimizer and logger state."""

from importlib.metadata import version

import torch
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
from tensordict import TensorDict

from aerialnav.agents.continuous_runner import ContinuousNavigationRunner
from aerialnav.agents.rsl_rl_cfg import NavigationPpoRunnerCfg


class VectorEnv:
    num_envs = 16
    num_actions = 3
    device = "cpu"
    cfg = {}
    max_episode_length = 8

    def __init__(self):
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.steps = 0

    def get_observations(self):
        return TensorDict({"policy": torch.ones(self.num_envs, 202)}, batch_size=[self.num_envs])

    def step(self, actions):
        self.steps += 1
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= self.max_episode_length
        self.episode_length_buf[dones] = 0
        return self.get_observations(), -actions.square().sum(-1), dones, {}


def test_two_upstream_updates_keep_parameters_optimizer_and_writer(tmp_path):
    torch.set_num_threads(1)
    env = VectorEnv()
    cfg = handle_deprecated_rsl_rl_cfg(NavigationPpoRunnerCfg(), version("rsl-rl-lib")).to_dict()
    cfg["num_steps_per_env"] = 2
    cfg["algorithm"].update(num_mini_batches=1, num_learning_epochs=1)
    runner = ContinuousNavigationRunner(env, cfg, str(tmp_path), "cpu")
    optimizer = runner.alg.optimizer
    initial = next(runner.alg.get_policy().parameters()).detach().clone()
    runner.current_learning_iteration = 0
    runner.learn(1, False)
    writer = runner.logger.writer
    runner.current_learning_iteration = 1
    runner.learn(1, False)
    assert runner.logger.writer is writer
    assert runner.alg.optimizer is optimizer
    assert env.steps == 4
    assert not torch.equal(initial, next(runner.alg.get_policy().parameters()))
    assert all(int(value["step"]) == 2 for value in optimizer.state.values())
    runner.save(str(tmp_path / "checkpoint_best_100.pt"), {"fresh": True})
    assert (tmp_path / "checkpoint_best_100.pt").exists()
    assert not list(tmp_path.glob("model_*.pt"))
    writer.close()
