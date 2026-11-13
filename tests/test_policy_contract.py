"""Policy behavior and official PPO checkpoint tests without a simulator."""

from __future__ import annotations

from importlib.metadata import version

import pytest
import torch
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab_navrl.agents.models import NavRLActor, NavRLCritic
from isaaclab_navrl.agents.rsl_rl_cfg import NavRLPpoRunnerCfg


def make_models(batch=8):
    observation = TensorDict({"policy": torch.randn(batch, 202)}, batch_size=[batch])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = NavRLActor(
        observation,
        groups,
        "actor",
        3,
        {"class_name": "rsl_rl.modules.distribution:BetaDistribution", "action_range": (-2.0, 2.0)},
    )
    critic = NavRLCritic(observation, groups, "critic", 1, cnns=actor.cnns)
    return observation, actor, critic


def test_shared_encoder_receives_value_gradient_once():
    obs, actor, critic = make_models()
    assert critic.encoder is actor.cnns
    parameters = list(actor.parameters()) + list(critic.parameters())
    assert len(parameters) == len({id(p) for p in parameters})
    critic(obs).square().mean().backward()
    assert all(p.grad is not None for p in actor.cnns.parameters())
    assert sum(float(p.grad.abs().sum()) for p in actor.cnns.parameters()) > 0
    assert all(p.grad is None for p in actor.mlp.parameters())


def test_beta_actions_and_likelihood_match_transformed_distribution():
    obs, actor, _ = make_models(128)
    sample = actor(obs, stochastic_output=True)
    assert torch.all((sample >= -2) & (sample <= 2))
    alpha, beta = actor.output_distribution_params
    expected = torch.distributions.Beta(alpha, beta)
    normalized = ((sample + 2) / 4).clamp(1e-6, 1 - 1e-6)
    torch.testing.assert_close(
        actor.get_output_log_prob(sample),
        (expected.log_prob(normalized) - torch.log(torch.tensor(4.0))).sum(-1),
    )
    torch.testing.assert_close(actor(obs), 4 * expected.mean - 2)
    torch.testing.assert_close(actor.get_kl_divergence((alpha, beta), (alpha, beta)), torch.zeros(128))


class TinyVecEnv:
    """Deterministic observations/rewards exercise the real PPO update and storage."""

    def __init__(self):
        self.num_envs = 8
        self.num_actions = 3
        self.device = "cpu"
        self.max_episode_length = 5
        self.episode_length_buf = torch.zeros(8, dtype=torch.long)
        self.cfg = {}
        self.observations = TensorDict({"policy": torch.randn(8, 202)}, batch_size=[8])

    def get_observations(self):
        return self.observations

    def step(self, actions):
        self.episode_length_buf += 1
        done = self.episode_length_buf >= self.max_episode_length
        self.episode_length_buf[done] = 0
        next_obs = self.observations["policy"].roll(1, dims=-1).clone()
        next_obs[:, :3] = actions
        self.observations = TensorDict({"policy": next_obs}, batch_size=[8])
        reward = -(actions - torch.tensor([0.4, -0.2, 0.1])).square().sum(-1)
        return self.observations, reward, done.long(), {"time_outs": done}


def test_official_ppo_updates_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(19)
    torch.set_num_threads(1)
    cfg = handle_deprecated_rsl_rl_cfg(NavRLPpoRunnerCfg(), version("rsl-rl-lib")).to_dict()
    cfg["num_steps_per_env"] = 4
    cfg["algorithm"]["num_learning_epochs"] = 2
    cfg["algorithm"]["num_mini_batches"] = 2
    runner = OnPolicyRunner(TinyVecEnv(), cfg, log_dir=None, device="cpu")
    before = {key: tensor.clone() for key, tensor in runner.alg.actor.state_dict().items()}
    runner.learn(2, init_at_random_ep_len=False)
    assert any(not torch.equal(before[key], tensor) for key, tensor in runner.alg.actor.state_dict().items())
    parameters = [p for group in runner.alg.optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert all(torch.isfinite(parameter).all() for parameter in parameters)
    probe = TensorDict({"policy": torch.randn(8, 202)}, batch_size=[8])
    expected_action = runner.get_inference_policy()(probe).detach()
    expected_value = runner.alg.critic(probe).detach()
    checkpoint = tmp_path / "checkpoint.pt"
    runner.save(str(checkpoint), infos={"fixture": True})
    loaded = OnPolicyRunner(
        TinyVecEnv(),
        handle_deprecated_rsl_rl_cfg(NavRLPpoRunnerCfg(), version("rsl-rl-lib")).to_dict(),
        log_dir=None,
        device="cpu",
    )
    assert loaded.load(str(checkpoint), map_location="cpu") == {"fixture": True}
    torch.testing.assert_close(loaded.get_inference_policy()(probe), expected_action, rtol=0, atol=0)
    torch.testing.assert_close(loaded.alg.critic(probe), expected_value, rtol=0, atol=0)
    assert loaded.alg.critic.encoder is loaded.alg.actor.cnns
    assert loaded.alg.optimizer.state_dict()["state"]


def test_encoder_feature_order_is_lidar_dynamic_state():
    obs, actor, _ = make_models()
    flat = obs["policy"]
    expected = actor.cnns.fusion(
        torch.cat(
            (
                actor.cnns.lidar(flat[:, 8:152].reshape(-1, 1, 36, 4)),
                actor.cnns.dynamic(flat[:, 152:202]),
                flat[:, :8],
            ),
            dim=-1,
        )
    )
    torch.testing.assert_close(actor.cnns(flat), expected, rtol=0, atol=0)


def test_public_distribution_overrides_cannot_silently_change_exported_action_units():
    observation = TensorDict({"policy": torch.zeros(2, 202)}, batch_size=[2])
    with pytest.raises(ValueError, match="fixed"):
        NavRLActor(
            observation,
            {"actor": ["policy"]},
            "actor",
            3,
            {"class_name": "rsl_rl.modules.distribution:BetaDistribution", "action_range": (-1.0, 1.0)},
        )
