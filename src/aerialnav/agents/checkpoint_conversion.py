"""One-time conversion of original Navigation TensorDict checkpoints."""

from __future__ import annotations

from importlib.metadata import version
from types import SimpleNamespace

import torch
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from .rsl_rl_cfg import NavigationPpoRunnerCfg


def convert_original_state_dict(original: dict[str, torch.Tensor]) -> dict:
    """Convert policy/value weights; original PPO optimizer state is not portable.

    The former normalized value head is folded into an unnormalized linear head,
    preserving its denormalized prediction. Converted checkpoints are explicitly
    marked non-fresh and are never loaded by the formal training entry point.
    """
    obs = TensorDict({"policy": torch.zeros(1, 202)}, batch_size=[1])
    cfg = handle_deprecated_rsl_rl_cfg(NavigationPpoRunnerCfg(), version("rsl-rl-lib")).to_dict()
    cfg["multi_gpu"] = None
    algorithm = PPO.construct_algorithm(obs, SimpleNamespace(num_envs=1, num_actions=3), cfg, "cpu")
    actor_state = {}
    prefixes = (
        ("cnns.lidar.", "feature_extractor.module.0.module."),
        ("cnns.dynamic.", "feature_extractor.module.1.module.1."),
        ("cnns.fusion.", "feature_extractor.module.3.module."),
        ("mlp.", "actor.module.0.module."),
    )
    consumed = set()
    for key, expected in algorithm.actor.state_dict().items():
        new_prefix, old_prefix = next(pair for pair in prefixes if key.startswith(pair[0]))
        old_key = old_prefix + key.removeprefix(new_prefix)
        tensor = original[old_key]
        if tensor.shape != expected.shape:
            raise ValueError(f"Unexpected shape for {old_key}: {tensor.shape}, expected {expected.shape}")
        actor_state[key] = tensor.detach().cpu().clone()
        consumed.add(old_key)
    algorithm.actor.load_state_dict(actor_state, strict=True)

    norm_keys = ["value_norm.running_mean", "value_norm.running_mean_sq", "value_norm.debiasing_term"]
    mean_sum, square_sum, weight = (original[key].detach().cpu() for key in norm_keys)
    mean = mean_sum / weight.clamp_min(1e-5)
    variance = (square_sum / weight.clamp_min(1e-5) - mean.square()).clamp_min(1e-2)
    scale = variance.sqrt()
    value_weight = original["critic.module.weight"].detach().cpu() * scale.reshape(-1, 1)
    value_bias = original["critic.module.bias"].detach().cpu() * scale + mean
    algorithm.critic.load_state_dict({"mlp.weight": value_weight, "mlp.bias": value_bias}, strict=True)
    consumed.update(norm_keys + ["critic.module.weight", "critic.module.bias", "gae.gamma", "gae.lmbda"])
    unexpected = set(original) - consumed
    if unexpected:
        raise ValueError(f"Unrecognized original checkpoint keys: {sorted(unexpected)}")
    for parameter in list(algorithm.actor.parameters()) + list(algorithm.critic.parameters()):
        if not torch.isfinite(parameter).all():
            raise ValueError("Original checkpoint contains non-finite model parameters")
    converted = algorithm.save()
    converted.update(
        iter=0,
        infos={
            "fresh": False,
            "converted_from_original_navigation": True,
            "optimizer_reset": True,
            "value_normalization_folded_into_head": True,
            "formal_acceptance_training_eligible": False,
        },
    )
    return converted
