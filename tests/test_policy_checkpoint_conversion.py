"""Compare converted weights with the original released NavRL policy."""

from __future__ import annotations

import ast
import subprocess
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import pytest
import torch
from einops.layers.torch import Rearrange
from tensordict import TensorDict
from torch import nn

from isaaclab_navrl.agents.checkpoint_conversion import convert_original_state_dict
from isaaclab_navrl.agents.models import NavRLActor, NavRLCritic


def test_released_checkpoint_conversion_preserves_actions_and_denormalized_values():
    source = Path(__file__).resolve().parents[2] / "NavRL"
    checkpoint = source / "ros1/navigation_runner/scripts/ckpts/navrl_checkpoint.pt"
    if not checkpoint.exists():
        pytest.skip("The released upstream checkpoint is not present")

    def source_ast(name):
        text = subprocess.check_output(
            ["git", "show", f"HEAD:isaac-training/training/scripts/{name}"], cwd=source, text=True
        )
        return ast.parse(text)

    namespace = {
        "torch": torch, "nn": nn, "Rearrange": Rearrange, "Iterable": Iterable,
        "Union": Union, "self": SimpleNamespace(device="cpu"),
    }
    definitions = [
        node for node in source_ast("utils.py").body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in {"make_mlp", "BetaActor", "ValueNorm"}
    ]
    exec(compile(ast.Module(definitions, type_ignores=[]), "upstream-utils", "exec"), namespace)
    ppo = next(node for node in source_ast("ppo.py").body if isinstance(node, ast.ClassDef) and node.name == "PPO")
    init = next(node for node in ppo.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    definitions = [
        node for node in init.body if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"feature_extractor_network", "dynamic_obstacle_network"}
    ]
    exec(compile(ast.Module(definitions, type_ignores=[]), "upstream-ppo", "exec"), namespace)
    lidar = namespace["feature_extractor_network"]
    dynamic = namespace["dynamic_obstacle_network"]
    fusion = namespace["make_mlp"]([256, 256])
    head = namespace["BetaActor"](3)
    value_head = nn.LazyLinear(1)
    value_norm = namespace["ValueNorm"](1)
    torch.manual_seed(23)
    torch.set_num_threads(1)
    flat = torch.randn(32, 202)
    flat[:, 8:152] = torch.rand(32, 144) * 4
    lidar(flat[:, 8:152].reshape(-1, 1, 36, 4))
    dynamic(flat[:, 152:].reshape(-1, 1, 5, 10))
    fusion(torch.zeros(32, 200))
    head(torch.zeros(32, 256))
    value_head(torch.zeros(32, 256))
    original = torch.load(checkpoint, weights_only=True, map_location="cpu")

    def load_subtree(module, prefix):
        module.load_state_dict({key.removeprefix(prefix): value for key, value in original.items() if key.startswith(prefix)})

    load_subtree(lidar, "feature_extractor.module.0.module.")
    load_subtree(dynamic, "feature_extractor.module.1.module.")
    load_subtree(fusion, "feature_extractor.module.3.module.")
    load_subtree(head, "actor.module.0.module.")
    load_subtree(value_head, "critic.module.")
    load_subtree(value_norm, "value_norm.")
    features = fusion(torch.cat((
        lidar(flat[:, 8:152].reshape(-1, 1, 36, 4)), flat[:, :8],
        dynamic(flat[:, 152:].reshape(-1, 1, 5, 10)),
    ), dim=-1))
    alpha, beta = head(features)
    expected_actions = 4 * alpha / (alpha + beta) - 2
    expected_values = value_norm.denormalize(value_head(features))
    converted = convert_original_state_dict(original)
    observations = TensorDict({"policy": flat}, batch_size=[32])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = NavRLActor(observations, groups, "actor", 3)
    critic = NavRLCritic(observations, groups, "critic", 1, actor.cnns)
    actor.load_state_dict(converted["actor_state_dict"])
    critic.load_state_dict(converted["critic_state_dict"])
    torch.testing.assert_close(actor(observations), expected_actions, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(critic(observations), expected_values, rtol=1e-5, atol=1e-5)
    assert converted["infos"]["fresh"] is False
    assert converted["infos"]["formal_acceptance_training_eligible"] is False
    assert converted["optimizer_state_dict"]["state"] == {}
