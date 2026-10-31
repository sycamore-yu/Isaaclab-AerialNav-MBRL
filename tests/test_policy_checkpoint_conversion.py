"""Compare converted PPO weights against the upstream released policy."""

from __future__ import annotations

import ast
import subprocess
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Union

import pytest
import torch
from einops.layers.torch import Rearrange
from reference_repo import UPSTREAM_NAVIGATION
from tensordict import TensorDict
from torch import nn

from aerialnav.agents.checkpoint_conversion import convert_original_state_dict
from aerialnav.agents.models import NavigationActor, NavigationCritic


def test_released_checkpoint_conversion_preserves_actions_and_denormalized_values():
    source = UPSTREAM_NAVIGATION
    checkpoint_dir = source / "ros1/navigation_runner/scripts/ckpts"
    candidates = sorted(checkpoint_dir.glob("*_checkpoint.pt"))
    if len(candidates) != 1:
        pytest.skip("Exactly one upstream released PPO checkpoint is required for parity")
    checkpoint = candidates[0]

    def source_ast(name):
        text = subprocess.check_output(
            ["git", "show", f"HEAD:isaac-training/training/scripts/{name}"], cwd=source, text=True
        )
        return ast.parse(text)

    namespace = {
        "torch": torch,
        "nn": nn,
        "Rearrange": Rearrange,
        "Iterable": Iterable,
        "Union": Union,
        "self": SimpleNamespace(device="cpu"),
    }
    definitions = [
        node
        for node in source_ast("utils.py").body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in {"make_mlp", "BetaActor", "ValueNorm"}
    ]
    exec(compile(ast.Module(definitions, type_ignores=[]), "upstream-utils", "exec"), namespace)
    ppo = next(node for node in source_ast("ppo.py").body if isinstance(node, ast.ClassDef) and node.name == "PPO")
    init = next(node for node in ppo.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    definitions = [
        node
        for node in init.body
        if isinstance(node, ast.Assign)
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
        module.load_state_dict(
            {key.removeprefix(prefix): value for key, value in original.items() if key.startswith(prefix)}
        )

    load_subtree(lidar, "feature_extractor.module.0.module.")
    load_subtree(dynamic, "feature_extractor.module.1.module.")
    load_subtree(fusion, "feature_extractor.module.3.module.")
    load_subtree(head, "actor.module.0.module.")
    load_subtree(value_head, "critic.module.")
    load_subtree(value_norm, "value_norm.")
    # Execute the vendored CatTensors constructor and forward: its default
    # sort=True, not the order of the call's in_keys list, defines checkpoint layout.
    transform_source = subprocess.check_output(
        ["git", "show", "HEAD:isaac-training/third_party/rl/torchrl/envs/transforms/transforms.py"],
        cwd=source,
        text=True,
    )
    transform_class = next(
        node
        for node in ast.parse(transform_source).body
        if isinstance(node, ast.ClassDef) and node.name == "CatTensors"
    )
    utils_source = subprocess.check_output(
        ["git", "show", "HEAD:isaac-training/third_party/rl/torchrl/envs/utils.py"], cwd=source, text=True
    )
    sort_function = next(
        node for node in ast.parse(utils_source).body if isinstance(node, ast.FunctionDef) and node.name == "_sort_keys"
    )

    class Transform(nn.Module):
        def __init__(self, in_keys, out_keys):
            super().__init__()
            self.in_keys, self.out_keys = in_keys, out_keys

    from tensordict import unravel_key

    transform_namespace = {"torch": torch, "Transform": Transform, "unravel_key": unravel_key}
    tree = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            sort_function,
            transform_class,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), "upstream-CatTensors", "exec"), transform_namespace)
    concatenate = transform_namespace["CatTensors"](
        ["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature"],
        "_feature",
        del_keys=False,
    )
    data = TensorDict(
        {
            "_cnn_feature": lidar(flat[:, 8:152].reshape(-1, 1, 36, 4)),
            ("agents", "observation", "state"): flat[:, :8],
            "_dynamic_obstacle_feature": dynamic(flat[:, 152:].reshape(-1, 1, 5, 10)),
        },
        batch_size=[32],
    )
    features = fusion(concatenate(data)["_feature"])
    alpha, beta = head(features)
    expected_actions = 4 * alpha / (alpha + beta) - 2
    expected_values = value_norm.denormalize(value_head(features))
    converted = convert_original_state_dict(original)
    observations = TensorDict({"policy": flat}, batch_size=[32])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = NavigationActor(
        observations,
        groups,
        "actor",
        3,
        {"class_name": "rsl_rl.modules.distribution:BetaDistribution", "action_range": (-2.0, 2.0)},
    )
    critic = NavigationCritic(observations, groups, "critic", 1, actor.cnns)
    actor.load_state_dict(converted["actor_state_dict"])
    critic.load_state_dict(converted["critic_state_dict"])
    torch.testing.assert_close(actor(observations), expected_actions, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(critic(observations), expected_values, rtol=1e-5, atol=1e-5)
    assert converted["infos"]["fresh"] is False
    assert converted["infos"]["formal_acceptance_training_eligible"] is False
    assert converted["optimizer_state_dict"]["state"] == {}
