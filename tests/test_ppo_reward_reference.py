"""Differential test against the unmodified upstream NavigationEnv method."""

import ast
import subprocess
from types import SimpleNamespace

import einops
import pytest
import torch
from reference_repo import UPSTREAM_NAVIGATION
from tensordict import TensorDict

from aerialnav.tasks.navigation.task_logic import DynamicObstacleMotion, navigation_terms


def test_original_observation_and_reward_on_the_same_states():
    upstream = UPSTREAM_NAVIGATION
    if not (upstream / ".git").exists():
        pytest.skip("Set UPSTREAM_NAVIGATION_SOURCE to a pinned upstream checkout")
    source = subprocess.check_output(
        ["git", "show", "HEAD:isaac-training/training/scripts/env.py"], cwd=upstream, text=True
    )
    helpers = subprocess.check_output(
        ["git", "show", "HEAD:isaac-training/training/scripts/utils.py"], cwd=upstream, text=True
    )
    tree = ast.parse(source)
    environment = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NavigationEnv")
    compute = next(
        node for node in environment.body if isinstance(node, ast.FunctionDef) and node.name == "_compute_state_and_obs"
    )
    helper = next(
        node
        for node in ast.parse(helpers).body
        if isinstance(node, ast.FunctionDef) and node.name == "vec_to_new_frame"
    )
    namespace = {"torch": torch, "einops": einops, "TensorDict": TensorDict}
    exec(compile(ast.Module(body=[helper, compute], type_ignores=[]), "upstream_aerialnav", "exec"), namespace)

    torch.manual_seed(71)
    count = 32
    position = torch.rand(count, 3) * torch.tensor([40.0, 40.0, 5.0]) - torch.tensor([20.0, 20.0, 0.0])
    velocity = torch.randn(count, 3)
    target = torch.randn(count, 3) * 20
    direction = torch.randn(count, 3)
    previous = torch.randn(count, 3)
    height = torch.tensor([[0.5, 2.5]]).repeat(count, 1)
    distances = 0.1 + 4.9 * torch.rand(count, 1, 36, 4)
    motion = DynamicObstacleMotion(100, 7, "cpu", 0.016)
    motion.advance()
    state = torch.zeros(count, 1, 13)
    state[:, 0, :3], state[:, 0, 7:10] = position, velocity
    state[:, 0, 3] = 1
    hits = torch.zeros(count, 144, 3)
    hits[..., 0] = distances.flatten(1)
    old = SimpleNamespace(
        num_envs=count,
        device="cpu",
        batch_size=[count],
        lidar_range=4.0,
        lidar_resolution=(36, 4),
        drone=SimpleNamespace(get_state=lambda **_: state, vel_w=state[..., 7:13], pos=state[..., :3]),
        lidar=SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits, pos_w=torch.zeros(count, 3))),
        target_pos=target[:, None],
        target_dir=direction[:, None],
        prev_drone_vel_w=previous[:, None],
        height_range=height[:, None],
        dyn_obs_state=torch.cat((motion.position, torch.zeros(100, 10)), dim=-1),
        dyn_obs_vel=motion.velocity,
        dyn_obs_size=motion.size,
        dyn_obs_width_res=0.25,
        max_obs_3d_height=1.0,
        progress_buf=torch.ones(count, dtype=torch.long),
        max_episode_length=2200,
        cfg=SimpleNamespace(
            device="cpu",
            sensor=SimpleNamespace(lidar_range=4.0),
            env_dyn=SimpleNamespace(num_obstacles=100),
            algo=SimpleNamespace(feature_extractor=SimpleNamespace(dyn_obs_num=5)),
        ),
        info=TensorDict({"drone_state": state.clone()}, [count]),
        stats=TensorDict(
            {name: torch.zeros(count, 1) for name in ("return", "episode_len", "reach_goal", "collision", "truncated")},
            [count],
        ),
        _should_render=lambda _: False,
    )
    reference = namespace["_compute_state_and_obs"](old)
    result = navigation_terms(position, velocity, target, direction, distances, motion, previous, height)
    for name in ("state", "lidar", "dynamic_obstacle"):
        torch.testing.assert_close(result[name], reference["agents", "observation", name], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result["reward"], old.reward[:, 0], rtol=1e-5, atol=1e-5)
    # Collision geometry is intentionally shared with the corrected DiffNav detector.
