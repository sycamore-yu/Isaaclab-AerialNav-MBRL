"""Differential checks against the original tracked Navigation source (CPU only)."""

from __future__ import annotations

import ast
import copy
import functools
import os.path as osp
import subprocess

import numpy as np
import pytest
import torch
import trimesh
import yaml
from reference_repo import UPSTREAM_NAVIGATION
from torch import nn

from aerialnav.control import GRAVITY, MASS, ROTOR_CONFIG, LeeRotorController, rotor_allocation
from aerialnav.terrain import NavigationObstaclesTerrainCfg, discrete_obstacles_terrain, obstacle_height_field

pytestmark = pytest.mark.skipif(not (UPSTREAM_NAVIGATION / ".git").exists(), reason="Set UPSTREAM_NAVIGATION_SOURCE to a pinned upstream checkout")

OMNI = "isaac-training/third_party/OmniDrones/omni_drones"
TERRAIN = "isaac-training/third_party/orbit/source/extensions/omni.isaac.orbit/omni/isaac/orbit/terrains"


def source(path):
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=UPSTREAM_NAVIGATION, text=True)


def load_definitions(path, names, namespace, strip_decorators=False):
    tree = ast.parse(source(path))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            if strip_decorators:
                node.decorator_list = []
            nodes.append(node)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)


def original_controller():
    namespace = {
        "torch": torch,
        "nn": nn,
        "yaml": yaml,
        "osp": osp,
        "__file__": str(UPSTREAM_NAVIGATION / OMNI / "controllers/lee_position_controller.py"),
    }
    load_definitions(
        f"{OMNI}/utils/torch.py",
        {
            "normalize",
            "quaternion_to_rotation_matrix",
            "quaternion_to_euler",
            "quat_rotate_inverse",
        },
        namespace,
        strip_decorators=True,
    )
    load_definitions(
        f"{OMNI}/controllers/lee_position_controller.py",
        {
            "compute_parameters",
            "LeePositionController",
        },
        namespace,
    )
    params = yaml.safe_load(source(f"{OMNI}/robots/assets/usd/hummingbird.yaml"))
    return namespace["LeePositionController"](GRAVITY, params)


@pytest.mark.parametrize("explicit_yaw", [False, True])
def test_lee_commands_match_original_for_random_states(explicit_yaw):
    torch.manual_seed(628)
    count = 32
    state = torch.randn(count, 13)
    state[:, 3:7] = torch.nn.functional.normalize(state[:, 3:7], dim=-1)
    velocity = torch.randn(count, 3) * 2
    acceleration = torch.randn(count, 3) * 3
    yaw = torch.randn(count, 1) if explicit_yaw else None
    original_state = torch.cat((state[:, :3], state[:, 6:7], state[:, 3:6], state[:, 7:]), -1)
    original = original_controller()
    expected = original(original_state, target_vel=velocity, target_acc=acceleration, target_yaw=yaw)
    actual = LeeRotorController(count, "cpu").motor_commands(state, velocity, acceleration, yaw)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_original_motor_state_and_reset_match_for_250_steps():
    namespace = {"torch": torch, "nn": nn}
    load_definitions(f"{OMNI}/actuators/rotor_group.py", {"RotorGroup"}, namespace)
    controller = LeeRotorController(8, "cpu", dt=0.016)
    serial = [namespace["RotorGroup"](ROTOR_CONFIG, 0.016) for _ in range(8)]
    hover = torch.sqrt(MASS * GRAVITY / 4 / controller.max_thrusts)
    for instance in serial:
        instance.throttle.copy_(hover)
    torch.manual_seed(91)
    for step in range(250):
        commands = torch.rand(8, 4) * 2.4 - 1.2
        thrust, moment = controller._motor_step(controller.rotor_parameters, commands)
        expected = [instance(command) for instance, command in zip(serial, commands, strict=True)]
        torch.testing.assert_close(thrust, torch.stack([item[0] for item in expected]), rtol=0, atol=0)
        torch.testing.assert_close(moment, torch.stack([item[1] for item in expected]), rtol=0, atol=0)
        torch.testing.assert_close(thrust @ rotor_allocation()[5], moment.sum(-1), rtol=1e-6, atol=1e-7)
        if step == 100:
            unchanged = controller.throttle[1:].clone()
            controller.reset(torch.tensor([0], dtype=torch.int32))
            serial[0].throttle.copy_(hover)
            torch.testing.assert_close(controller.throttle[1:], unchanged, rtol=0, atol=0)


def original_height_function():
    namespace = {"np": np}
    load_definitions(
        f"{TERRAIN}/height_field/hf_terrains.py", {"discrete_obstacles_terrain"}, namespace, strip_decorators=True
    )
    return namespace["discrete_obstacles_terrain"]


@pytest.mark.parametrize("seed", [0, 73])
def test_height_field_matches_original_sampling(seed):
    cfg = NavigationObstaclesTerrainCfg(size=(39.9, 39.9), horizontal_scale=0.1, vertical_scale=0.1)
    cfg.obstacle_height_mode = "range"
    original = original_height_function()
    np.random.seed(seed)
    expected = original(0.5, cfg)
    actual = obstacle_height_field(0.5, cfg, np.random.RandomState(seed))
    np.testing.assert_array_equal(actual, expected)


def test_mesh_and_origin_match_original_seed_zero():
    namespace = {"np": np, "trimesh": trimesh, "copy": copy, "functools": functools}
    load_definitions(
        f"{TERRAIN}/height_field/utils.py",
        {
            "height_field_to_mesh",
            "convert_height_field_to_mesh",
        },
        namespace,
    )
    original_mesh = namespace["height_field_to_mesh"](original_height_function())
    cfg = NavigationObstaclesTerrainCfg(
        size=(40.0, 40.0),
        horizontal_scale=0.1,
        vertical_scale=0.1,
        slope_threshold=0.75,
        seed=0,
    )
    cfg.obstacle_height_mode = "range"
    np.random.seed(0)
    np.random.choice(1, p=[1.0])
    difficulty = np.random.uniform(0.0, 1.0)
    expected_mesh, expected_origin = original_mesh(difficulty, copy.deepcopy(cfg))
    actual_mesh, actual_origin = discrete_obstacles_terrain(difficulty, copy.deepcopy(cfg))
    np.testing.assert_array_equal(actual_mesh[0].vertices, expected_mesh[0].vertices)
    np.testing.assert_array_equal(actual_mesh[0].faces, expected_mesh[0].faces)
    np.testing.assert_allclose(actual_origin, expected_origin, rtol=0, atol=1e-12)
