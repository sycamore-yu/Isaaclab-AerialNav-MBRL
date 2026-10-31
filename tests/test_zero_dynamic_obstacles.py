"""Zero-dynamic S0 keeps static perception, navigation and geometric gradients."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

NAVIGATION = Path(__file__).resolve().parents[1] / "src/aerialnav/tasks/navigation"


def isolated_methods(filename, class_name, names, namespace=None):
    """Exercise actual dependency-free branches without importing Isaac/Kit."""
    tree = ast.parse((NAVIGATION / filename).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    functions = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=functions, type_ignores=[]), filename, "exec"), namespace)
    return {name: namespace[name] for name in names}


def test_static_only_camera_initializes_and_skips_dynamic_transform_fetch():
    methods = isolated_methods("sensor_mixins.py", "_BatchedDepthCameraMixin", ("_clear_shared_mesh_tracking",))
    methods.update(
        isolated_methods(
            "depth_camera.py", "BatchedDepthCamera", ("_initialize_shared_mesh_tracking", "_update_mesh_transforms")
        )
    )
    camera = SimpleNamespace(
        _mesh_views=[None, None],
        _raycast_targets_cfg=[
            SimpleNamespace(prim_expr="/World/ground", track_mesh_transforms=False),
            SimpleNamespace(prim_expr="/World/defaultGroundPlane", track_mesh_transforms=False),
        ],
        _num_meshes_per_env={"/World/ground": 1, "/World/defaultGroundPlane": 1},
    )
    for name, method in methods.items():
        setattr(camera, name, MethodType(method, camera))
    # No PhysX view creator, torch or Warp objects are available in this fixture.
    camera._initialize_shared_mesh_tracking()
    camera._update_mesh_transforms()
    assert camera._shared_mesh_tracking_ready
    assert camera._shared_mesh_view is None and camera._shared_mesh_count == 0
    assert camera.mesh_transform_info["body_count"] == 0
    assert camera.mesh_transform_info["mapping"] == []


def test_zero_dynamic_motion_and_navigation_keep_fixed_observation_shapes():
    import torch

    from aerialnav.tasks.navigation.curriculum import active_obstacle_indices
    from aerialnav.tasks.navigation.task_logic import DynamicObstacleMotion, dynamic_layout, navigation_terms

    position, size = dynamic_layout(0, 0)
    assert position.shape == size.shape == (0, 3)
    assert active_obstacle_indices(0, 0) == active_obstacle_indices(0, 120) == []
    motion = DynamicObstacleMotion(0, 0, "cpu", 0.01)
    for _ in range(20):
        motion.advance()
    assert motion.position.shape == motion.velocity.shape == motion.size.shape == (0, 3)
    drone = torch.tensor([[0.0, 0.0, 2.0], [1.0, 1.0, 2.0]])
    velocity = torch.zeros_like(drone)
    target = torch.tensor([[10.0, 0.0, 2.0], [10.0, 1.0, 2.0]])
    terms = navigation_terms(
        drone,
        velocity,
        target,
        target - drone,
        torch.full((2, 1, 36, 4), 4.0),
        motion,
        velocity,
        torch.tensor([[0.5, 2.5], [0.5, 2.5]]),
    )
    assert terms["dynamic_obstacle"].shape == (2, 1, 5, 10)
    torch.testing.assert_close(terms["dynamic_obstacle"], torch.zeros(2, 1, 5, 10))
    torch.testing.assert_close(terms["reward_dynamic"], torch.zeros(2))
    observation = torch.cat([terms[key].flatten(1) for key in ("state", "lidar", "dynamic_obstacle")], dim=-1)
    assert observation.shape == (2, 202) and torch.isfinite(observation).all()


def test_setting_zero_active_obstacles_does_not_require_a_dynamic_pose_buffer():
    import torch

    from aerialnav.tasks.navigation.curriculum import active_obstacle_indices
    from aerialnav.tasks.navigation.task_logic import DynamicObstacleMotion

    methods = isolated_methods(
        "navigation_env.py",
        "NavigationEnv",
        ("set_active_obstacles", "_write_dynamic_obstacles"),
        {
            "torch": torch,
            "active_obstacle_indices": active_obstacle_indices,
            "DynamicObstacleMotion": DynamicObstacleMotion,
        },
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            dynamic_capacity=lambda: 0,
            terrain_seed=0,
            method="diffnav",
            dynamic_speed_range=(0.5, 1.5),
            num_dynamic_obstacles=0,
        ),
        device="cpu",
        physics_dt=0.01,
        step_dt=0.1,
        dynamic_collection=None,
    )
    for name, method in methods.items():
        setattr(env, name, MethodType(method, env))
    env.set_active_obstacles(0)
    assert env._active_obstacle_ids.shape == (0,) and env.motion.position.shape == (0, 3)
    assert env.cfg.num_dynamic_obstacles == 0 and not hasattr(env, "_dynamic_pose")


@pytest.mark.parametrize("snapshots", [False, True])
def test_empty_dynamic_geometry_keeps_ground_and_static_gradients(snapshots):
    import torch

    from aerialnav.diffnav.dcol import hybrid_collision_gap

    query = torch.tensor([[[0.0, 0.0, 0.6], [0.0, 0.0, 1.2]], [[0.0, 0.0, 0.7], [0.0, 0.0, 1.3]]], requires_grad=True)
    static_closest = query.detach().clone()
    static_closest[..., 0] = torch.tensor([[10.0, 0.1], [10.0, 0.1]])
    positions = torch.empty(2, 1, 0, 3) if snapshots else torch.empty(0, 3)
    # Every synthetic point is outside its nearest vertical static surface,
    # so the explicit old vector norm equals the positive signed distance.
    static_signed_distance = (query - static_closest).norm(dim=-1)
    gap = hybrid_collision_gap(query, 0.3, static_signed_distance, positions, torch.empty(0, 3))
    torch.testing.assert_close(gap, torch.tensor([[0.29, -0.2], [0.39, -0.2]]))
    gap.sum().backward()
    torch.testing.assert_close(
        query.grad, torch.tensor([[[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]])
    )
