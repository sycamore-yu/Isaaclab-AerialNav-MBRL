"""Official Isaac Lab 3.0 configuration for the two NavRL training methods."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiMeshRayCasterCameraCfg, MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.utils import configclass
from isaaclab.utils.string import ResolvableString
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from isaaclab_navrl.control import make_drone_cfg
from isaaclab_navrl.terrain import build_terrain_cfg

from ...depth_camera_cfg import BatchedDepthCameraCfg
from ...task_logic import dynamic_layout


@configclass
class SharedNavigationSceneCfg(InteractiveSceneCfg):
    class_type = "isaaclab_navrl.tasks.navigation.scene:SharedNavigationScene"

    drone = make_drone_cfg("{ENV_REGEX_NS}/Hummingbird")
    terrain = build_terrain_cfg(num_envs=1024, num_obstacles=350, seed=0)
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(300.0, 300.0), color=(0.1, 0.1, 0.1)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.01)),
        collision_group=-1,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    lidar = RayCasterCfg(
        class_type="isaaclab_navrl.tasks.navigation.lidar:NavRLRayCaster",
        prim_path="{ENV_REGEX_NS}/Hummingbird/base_link",
        spawn=None,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.BpearlPatternCfg(horizontal_res=10.0, vertical_ray_angles=(-10.0, 0.0, 10.0, 20.0)),
        mesh_prim_paths=["/World/ground"],
        max_distance=4.0,
        update_period=0.016,
        debug_vis=False,
    )
    dynamic_obstacles: RigidObjectCollectionCfg | None = None
    depth_camera: MultiMeshRayCasterCameraCfg | None = None


def make_dynamic_cfg(count: int, seed: int) -> RigidObjectCollectionCfg | None:
    if count == 0:
        return None
    positions, sizes = dynamic_layout(count, seed)
    objects = {}
    for index, (position, size) in enumerate(zip(positions, sizes, strict=True)):
        properties = {
            "rigid_props": [UsdPhysicsRigidBodyCfg(kinematic_enabled=True), PhysxRigidBodyCfg(disable_gravity=True)],
            "collision_props": UsdPhysicsCollisionCfg(collision_enabled=False),
            "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        }
        if size[2] > 1:
            spawn = sim_utils.CylinderCfg(radius=float(size[0] / 2), height=5.0, **properties)
        else:
            spawn = sim_utils.CuboidCfg(size=tuple(float(x) for x in size), **properties)
        objects[f"obstacle_{index:03d}"] = RigidObjectCfg(
            prim_path=f"/World/Dynamic/obstacle_{index:03d}",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(x) for x in position)),
            collision_group=-1,
        )
    return RigidObjectCollectionCfg(rigid_objects=objects)


@configclass
class NavRLEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 35.2
    action_space = 3
    observation_space = 202
    state_space = 0
    seed = 0
    num_static_obstacles: int = 350
    num_dynamic_obstacles: int = 100
    terrain_seed: int = 0
    dynamic_speed_range: tuple[float, float] = (0.5, 1.5)
    evaluation: bool = False
    method: str = "navrl"
    sim = sim_utils.SimulationCfg(
        dt=0.016,
        render_interval=1,
        use_newton_actuators=False,
        physics=PhysxCfg(
            enable_stabilization=True,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=524288,
            gpu_max_rigid_patch_count=163840,
            gpu_found_lost_pairs_capacity=4194304,
            gpu_found_lost_aggregate_pairs_capacity=33554432,
            gpu_total_aggregate_pairs_capacity=4194304,
        ),
    )
    scene = SharedNavigationSceneCfg(
        num_envs=1024, env_spacing=8.0, replicate_physics=False, filter_collisions=True,
    )

    def __post_init__(self):
        self.configure_scene()

    def configure_scene(self):
        """Apply task counts after command-line overrides and before launch."""
        self.scene.terrain = build_terrain_cfg(self.scene.num_envs, self.num_static_obstacles, self.terrain_seed)
        self.scene.dynamic_obstacles = make_dynamic_cfg(self.num_dynamic_obstacles, self.terrain_seed)
        self.scene.lidar.update_period = self.sim.dt * self.decimation
        for actuator in self.scene.drone.actuators.values():
            actuator.dt = self.sim.dt


@configclass
class ZhangEventsCfg:
    parameters = EventTermCfg(
        func="isaaclab_navrl.zhang.events:randomize_zhang_parameters", mode="reset",
    )
    disturbance = EventTermCfg(
        func="isaaclab_navrl.zhang.events:update_zhang_disturbance", mode="interval",
        interval_range_s=(0.1, 0.1), is_global_time=True, params={"interval_s": 0.1},
    )


@configclass
class ZhangNavRLEnvCfg(NavRLEnvCfg):
    method = "zhang"
    decimation = 10
    events = ZhangEventsCfg()

    def __post_init__(self):
        self.sim.dt = 0.01
        self.sim.render_interval = 10
        super().__post_init__()

    def configure_scene(self):
        super().configure_scene()
        self.scene.lidar.update_period = self.sim.dt
        self.scene.drone.class_type = ResolvableString(
            "isaaclab_navrl.tasks.navigation.disturbed_multirotor:DisturbedMultirotor"
        )
        # NavRL ros1/map_manager/cfg/dynamic_map_param.yaml and quadcopter.urdf.
        target = MultiMeshRayCasterCfg.RaycastTargetCfg
        meshes = [
            target(prim_expr="/World/ground", is_shared=True, track_mesh_transforms=False),
            target(prim_expr="/World/defaultGroundPlane", is_shared=True, track_mesh_transforms=False),
        ]
        meshes.extend(
            target(prim_expr=f"/World/Dynamic/obstacle_{i:03d}", is_shared=True, track_mesh_transforms=True)
            for i in range(self.num_dynamic_obstacles)
        )
        self.scene.depth_camera = BatchedDepthCameraCfg(
            prim_path="{ENV_REGEX_NS}/Hummingbird/base_link",
            spawn=None,
            update_period=0.1,
            seed=self.seed + 101,
            pixel_noise_std=0.0 if self.evaluation else 0.02,
            mesh_prim_paths=meshes,
            offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
                pos=(0.09, 0.0, 0.095), rot=(0.0, 0.0, 0.0, 1.0), convention="world"
            ),
            pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
                [554.254691191187, 0.0, 320.5, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 1.0],
                width=640, height=480,
            ),
            max_distance=24.0,
            depth_clipping_behavior="max",
            data_types=["distance_to_image_plane"],
        )
