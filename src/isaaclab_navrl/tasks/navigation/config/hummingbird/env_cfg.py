"""Official Isaac Lab 3.0 configuration for the two NavRL training methods."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiMeshRayCasterCameraCfg, MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.utils import configclass
from isaaclab.utils.string import ResolvableString
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from isaaclab_navrl.control import ROTOR_CONFIG, make_drone_cfg
from isaaclab_navrl.terrain import build_terrain_cfg

from ...depth_camera_cfg import BatchedDepthCameraCfg
from ...randomization import SimToRealCfg
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
class NavigationEventsCfg:
    nominal_dynamics = EventTermCfg(
        func="isaaclab_navrl.tasks.navigation.randomization:capture_nominal_dynamics", mode="startup",
    )
    mass = EventTermCfg(
        func="isaaclab.envs.mdp.events:randomize_rigid_body_mass", mode="reset",
        params={"asset_cfg": SceneEntityCfg("drone"), "mass_distribution_params": (0.9, 1.1),
                "operation": "scale", "distribution": "uniform", "recompute_inertia": False, "min_mass": 1e-6},
    )
    inertia = EventTermCfg(
        func="isaaclab.envs.mdp.events:randomize_rigid_body_inertia", mode="reset",
        params={"asset_cfg": SceneEntityCfg("drone"), "inertia_distribution_params": (0.9, 1.1),
                "operation": "scale", "distribution": "uniform", "diagonal_only": False},
    )
    policy_parameters = EventTermCfg(
        func="isaaclab_navrl.tasks.navigation.randomization:reset_sim_to_real", mode="reset",
    )


@configclass
class NavRLEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 35.2
    action_space = 3
    observation_space = 202
    state_space = 0
    seed = 0
    num_static_obstacles: int = 350
    num_dynamic_obstacles: int = 60
    training_curriculum: bool = True
    training_stage: str | None = None
    sim_to_real: SimToRealCfg = SimToRealCfg()
    events = NavigationEventsCfg()
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
        self.events = ZhangEventsCfg() if self.method == "zhang" else NavigationEventsCfg()
        active = self.sim_to_real.enabled and not self.evaluation
        if active and self.sim_to_real.mass:
            self.events.mass.params["mass_distribution_params"] = self.sim_to_real.mass_scale_range
        else:
            self.events.mass = None
        if active and self.sim_to_real.inertia:
            self.events.inertia.params["inertia_distribution_params"] = self.sim_to_real.inertia_scale_range
        else:
            self.events.inertia = None
        self.scene.terrain = build_terrain_cfg(self.scene.num_envs, self.num_static_obstacles, self.terrain_seed)
        self.scene.dynamic_obstacles = make_dynamic_cfg(self.dynamic_capacity(), self.terrain_seed)
        self.scene.lidar.update_period = self.sim.dt * self.decimation
        for actuator in self.scene.drone.actuators.values():
            actuator.dt = self.sim.dt
            thrust_scale_max = self.sim_to_real.thrust_scale_range[1] if active and self.sim_to_real.thrust else 1.0
            # RotorGroup enforces each sampled motor limit. The native force
            # adapter must not clip a +10% coefficient back to the nominal limit.
            actuator.thrust_range = (0.0, ROTOR_CONFIG["force_constants"][0] * 838**2 * thrust_scale_max)

    def dynamic_capacity(self) -> int:
        return 120 if self.training_curriculum and self.training_stage is None else self.num_dynamic_obstacles


@configclass
class ZhangEventsCfg(NavigationEventsCfg):
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
            for i in range(self.dynamic_capacity())
        )
        self.scene.depth_camera = BatchedDepthCameraCfg(
            prim_path="{ENV_REGEX_NS}/Hummingbird/base_link",
            spawn=None,
            update_period=0.1,
            seed=self.seed + 101,
            pixel_noise_std=(
                0.0 if self.evaluation else
                (self.sim_to_real.depth_std if self.sim_to_real.enabled and self.sim_to_real.observation_noise else 0.0)
                if self.training_stage is not None or self.sim_to_real.enabled else 0.02
            ),
            mesh_prim_paths=meshes,
            offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
                pos=(0.09, 0.0, 0.095), rot=(0.0, 0.0, 0.0, 1.0), convention="world"
            ),
            pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
                # Scale the NavRL calibration with the image to preserve its field of view.
                [554.254691191187 / 4, 0.0, 320.5 / 4,
                 0.0, 554.254691191187 / 4, 240.5 / 4, 0.0, 0.0, 1.0],
                width=160, height=120,
            ),
            max_distance=24.0,
            depth_clipping_behavior="max",
            data_types=["distance_to_image_plane"],
        )
