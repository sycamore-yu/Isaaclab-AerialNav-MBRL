"""Official Isaac Lab 3.0 configuration for the two Navigation training methods."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiMeshRayCasterCameraCfg, MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.sim.schemas import MassCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.utils import configclass
from isaaclab.utils.string import ResolvableString
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg
from isaaclab_tasks.utils import PresetCfg

from aerialnav.control import make_drone_cfg
from aerialnav.physics import DEFAULT_PHYSICS
from aerialnav.terrain import build_terrain_cfg
from aerialnav.training import DEFAULT_STAGE, STAGE_CONFIG

from ...depth_camera_cfg import BatchedDepthCameraCfg
from ...randomization import SimToRealCfg
from ...task_logic import dynamic_layout


@configclass
class NavigationPhysicsCfg(PresetCfg):
    """Named physics choices used by both navigation tasks."""

    isaacsim_physx = PhysxCfg(
        enable_stabilization=True,
        bounce_threshold_velocity=0.2,
        gpu_max_rigid_contact_count=524288,
        gpu_max_rigid_patch_count=163840,
        gpu_found_lost_pairs_capacity=4194304,
        gpu_found_lost_aggregate_pairs_capacity=33554432,
        gpu_total_aggregate_pairs_capacity=4194304,
    )
    newton_mjwarp = NewtonCfg(
        # MuJoCo mesh contacts use a convex hull. Newton's triangle contacts
        # preserve the original non-convex forest mesh and its free corridors.
        solver_cfg=MJWarpSolverCfg(integrator="implicitfast", disable_sensors=True, use_mujoco_contacts=False),
        load_visual_shapes=True,
    )
    default = {"isaacsim_physx": isaacsim_physx, "newton_mjwarp": newton_mjwarp}[DEFAULT_PHYSICS]


@configclass
class SharedNavigationSceneCfg(InteractiveSceneCfg):
    class_type = "aerialnav.tasks.navigation.scene:SharedNavigationScene"

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
        class_type="aerialnav.tasks.navigation.lidar:NavigationRayCaster",
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
    root = "/World/Dynamic"
    objects = {}
    for index, (position, size) in enumerate(zip(positions, sizes, strict=True)):
        properties = {
            "rigid_props": [UsdPhysicsRigidBodyCfg(kinematic_enabled=True), PhysxRigidBodyCfg(disable_gravity=True)],
            "mass_props": MassCfg(mass=1.0),
            "collision_props": UsdPhysicsCollisionCfg(collision_enabled=False),
            "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        }
        if size[2] > 1:
            spawn = sim_utils.CylinderCfg(radius=float(size[0] / 2), height=5.0, **properties)
        else:
            spawn = sim_utils.CuboidCfg(size=tuple(float(x) for x in size), **properties)
        objects[f"obstacle_{index:03d}"] = RigidObjectCfg(
            prim_path=f"{root}/obstacle_{index:03d}",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(x) for x in position)),
            collision_group=-1,
        )
    return RigidObjectCollectionCfg(rigid_objects=objects)


def make_shared_dynamic_geometry(count: int, seed: int) -> dict[str, AssetBaseCfg]:
    """Author one global render/raycast geometry set, without Newton rigid bodies.

    Dynamic obstacles use analytic signed clearance, not solver contacts. The
    camera receives their live shared transforms directly from the motion state.
    AssetBaseCfg's geometry-only spawners retain the source cuboid/cylinder
    meshes while avoiding N-world kinematic link duplication.
    """
    positions, sizes = dynamic_layout(count, seed)
    result = {}
    for index, (position, size) in enumerate(zip(positions, sizes, strict=True)):
        geometry = (
            sim_utils.CylinderCfg(radius=float(size[0] / 2), height=5.0)
            if size[2] > 1
            else sim_utils.CuboidCfg(size=tuple(float(x) for x in size))
        )
        result[f"dynamic_visual_{index:03d}"] = AssetBaseCfg(
            prim_path=f"/World/Dynamic/obstacle_{index:03d}",
            spawn=geometry,
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(float(x) for x in position)),
            collision_group=0,
        )
    return result


@configclass
class NavigationEventsCfg:
    nominal_dynamics = EventTermCfg(
        func="aerialnav.tasks.navigation.randomization:capture_nominal_dynamics",
        mode="startup",
    )
    mass = EventTermCfg(
        func="isaaclab.envs.mdp.events:randomize_rigid_body_mass",
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("drone"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": False,
            "min_mass": 1e-6,
        },
    )
    inertia = EventTermCfg(
        func="isaaclab.envs.mdp.events:randomize_rigid_body_inertia",
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("drone"),
            "inertia_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
            "diagonal_only": False,
        },
    )
    policy_parameters = EventTermCfg(
        func="aerialnav.tasks.navigation.randomization:reset_sim_to_real",
        mode="reset",
    )


@configclass
class NavigationEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 35.2
    action_space = 3
    observation_space = 202
    state_space = 0
    seed = 0
    num_static_obstacles: int = 300
    num_dynamic_obstacles: int = 0
    depth_camera_batch_size: int = 16
    training_curriculum: bool = True
    training_stage: str | None = None
    training_stage_config: dict | None = None
    training_report_episodes: int | None = None
    sim_to_real: SimToRealCfg = SimToRealCfg()
    events = NavigationEventsCfg()
    terrain_seed: int = 0
    dynamic_speed_range: tuple[float, float] = (0.5, 1.5)
    evaluation: bool = False
    method: str = "ppo"
    sim = sim_utils.SimulationCfg(
        dt=0.016,
        render_interval=1,
        use_newton_actuators=False,
        physics=NavigationPhysicsCfg(),
    )
    scene = SharedNavigationSceneCfg(
        num_envs=1024,
        env_spacing=8.0,
        replicate_physics=False,
        filter_collisions=True,
    )

    def __post_init__(self):
        self.configure_scene()

    def configure_scene(self):
        """Apply task counts after command-line overrides and before launch."""
        newton = isinstance(self.sim.physics, NewtonCfg)
        # Newton's global geometry is read by one camera BVH; movement comes
        # from shared GPU tensors. No dynamic rigid body enters any MJWarp world.
        self.scene.replicate_physics = newton
        self.scene.env_spacing = 0.0 if newton else 8.0
        self.scene.lidar.class_type = ResolvableString(
            "aerialnav.tasks.navigation.newton_sensors:NewtonNavigationRayCaster"
            if newton
            else "aerialnav.tasks.navigation.lidar:NavigationRayCaster"
        )
        self.events = DiffNavEventsCfg() if self.method == "diffnav" else NavigationEventsCfg()
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
        self.scene.dynamic_obstacles = None if newton else make_dynamic_cfg(self.dynamic_capacity(), self.terrain_seed)
        # configure_scene is idempotent, including when preset resolution runs
        # before the final CLI/backend selection.
        for name in list(vars(self.scene)):
            if name.startswith("dynamic_visual_"):
                delattr(self.scene, name)
        if newton:
            for name, asset_cfg in make_shared_dynamic_geometry(self.dynamic_capacity(), self.terrain_seed).items():
                setattr(self.scene, name, asset_cfg)
        self.scene.lidar.update_period = self.sim.dt * self.decimation

        # Navigation ros1/map_manager/cfg/dynamic_map_param.yaml and quadcopter.urdf.
        target = MultiMeshRayCasterCfg.RaycastTargetCfg
        meshes = [
            target(prim_expr="/World/ground", is_shared=True, track_mesh_transforms=False),
            target(prim_expr="/World/defaultGroundPlane", is_shared=True, track_mesh_transforms=False),
        ]
        dynamic_root = "/World/Dynamic"
        meshes.extend(
            target(prim_expr=f"{dynamic_root}/obstacle_{i:03d}", is_shared=True, track_mesh_transforms=not newton)
            for i in range(self.dynamic_capacity())
        )
        self.scene.depth_camera = BatchedDepthCameraCfg(
            class_type=(
                "aerialnav.tasks.navigation.newton_sensors:NewtonBatchedDepthCamera"
                if newton
                else "aerialnav.tasks.navigation.depth_camera:BatchedDepthCamera"
            ),
            prim_path="{ENV_REGEX_NS}/Hummingbird/base_link",
            spawn=None,
            batch_size=self.depth_camera_batch_size,
            update_period=self.sim.dt * self.decimation,
            seed=self.seed + 101,
            pixel_noise_std=(
                0.0
                if self.evaluation
                else (
                    self.sim_to_real.depth_std
                    if self.sim_to_real.enabled and self.sim_to_real.observation_noise
                    else 0.0
                )
                if self.training_stage is not None or self.sim_to_real.enabled
                else 0.02
            ),
            mesh_prim_paths=meshes,
            offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
                pos=(0.09, 0.0, 0.095), rot=(0.0, 0.0, 0.0, 1.0), convention="world"
            ),
            pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
                # Scale the Navigation calibration with the image to preserve its field of view.
                [554.254691191187 / 4, 0.0, 320.5 / 4, 0.0, 554.254691191187 / 4, 240.5 / 4, 0.0, 0.0, 1.0],
                width=160,
                height=120,
            ),
            max_distance=24.0,
            depth_clipping_behavior="max",
            data_types=["distance_to_image_plane"],
        )

    def dynamic_capacity(self) -> int:
        return 120 if self.training_curriculum and self.training_stage is None else self.num_dynamic_obstacles


@configclass
class DiffNavEventsCfg(NavigationEventsCfg):
    parameters = EventTermCfg(
        func="aerialnav.diffnav.events:randomize_diffnav_parameters",
        mode="reset",
    )
    disturbance = EventTermCfg(
        func="aerialnav.diffnav.events:update_diffnav_disturbance",
        mode="interval",
        interval_range_s=(0.1, 0.1),
        is_global_time=True,
        params={"interval_s": 0.1},
    )


@configclass
class DiffNavEnvCfg(NavigationEnvCfg):
    method = "diffnav"
    training_stage: str | None = DEFAULT_STAGE
    num_static_obstacles: int = STAGE_CONFIG[DEFAULT_STAGE]["static_obstacles"]
    num_dynamic_obstacles: int = STAGE_CONFIG[DEFAULT_STAGE]["dynamic_obstacles"]
    training_report_episodes: int | None = 1024
    sim_to_real: SimToRealCfg = SimToRealCfg(enabled=STAGE_CONFIG[DEFAULT_STAGE]["randomization"])
    variant: str = "dcol"
    timing: str = "unified_20hz"
    capture_terminal_observation: bool = False
    depth_camera_batch_size: int = 64
    height_target_range: tuple[float, float] = (0.5, 2.5)
    target_speed_range: tuple[float, float] | None = None
    decimation = 10
    events = DiffNavEventsCfg()

    def __post_init__(self):
        self.sim.dt = 0.01
        self.sim.render_interval = 10
        super().__post_init__()

    def configure_scene(self):
        if self.variant not in ("step", "continuous", "dcol"):
            raise ValueError(f"Unknown DiffNav variant: {self.variant}")
        if self.timing not in ("baseline", "unified_20hz", "unified_50hz"):
            raise ValueError(f"Unknown DiffNav timing: {self.timing}")
        self.sim.dt = {"baseline": 0.01, "unified_20hz": 0.05, "unified_50hz": 0.02}[self.timing]
        self.decimation = 10 if self.timing == "baseline" else 1
        self.sim.render_interval = self.decimation
        low, high = self.height_target_range
        if not 0.2 < low < high < 4.0:
            raise ValueError("DiffNav height_target_range must lie strictly within (0.2, 4.0) metres")
        super().configure_scene()
        if self.target_speed_range is not None:
            self.events.parameters.params["target_speed_range"] = self.target_speed_range
        period = self.sim.dt * self.decimation
        self.events.disturbance.interval_range_s = (period, period)
        self.events.disturbance.params["interval_s"] = period
        # UsdFileCfg's original None preserves the asset's collision attributes.
        # Reassign on every configuration pass so a previous variant cannot leak.
        self.scene.drone.spawn.collision_props = (
            UsdPhysicsCollisionCfg(collision_enabled=False) if self.variant == "continuous" else None
        )
        self.scene.lidar.update_period = self.sim.dt
