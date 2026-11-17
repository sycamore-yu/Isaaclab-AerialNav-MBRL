"""Newton/Warp navigation with velocity and world-acceleration action semantics."""

from __future__ import annotations

import torch
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    matrix_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
)
from isaaclab_tasks.utils.hydra import resolve_presets

from isaaclab_navrl.training import MAP_SIZE
from isaaclab_navrl.warp_control import GraphedWarpController
from isaaclab_navrl.zhang.termination import termination_mask

from .context import ObservationContext
from .curriculum import TrainingCurriculum, active_obstacle_indices
from .depth_history import (
    compensated_depth_features,
    pooled_event_channels,
    pooled_signed_event_channel,
    reproject_depth,
    reproject_event_channels,
    reproject_signed_event_channel,
)
from .geometry import TerrainDistance, query_clearance
from .randomization import PolicyPerturbation
from .task import NavigationTask
from .task_logic import DynamicObstacleMotion, goal_frame, sample_navigation_route
from .transition import TransitionPublisher
from .warp_loop import _NavigationWarpLoop
from .warp_task_kernels import WarpFirstEvents, WarpNavigationTerms


class NavRLEnv(_NavigationWarpLoop):
    """NavRL velocity-command task on the Newton/Warp loop."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        # Direct gym.make callers use the same task default as the public CLI.
        resolve_presets(cfg)
        if cfg.evaluation:
            cfg.sim_to_real.enabled = False
        cfg.configure_scene()
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.terrain_distance = TerrainDistance.from_stage(self.sim.stage, self.device)
        self.drone = self.scene["drone"]
        self.lidar = self.scene["lidar"]
        self.controller = self._create_controller()
        self.curriculum = (
            TrainingCurriculum(
                self.num_envs,
                self.device,
                stage=cfg.training_stage,
                stage_config=getattr(cfg, "training_stage_config", None),
            )
            if cfg.training_curriculum
            else None
        )
        if self.curriculum is not None and cfg.training_report_episodes is not None:
            if cfg.training_report_episodes < 1:
                raise ValueError("training_report_episodes must be positive")
            self.curriculum.min_episodes = cfg.training_report_episodes
        self.perturbation = PolicyPerturbation(
            cfg.sim_to_real,
            self.num_envs,
            self.device,
            cfg.seed,
            policy_dt=self.step_dt,
            nominal_action_delay=1 if cfg.method == "zhang" else 0,
        )
        self.mass_scale = torch.ones(self.num_envs, 1, device=self.device)
        self.inertia_scale = torch.ones_like(self.mass_scale)
        if self.curriculum is not None:
            cfg.num_dynamic_obstacles = self.curriculum.active_obstacles
        self.motion = DynamicObstacleMotion(
            cfg.num_dynamic_obstacles,
            cfg.terrain_seed,
            self.device,
            self.physics_dt if cfg.method == "zhang" else self.step_dt,
            cfg.dynamic_speed_range,
        )
        self.set_active_obstacles(cfg.num_dynamic_obstacles)
        if getattr(cfg, "geometry_backend", "native") == "curobo_v2":
            from isaaclab_navrl.reference.curobo_geometry import CuroboGeometry

            self.terrain_distance = CuroboGeometry(self.terrain_distance, self.motion.size)
        self.task = NavigationTask(self.num_envs, self.device)
        self.transitions = TransitionPublisher()
        self.command = torch.zeros_like(self.target_pos)
        self.executed_actions = torch.zeros_like(self.command)
        self.previous_actions = torch.zeros_like(self.command)
        self.action_difference = torch.zeros(self.num_envs, device=self.device)
        self.nonfinite_events = torch.zeros((), dtype=torch.long, device=self.device)
        self._reset_rng = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self._terms = None
        self._stepping = False
        self.depth_event_detector = None
        self._depth_stack = None
        self._depth_stack_poses = None
        self._depth_stack_seen = None
        if cfg.method == "zhang" and getattr(cfg, "depth_event_channel", False):
            from .geometric_depth_events import DepthEventDetector

            shape = cfg.scene.depth_camera.pattern_cfg
            history_frames = getattr(cfg, "depth_event_history_frames", 1)
            temporal = history_frames > 1
            kinematic_field = bool(getattr(cfg, "depth_event_kinematic_field", False))
            naive_diff = bool(getattr(cfg, "depth_event_naive_diff", False))
            edge_suppression = bool(
                (getattr(cfg, "depth_event_edge_suppression", False) or getattr(cfg, "depth_event_dual_channel", False))
                and not kinematic_field
                and not naive_diff
            )
            self.depth_event_detector = DepthEventDetector(
                shape.height,
                shape.width,
                max_depth=cfg.scene.depth_camera.max_distance,
                threshold=0.25 if (naive_diff or not temporal) else 0.0,
                relative_threshold=0.01 if (naive_diff or not temporal) else 0.002,
                match_radius=0 if naive_diff else 1,
                history_frames=history_frames,
                depth_sigma=0.0 if (naive_diff or not temporal) else 0.02,
                pose_position_sigma=0.0 if (naive_diff or not temporal) else 0.01,
                pose_rotation_sigma=0.0 if (naive_diff or not temporal) else 0.002,
                sigma_multiplier=0.0 if (naive_diff or not temporal) else 2.5,
                edge_suppression=edge_suppression,
                kinematic_field=kinematic_field,
                step_dt=self.step_dt,
            )
        self._warp_term_bindings = {}
        self._warp_obstacle_count = len(self.motion.position)
        self._warp_events = WarpFirstEvents(
            root_state=torch.zeros((self.num_envs, 13), device=self.device),
            target=self.target_pos,
            collision_gap=torch.zeros(self.num_envs, device=self.device),
            collision_terminates=getattr(self.cfg, "variant", None) != "continuous",
            bounds_terminates=getattr(self.cfg, "variant", None) != "continuous",
        )

    # Existing environment readers and Warp graphs see the owner's tensors,
    # never copied state or a second reset implementation.
    target_pos = property(lambda self: self.task.target_pos)
    target_speed = property(lambda self: self.task.target_speed)
    target_dir = property(lambda self: self.task.target_dir)
    height_range = property(lambda self: self.task.height_range)
    previous_velocity = property(lambda self: self.task.previous_velocity)
    first_reach_step = property(lambda self: self.task.first_reach_step)
    arrived = property(lambda self: self.task.arrived)
    arrive_time = property(lambda self: self.task.arrive_time)
    episode_return = property(lambda self: self.task.episode_return)
    episode_id = property(lambda self: self.task.episode_id)
    _episode_failure = property(lambda self: self.task.episode_failure)
    _event_code = property(lambda self: self.task.first_event_code)
    _event_time = property(lambda self: self.task.first_event_time)

    def _create_controller(self):
        return GraphedWarpController(self.num_envs, self.device, dt=self.physics_dt)

    def _setup_scene(self):
        self.scene.filter_collisions(global_prim_paths=["/World/ground", "/World/defaultGroundPlane"])

    def root_state(self):
        """World position, XYZW orientation, linear and angular velocity."""
        data = self.drone.data
        return torch.cat(
            (data.root_pos_w.torch, data.root_quat_w.torch, data.root_lin_vel_w.torch, data.root_ang_vel_w.torch),
            dim=-1,
        )

    def _reset_idx(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.int32)
        if self._stepping:
            transition = self.extras["transition"]
            self.extras["log"] = {
                "Episode/return": self.episode_return[env_ids].mean(),
                "Episode/length": self.episode_length_buf[env_ids].float().mean(),
                "Episode/first_arrival": (self.first_reach_step[env_ids] > 0).float().mean(),
                "Episode/reach_goal": transition["reached"][env_ids].float().mean(),
                "Episode/collision": transition["collision"][env_ids].float().mean(),
            }
        # No shared-world object or trajectory is reset here.
        super()._reset_idx(env_ids)
        count = len(env_ids)
        if self.cfg.method == "zhang":
            speed = self.zhang_parameters["max_speed"][env_ids].clone()
        else:
            speed = 1 + 5 * torch.rand(count, 1, generator=self._reset_rng, device=self.device)
        position, target = sample_navigation_route(
            speed, self._reset_rng, self.cfg.height_target_range if self.cfg.method == "zhang" else (0.5, 2.5)
        )
        # Keep the uniform speed draw; reject occupied endpoints instead of shortening the route.
        for _ in range(64):
            clearance = self.query_clearance(torch.cat((position, target)), body_radius=0.3).reshape(2, count).amin(0)
            blocked = clearance <= 0
            if not blocked.any():
                break
            position[blocked], target[blocked] = sample_navigation_route(
                speed[blocked],
                self._reset_rng,
                self.cfg.height_target_range if self.cfg.method == "zhang" else (0.5, 2.5),
            )
        else:
            raise RuntimeError("Could not sample collision-free 18v route endpoints after 64 attempts")
        self.task.reset(env_ids, position, target, speed)
        direction = target - position
        zero = torch.zeros(count, device=self.device)
        quaternion = quat_from_euler_xyz(zero, zero, torch.atan2(direction[:, 1], direction[:, 0]))
        self.drone.write_root_pose_to_sim_index(root_pose=torch.cat((position, quaternion), dim=-1), env_ids=env_ids)
        self.drone.write_root_velocity_to_sim_index(root_velocity=position.new_zeros(count, 6), env_ids=env_ids)
        self.drone.write_joint_position_to_sim_index(
            position=self.drone.data.default_joint_pos.torch[env_ids], env_ids=env_ids
        )
        self.controller.reset(env_ids)
        self.controller.apply(self.drone)
        reset_ids = env_ids.long()
        self.command.index_fill_(0, reset_ids, 0)
        self.previous_actions.index_fill_(0, reset_ids, 0)
        if getattr(self, "depth_event_detector", None) is not None:
            self.depth_event_detector.reset(env_ids)
        if getattr(self, "_depth_stack_seen", None) is not None:
            self._depth_stack_seen[env_ids] = 0
        self._terms = None

    def reset(self, *args, **kwargs):
        self._stepping = False
        self.transitions.clear()
        self.extras = {}
        if self.curriculum is not None:
            self.curriculum.reset_episodes()
        return super().reset(*args, **kwargs)

    def step(self, actions):
        try:
            result = super().step(actions)
        finally:
            # Only in-step autoresets may publish completed-episode logs.
            self._stepping = False
        if self.curriculum is not None:
            self.curriculum.observe(result[-1]["transition"])
        return result

    def set_active_obstacles(self, count: int):
        """Change the shared difficulty without respawning USD or copying worlds."""
        self._active_obstacle_ids = torch.tensor(
            active_obstacle_indices(count, self.cfg.dynamic_capacity()), device=self.device, dtype=torch.long
        )
        self.motion = DynamicObstacleMotion(
            count,
            self.cfg.terrain_seed,
            self.device,
            self.physics_dt if self.cfg.method == "zhang" else self.step_dt,
            self.cfg.dynamic_speed_range,
        )
        self.cfg.num_dynamic_obstacles = count
        self._bind_shared_dynamic_geometry()

    def _bind_shared_dynamic_geometry(self):
        """Connect one global obstacle process to Newton's camera meshes."""
        self.scene["depth_camera"].set_motion(self.motion, self._active_obstacle_ids)

    def advance_curriculum(self):
        count = self.curriculum.advance()
        self.set_active_obstacles(count)
        # Every new episode belongs to one difficulty; interrupted episodes do
        # not enter success statistics or cross-stage temporal histories.
        return self.reset()

    def _pre_physics_step(self, actions):
        self._stepping = True
        self._terms = None
        self.extras = {}
        self.actions = actions.detach().clone()
        self.action_difference = (self.actions - self.previous_actions).square().sum(-1)
        self.previous_actions.copy_(self.actions)
        self.executed_actions = self.perturbation.execute(self.actions)
        self.executed_action_lags = self.perturbation.action_buffer.time_lags.clone()
        self.command = goal_frame(self.executed_actions, self.target_dir, inverse=True)

    @torch.no_grad()
    def _apply_action(self):
        self.controller.compute(self.root_state(), self.command)
        self.controller.apply(self.drone)

    def _update_dynamic_obstacles(self):
        self.motion.advance()

    def query_clearance(self, query, *, body_radius=0.0, query_time=0.0):
        """Metre clearance from current shared geometry at an explicit time offset."""
        return query_clearance(
            query,
            self.terrain_distance.signed_distance(query),
            self.motion.position,
            self.motion.size,
            body_radius=body_radius,
            query_time=query_time,
            dynamic_velocity=self.motion.velocity,
            dynamic_backend=getattr(self.terrain_distance, "dynamic_signed_distance", None),
        )

    def _compute_first_events(self):
        state = self.root_state()
        self._warp_events.update(
            root_state=state,
            collision_gap=self.query_clearance(state[:, :3], body_radius=0.3),
        )
        self._run_warp("first_events", self._warp_events.launch)
        terms = dict(self._warp_events.terms)
        if not getattr(self.cfg, "height_bounds_reset", True):
            terms["bounds"] = (state[:, :2].abs() > state.new_tensor(MAP_SIZE) / 2).any(-1)
        return terms

    def _compute_terms(self):
        state = self.root_state()
        hits = self.lidar.data.ray_hits_w.torch
        origin = self.lidar.data.pos_w.torch
        distances = (hits - origin[:, None]).norm(dim=-1).clamp_max(4)
        binding = self._navigation_terms(state[:, :3], state[:, 7:10], distances)
        self._terms = dict(binding.terms)
        self._terms.update(self._compute_first_events())
        return self._terms

    def _get_dones(self):
        # Preserve physics -> shared obstacle motion -> observation ordering.
        self._update_dynamic_obstacles()
        terms = self._finish_policy_terms(self._compute_terms())
        self.nonfinite_events += terms["invalid"].sum()
        terminated = self._termination_mask(terms)
        outcome = self.task.advance(
            terms,
            self.episode_length_buf,
            policy_dt=self.step_dt,
            max_episode_length=self.max_episode_length,
            wait_before_truncate=self.cfg.wait_before_truncate,
            terminated=terminated,
            first_arrival=self._first_arrival_mask(terms, terminated),
        )
        timed_out = outcome["timeout"]
        state = self.root_state()
        # These values belong to the completed transition, before same-step autoreset.
        transition = {
            "position": state[:, :3],
            "quaternion": state[:, 3:7],
            "velocity": state[:, 7:10],
            "angular_velocity": state[:, 10:13],
            "target": self.target_pos,
            "episode_id": self.episode_id,
            "height_target": self.target_pos[:, 2],
            "height_range": self.height_range,
            "step": self.episode_length_buf,
            "time_s": self.episode_length_buf.float() * self.step_dt,
            "reached": terms["reached"],
            "collision": terms["collision"],
            "bounds": terms["bounds"],
            "invalid": terms["invalid"],
            "terminated": terminated,
            "arrived": self.arrived,
            "arrive_time": self.arrive_time,
            "raw_action": self.actions,
            "executed_action": self.executed_actions,
            "action_delay_steps": self.executed_action_lags,
            **outcome,
        }
        if self.cfg.method == "zhang" and self.cfg.capture_terminal_observation:
            # Capture before DirectRLEnv autoresets sensors, targets and randomization.
            transition["depth_features"] = (
                self.scene["depth_camera"].data.output["policy_depth"].torch.permute(0, 3, 1, 2)
            )
            transition["input_delay_steps"] = self.perturbation.observation_buffer.time_lags
            transition["critic_parameters"] = self.zhang_parameters
        transition.update(self._transition_details(timed_out))
        self.extras["transition"] = self.transitions.publish(transition)
        return terminated, timed_out

    def _transition_details(self, timeout):
        return {}

    def _termination_mask(self, terms):
        return terms["collision"] | terms["bounds"] | terms["invalid"]

    def _first_arrival_mask(self, terms, terminated):
        return terms["reached"] & ~terminated & (self.first_reach_step == 0)

    def _finish_policy_terms(self, terms):
        return terms

    def _get_rewards(self):
        reward = self._terms["reward"]
        cfg = self.cfg.sim_to_real
        if self.cfg.method == "navrl" and cfg.enabled and cfg.action_difference:
            reward = reward - cfg.action_difference_weight * self.action_difference
        self.task.record_reward(reward, self.drone.data.root_lin_vel_w.torch)
        self.extras["transition"] = self.extras["transition"].with_fields(episode_return=self.episode_return)
        return reward

    def _get_observations(self):
        def sample():
            terms = self._compute_terms()
            binding = self._warp_term_bindings["navigation_terms"]
            if self.cfg.method == "navrl" and self.perturbation.noisy:
                state = self.root_state()
                position, velocity, _ = self.perturbation.state(state[:, :3], state[:, 7:10], state[:, 3:7])
                distances = self.perturbation.normal(4.0 - terms["lidar"], self.cfg.sim_to_real.lidar_std).clamp(0, 4)
                binding = self._navigation_terms(position, velocity, distances, "noisy_navigation_terms")
            # Future graph replays must not mutate retained policy inputs.
            return binding.policy_tensor.clone()

        if self.cfg.method == "zhang":
            return {"policy": sample()}
        return {"policy": self.perturbation.observe(self.common_step_counter, sample)}

    def get_zhang_observation(self):
        state = self.root_state()
        camera = self.scene["depth_camera"]
        features = camera.data.output["policy_depth"].torch
        depth = features.permute(0, 3, 1, 2).clone()
        measured_depth = camera.data.output["distance_to_image_plane"].torch.permute(0, 3, 1, 2)

        event_enabled = self.depth_event_detector is not None
        temporal_event = event_enabled and getattr(self.cfg, "depth_event_history_frames", 1) > 1
        kinematic_event = event_enabled and bool(getattr(self.cfg, "depth_event_kinematic_field", False))
        dual_event = event_enabled and bool(getattr(self.cfg, "depth_event_dual_channel", False)) and not kinematic_event
        highres_event = kinematic_event or dual_event
        event_channels = 3 if kinematic_event else (2 if dual_event else (1 if event_enabled else 0))
        feature_size = (24, 32) if highres_event else (12, 16)
        stack_frames = int(getattr(self.cfg, "depth_stack_frames", 1))
        raw_count = measured_depth[0].numel()
        pose_source = getattr(self.cfg, "depth_event_pose_source", "estimated")
        if pose_source not in ("estimated", "truth"):
            raise ValueError("depth_event_pose_source must be estimated or truth")

        def sample():
            position, velocity, quaternion = self.perturbation.state(state[:, :3], state[:, 7:10], state[:, 3:7])
            pieces = [position, velocity, quaternion, self.target_pos, depth.flatten(1), measured_depth.flatten(1)]
            if event_enabled:
                if pose_source == "truth":
                    pose = torch.cat((camera.data.pos_w.torch, camera.data.quat_w_world.torch), dim=-1)
                else:
                    # Default onboard contract: use estimated body odometry + calibrated
                    # nominal mount in the robotics (+X forward, +Y left, +Z up) frame.
                    offset = camera.cfg.offset
                    offset_world_q = convert_camera_frame_orientation_convention(
                        state.new_tensor(offset.rot).expand(self.num_envs, -1),
                        origin=offset.convention,
                        target="world",
                    )
                    offset_p = state.new_tensor(offset.pos).expand(self.num_envs, -1)
                    cam_pos = position + quat_apply(quaternion, offset_p)
                    cam_quat = quat_mul(quaternion, offset_world_q)
                    pose = torch.cat((cam_pos, cam_quat), dim=-1)
                result = self.depth_event_detector.update(
                    measured_depth[:, 0],
                    pose,
                    camera.data.intrinsic_matrices.torch,
                    float(self.common_step_counter) * self.step_dt,
                )
                if kinematic_event:
                    event_map = result.kinematic_image(pose, camera.data.intrinsic_matrices.torch, camera.cfg.max_distance)
                elif dual_event:
                    event_map = result.dual_image(pose, camera.data.intrinsic_matrices.torch, camera.cfg.max_distance)
                else:
                    event_map = result.signed_image(pose, camera.data.intrinsic_matrices.torch, camera.cfg.max_distance)
                pieces.append(event_map.flatten(1))
            return torch.cat(pieces, dim=-1)

        policy = self.perturbation.observe(self.common_step_counter, sample)
        # The fast odometry channel models an onboard state estimate, with the
        # configured measurement noise; only the delivered depth frame is delayed.
        current = self.perturbation.latest_observation
        age = self.perturbation.observation_age_s
        delivered_raw_depth = policy[:, 205 : 205 + raw_count].reshape_as(measured_depth)
        if highres_event:
            unwarped_known = torch.isfinite(delivered_raw_depth) & (delivered_raw_depth > 0)
            policy_depth = compensated_depth_features(
                delivered_raw_depth, unwarped_known, output_size=feature_size
            )
            delivered_event = policy[:, 205 + raw_count : 205 + (1 + event_channels) * raw_count].reshape(
                self.num_envs, event_channels, *measured_depth.shape[2:]
            )
        else:
            policy_depth = policy[:, 13:205].reshape(self.num_envs, 1, 12, 16)
            delivered_event = (
                policy[:, 205 + raw_count : 205 + 2 * raw_count].reshape_as(measured_depth) if event_enabled else None
            )
        known_fraction = age.new_ones(self.num_envs)
        if self.cfg.sim_to_real.enabled and self.cfg.sim_to_real.input_delay:
            # Use calibrated nominal mounting; hidden pitch perturbations remain
            # calibration error instead of becoming privileged actor information.
            offset = camera.cfg.offset
            offset_q = convert_camera_frame_orientation_convention(
                state.new_tensor(offset.rot).expand(self.num_envs, -1), origin=offset.convention, target="ros"
            )
            offset_p = state.new_tensor(offset.pos).expand(self.num_envs, -1)

            def camera_pose(estimate):
                quaternion = estimate[:, 6:10]
                return matrix_from_quat(quat_mul(quaternion, offset_q)), estimate[:, :3] + quat_apply(
                    quaternion, offset_p
                )

            source_rotation, source_position = camera_pose(policy)
            target_rotation, target_position = camera_pose(current)
            warped, known = reproject_depth(
                delivered_raw_depth,
                camera.data.intrinsic_matrices.torch,
                source_rotation,
                source_position,
                target_rotation,
                target_position,
            )
            policy_depth = torch.where(
                (age > 0)[:, None, None, None],
                compensated_depth_features(warped, known, output_size=feature_size),
                policy_depth,
            )
            known_fraction = torch.where(age > 0, known.float().mean((1, 2, 3)), known_fraction)
            if event_enabled:
                if highres_event:
                    aligned_event = reproject_event_channels(
                        delivered_raw_depth,
                        delivered_event,
                        camera.data.intrinsic_matrices.torch,
                        source_rotation,
                        source_position,
                        target_rotation,
                        target_position,
                        warped,
                        known,
                    )
                    pooled_event = pooled_event_channels(
                        warped, aligned_event, known, output_size=feature_size, region_pool=temporal_event
                    )
                    original_event = pooled_event_channels(
                        measured_depth, delivered_event, output_size=feature_size, region_pool=temporal_event
                    )
                else:
                    aligned_event = reproject_signed_event_channel(
                        delivered_raw_depth,
                        delivered_event,
                        camera.data.intrinsic_matrices.torch,
                        source_rotation,
                        source_position,
                        target_rotation,
                        target_position,
                        warped,
                        known,
                    )
                    pooled_event = pooled_signed_event_channel(warped, aligned_event, known, region_pool=temporal_event)
                    original_event = pooled_signed_event_channel(
                        measured_depth, delivered_event, region_pool=temporal_event
                    )
                delivered_event_feature = torch.where((age > 0)[:, None, None, None], pooled_event, original_event)
                delivered_motion_pixels = torch.where((age > 0)[:, None, None, None], aligned_event, delivered_event)
        if event_enabled:
            if not (self.cfg.sim_to_real.enabled and self.cfg.sim_to_real.input_delay):
                delivered_event_feature = (
                    pooled_event_channels(
                        measured_depth, delivered_event, output_size=feature_size, region_pool=temporal_event
                    )
                    if highres_event
                    else pooled_signed_event_channel(measured_depth, delivered_event, region_pool=temporal_event)
                )
                delivered_motion_pixels = delivered_event
            policy_depth = torch.cat((policy_depth, delivered_event_feature), dim=1)
        elif stack_frames > 1:
            # 4-Frame Depth + Delta-Pose Stacking baseline (no explicit geometric reprojection).
            cur_pose = torch.cat((current[:, :3], current[:, 6:10]), dim=-1)
            if self._depth_stack is None or self._depth_stack.shape[0] != self.num_envs:
                self._depth_stack = policy_depth[:, :1].unsqueeze(1).repeat(1, stack_frames, 1, 1, 1).clone()
                self._depth_stack_poses = cur_pose[:, None, :].repeat(1, stack_frames, 1).clone()
                self._depth_stack_seen = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            fresh_env = self._depth_stack_seen == 0
            if bool(fresh_env.any()):
                self._depth_stack[fresh_env] = policy_depth[fresh_env, :1].unsqueeze(1).repeat(1, stack_frames, 1, 1, 1)
                self._depth_stack_poses[fresh_env] = cur_pose[fresh_env, None, :].repeat(1, stack_frames, 1)
            # Shift older frames right and store newest at slot 0.
            self._depth_stack[:, 1:] = self._depth_stack[:, :-1].clone()
            self._depth_stack[:, 0] = policy_depth[:, :1]
            self._depth_stack_poses[:, 1:] = self._depth_stack_poses[:, :-1].clone()
            self._depth_stack_poses[:, 0] = cur_pose
            self._depth_stack_seen.add_(1)
            # Build relative 6-DoF delta pose (dx, dy, dz, rx, ry, rz) between consecutive frames in current frame.
            delta_channels = []
            for s_idx in range(stack_frames - 1):
                p_newer = self._depth_stack_poses[:, s_idx]
                p_older = self._depth_stack_poses[:, s_idx + 1]
                dp_w = p_older[:, :3] - p_newer[:, :3]
                q_new = p_newer[:, 3:7]
                q_inv = torch.cat((-q_new[:, :3], q_new[:, 3:4]), dim=-1)
                dp_local = quat_apply(q_inv, dp_w)
                dq = quat_mul(q_inv, p_older[:, 3:7])
                dr_local = 2.0 * dq[:, :3] * torch.where(dq[:, 3:4] < 0, -1.0, 1.0)
                pose6 = torch.cat((dp_local, dr_local), dim=-1)[:, :, None, None].expand(-1, -1, 12, 16)
                delta_channels.append(pose6)
            policy_depth = torch.cat((self._depth_stack[:, :, 0], *delta_channels), dim=1)
        privileged = {
            "depth_features": depth,
            "measured_depth": measured_depth[:, 0],
            "position": state[:, :3],
            "quaternion": state[:, 3:7],
            "velocity": state[:, 7:10],
            "target": self.target_pos.clone(),
            "episode_id": self.episode_id.clone(),
            "height_target": self.target_pos[:, 2].clone(),
            "height_range": self.height_range.clone(),
            "depth_age_s": age.clone(),
            "depth_known_fraction": known_fraction,
            "input_delay_steps": self.perturbation.observation_buffer.time_lags.clone(),
        }
        policy_observation = {
            "position": current[:, :3],
            "velocity": current[:, 3:6],
            "quaternion": current[:, 6:10],
            "target": current[:, 10:13],
            "depth_features": policy_depth,
        }
        if event_enabled:
            # Full-resolution signed evidence is available to mask consumers.
            # It is not a full object segmentation and is not fed into the CNN.
            policy_observation["depth_motion_pixels"] = delivered_motion_pixels
        if self.cfg.perception_cfg.mode == "pointnet":
            from .pointcloud import pack_body_points

            policy_observation["points"] = pack_body_points(
                self.scene["point_lidar"].data.ray_hits_w.torch, privileged["position"], privileged["quaternion"]
            )
        return ObservationContext.build(policy=policy_observation, privileged=privileged)

    def _navigation_terms(self, position, velocity, distances, stage="navigation_terms"):
        if self._warp_obstacle_count != len(self.motion.position):
            # Curriculum changes are infrequent host boundaries. Never replay
            # a graph whose obstacle buffer shape or pointer has changed.
            for key in self._warp_term_bindings:
                self._graph_cache.invalidate(key)
            self._warp_term_bindings.clear()
            self._warp_obstacle_count = len(self.motion.position)
        selected_count = min(5, len(self.motion.position))
        if selected_count:
            distance_xy = (self.motion.position[None, :, :2] - position[:, None, :2]).norm(dim=-1)
            selected_distances, selected_indices = distance_xy.topk(selected_count, dim=-1, largest=False)
        else:
            selected_distances = position.new_empty((self.num_envs, 0))
            selected_indices = torch.empty((self.num_envs, 0), dtype=torch.long, device=self.device)
        changed = {
            "position": position,
            "velocity": velocity,
            "lidar_distance": distances.reshape(self.num_envs, 144),
            "obstacle_position": self.motion.position,
            "obstacle_velocity": self.motion.velocity,
            "obstacle_size": self.motion.size,
            "selected_indices": selected_indices,
            "selected_distance_xy": selected_distances,
        }
        binding = self._warp_term_bindings.get(stage)
        if binding is None:
            binding = WarpNavigationTerms(
                **{name: value.clone() for name, value in changed.items()},
                target=self.target_pos,
                direction=self.target_dir,
                previous_velocity=self.previous_velocity,
                height_range=self.height_range,
            )
            self._warp_term_bindings[stage] = binding
        else:
            binding.update(**changed)
        self._run_warp(stage, binding.launch)
        return binding


class ZhangNavRLEnv(NavRLEnv):
    """The same navigation world with unrestricted world-acceleration actions."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._configure_controller()
        self._substep_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._collision_position = torch.zeros(self.num_envs, 3, device=self.device)
        self._collision_velocity = torch.zeros_like(self._collision_position)
        self._collision_substep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._dynamic_positions_by_substep = (
            self.motion.position.new_zeros((self.cfg.decimation, *self.motion.position.shape))
            if self.cfg.variant == "dcol"
            else None
        )
        self._dynamic_velocities_by_substep = (
            torch.zeros_like(self._dynamic_positions_by_substep) if self.cfg.variant == "dcol" else None
        )
        self._substep_bounds = torch.zeros_like(self._substep_collision)
        self._substep_invalid = torch.zeros_like(self._substep_collision)
        self.filtered_command = torch.zeros(self.num_envs, 3, device=self.device)

    def _configure_controller(self):
        self._rescale_motor_response()

    def _rescale_motor_response(self):
        if self.physics_dt != 0.01:
            # Preserve the baseline motor time constant when changing its sample period.
            for name in ("tau_up", "tau_down"):
                tau = self.controller.rotor_parameters[name]
                tau.copy_(1 - (1 - tau).pow(self.physics_dt / 0.01))

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        reset_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.filtered_command.index_fill_(0, reset_ids, 0)
        self._substep_collision.index_fill_(0, reset_ids, False)
        self._collision_position.index_fill_(0, reset_ids, 0)
        self._collision_velocity.index_fill_(0, reset_ids, 0)
        self._collision_substep.index_fill_(0, reset_ids, 0)
        self._substep_bounds.index_fill_(0, reset_ids, False)
        self._substep_invalid.index_fill_(0, reset_ids, False)

    def _termination_mask(self, terms):
        return termination_mask(
            terms,
            collision_terminates=self.cfg.variant != "continuous",
            bounds_terminates=self.cfg.variant != "continuous",
        )

    def _first_arrival_mask(self, terms, terminated):
        # The latched physics-substep outcome also excludes later arrivals
        # after a nonterminal collision or bounds failure in the continuous variant.
        return (self._event_code == 1) & (self.first_reach_step == 0)

    def apply_zhang_camera_randomization(self, env_ids):
        camera = self.scene["depth_camera"]
        pattern = camera.cfg.pattern_cfg
        scale = self.zhang_parameters["camera_fov_scale"][env_ids].reshape(-1)
        matrices = torch.zeros(len(env_ids), 3, 3, device=self.device)
        base_fx = pattern.width * pattern.focal_length / pattern.horizontal_aperture
        base_fy = pattern.height * pattern.focal_length / pattern.vertical_aperture
        matrices[:, 0, 0], matrices[:, 1, 1] = base_fx / scale, base_fy / scale
        matrices[:, 0, 2] = pattern.horizontal_aperture_offset * base_fx + pattern.width / 2
        matrices[:, 1, 2] = pattern.vertical_aperture_offset * base_fy + pattern.height / 2
        matrices[:, 2, 2] = 1
        camera.set_intrinsic_matrices(matrices, env_ids=env_ids)
        camera.set_mount_pitch(self.zhang_parameters["camera_pitch"][env_ids].reshape(-1), env_ids)

    def _pre_physics_step(self, actions):
        self._stepping = True
        self._terms = None
        self.extras = {}
        self.actions = actions.detach().clone()
        self.executed_actions = self.perturbation.execute(self.actions)
        self.executed_action_lags = self.perturbation.action_buffer.time_lags.clone()
        self.command = self.executed_actions
        self._physical_substep = 0
        self._policy_start_time = self.episode_length_buf.float() * self.step_dt
        self._substep_collision.zero_()
        self._collision_position.zero_()
        self._collision_velocity.zero_()
        self._collision_substep.zero_()
        self._substep_bounds.zero_()
        self._substep_invalid.zero_()
        if self.cfg.variant == "dcol":
            shape = (self.cfg.decimation, *self.motion.position.shape)
            if self._dynamic_positions_by_substep.shape != shape:
                # The legacy curriculum can change the shared obstacle count.
                self._dynamic_positions_by_substep = self.motion.position.new_zeros(shape)
                self._dynamic_velocities_by_substep = torch.zeros_like(self._dynamic_positions_by_substep)

    def _record_substep(self, terms):
        # A 10 Hz policy can cross the collision band between observations.
        # Use the same sphere clearance at every completed Newton step.
        elapsed_s = self._policy_start_time + self._physical_substep * self.physics_dt
        self.task.record_substep(terms, elapsed_s)
        # Keep this policy step's first collision even if the episode's first
        # outcome was already a success. Capture before the aggregate is set.
        first_collision = terms["collision"] & ~self._substep_collision
        self._collision_position.copy_(
            torch.where(first_collision[:, None], self.root_state()[:, :3], self._collision_position)
        )
        self._collision_velocity.copy_(
            torch.where(first_collision[:, None], self.root_state()[:, 7:10], self._collision_velocity)
        )
        self._collision_substep.copy_(torch.where(first_collision, self._physical_substep, self._collision_substep))
        if self.cfg.variant == "dcol":
            self._dynamic_positions_by_substep[self._physical_substep - 1].copy_(self.motion.position)
            self._dynamic_velocities_by_substep[self._physical_substep - 1].copy_(self.motion.velocity)
        self._substep_collision |= terms["collision"]
        self._substep_bounds |= terms["bounds"]
        self._substep_invalid |= terms["invalid"]

    def _finish_policy_terms(self, terms):
        self._record_substep(terms)
        terms["collision"] = self._substep_collision.clone()
        terms["bounds"] = self._substep_bounds.clone()
        terms["invalid"] = self._substep_invalid.clone()
        return terms

    def _transition_details(self, timeout):
        self.task.finish_first_events(timeout, self.episode_length_buf.float() * self.step_dt)
        details = {
            "first_event_code": self._event_code,
            "first_event_time_s": self._event_time,
            "collision_position": self._collision_position,
            "collision_substep": self._collision_substep,
            "collision_velocity": self._collision_velocity,
            "actuator": self.filtered_command,
        }
        if self.cfg.variant == "dcol":
            details["dynamic_positions_by_substep"] = self._dynamic_positions_by_substep
            details["dynamic_velocities_by_substep"] = self._dynamic_velocities_by_substep
        return details

    @torch.no_grad()
    def _apply_action(self):
        from isaaclab_navrl.zhang.dynamics import disturbance_acceleration

        if self._physical_substep:
            self._update_dynamic_obstacles()
            self._record_substep(self._compute_first_events())
        self._physical_substep += 1
        state = self.root_state()
        alpha = torch.exp(
            -self.zhang_parameters["pitch_ctl_delay"] * self.zhang_parameters["control_gain_scale"] * self.physics_dt
        )
        self.filtered_command.mul_(alpha).add_(self.command * (1 - alpha))
        forward = quat_apply(state[:, 3:7], state.new_tensor((1.0, 0.0, 0.0)).expand(self.num_envs, -1))
        target_dir = self.target_pos - state[:, :3]
        yaw_weight = float(getattr(self.cfg, "active_yaw_alignment", 0.0))
        if yaw_weight > 0.0:
            goal_xy = target_dir.clone()
            goal_xy[:, 2] = 0.0
            vel_xy = state[:, 7:10].clone()
            vel_xy[:, 2] = 0.0
            blended_xy = torch.nn.functional.normalize(goal_xy, dim=-1) + yaw_weight * vel_xy
            target_dir = torch.cat((blended_xy[:, :2], target_dir[:, 2:3]), dim=-1)
        desired_forward = torch.nn.functional.normalize(5.0 * forward + target_dir, dim=-1)
        yaw_alpha = torch.exp(-self.zhang_parameters["yaw_ctl_delay"] * self.physics_dt)
        yaw_vector = yaw_alpha * forward + (1.0 - yaw_alpha) * desired_forward
        yaw = torch.atan2(yaw_vector[:, 1], yaw_vector[:, 0]).unsqueeze(-1)
        self.controller.compute(state, state[:, 7:10], acceleration_world=self.filtered_command, yaw_world=yaw)
        self.controller.apply(
            self.drone,
            disturbance_acceleration(state[:, 7:10], state[:, 3:7], self.zhang_parameters),
        )
