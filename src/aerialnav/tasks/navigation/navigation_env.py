"""Shared Navigation world with backend-selectable stepping and two action semantics."""

from __future__ import annotations

import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    matrix_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
)
from isaaclab_tasks.utils.hydra import resolve_presets

from aerialnav.control import LeeRotorController
from aerialnav.diffnav.termination import (
    arrival_wait_timeout,
    latch_first_event,
    record_arrival,
    reset_first_event,
    terminal_outcomes,
    termination_mask,
)

from .curriculum import TrainingCurriculum, active_obstacle_indices
from .depth_history import compensated_depth_features, reproject_depth
from .geometry import TerrainDistance, nearest_obstacle_signed_distance
from .randomization import PolicyPerturbation
from .task_logic import DynamicObstacleMotion, goal_frame, navigation_terms, sample_navigation_route


class NavigationEnv(DirectRLEnv):
    """The original Navigation observation/reward task on Isaac Lab 3.0."""

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
            nominal_action_delay=1 if cfg.method == "diffnav" else 0,
        )
        self.mass_scale = torch.ones(self.num_envs, 1, device=self.device)
        self.inertia_scale = torch.ones_like(self.mass_scale)
        if self.curriculum is not None:
            cfg.num_dynamic_obstacles = self.curriculum.active_obstacles
        self.motion = DynamicObstacleMotion(
            cfg.num_dynamic_obstacles,
            cfg.terrain_seed,
            self.device,
            self.physics_dt if cfg.method == "diffnav" else self.step_dt,
            cfg.dynamic_speed_range,
        )
        self.dynamic_collection = (
            self.scene["dynamic_obstacles"]
            if cfg.num_dynamic_obstacles and self.drone.__backend_name__ != "newton"
            else None
        )
        if self.dynamic_collection is not None:
            instances = self.dynamic_collection.data.body_pose_w.torch.shape[0]
            expected = self.num_envs if self.drone.__backend_name__ == "newton" else 1
            if instances != expected:
                raise RuntimeError(f"Dynamic obstacle instances: expected {expected}, got {instances}.")
            self._dynamic_pose = self.dynamic_collection.data.body_pose_w.torch.clone()
            self.set_active_obstacles(cfg.num_dynamic_obstacles)
        elif self.drone.__backend_name__ == "newton":
            self.set_active_obstacles(cfg.num_dynamic_obstacles)
        if getattr(cfg, "geometry_backend", "native") == "curobo_v2":
            from aerialnav.reference.curobo_geometry import CuroboGeometry

            self.terrain_distance = CuroboGeometry(self.terrain_distance, self.motion.size)
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_speed = torch.zeros(self.num_envs, 1, device=self.device)
        self.target_dir = torch.zeros_like(self.target_pos)
        self.height_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.previous_velocity = torch.zeros_like(self.target_pos)
        self.command = torch.zeros_like(self.target_pos)
        self.executed_actions = torch.zeros_like(self.command)
        self.previous_actions = torch.zeros_like(self.command)
        self.action_difference = torch.zeros(self.num_envs, device=self.device)
        self.first_reach_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.arrived = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.arrive_time = torch.zeros(self.num_envs, device=self.device)
        self._episode_failure = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.nonfinite_events = torch.zeros((), dtype=torch.long, device=self.device)
        self.episode_id = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._reset_rng = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self._terms = None
        self._stepping = False

    def _create_controller(self):
        return LeeRotorController(self.num_envs, self.device, dt=self.physics_dt)

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
        if self.cfg.method == "diffnav":
            speed = self.diffnav_parameters["max_speed"][env_ids].clone()
        else:
            speed = 1 + 5 * torch.rand(count, 1, generator=self._reset_rng, device=self.device)
        self.target_speed[env_ids] = speed
        position, target = sample_navigation_route(
            speed, self._reset_rng, self.cfg.height_target_range if self.cfg.method == "diffnav" else (0.5, 2.5)
        )
        # Keep the uniform speed draw; reject occupied endpoints instead of shortening the route.
        for _ in range(64):
            clearance = self.nearest_obstacle_signed_distance(torch.cat((position, target))).reshape(2, count).amin(0)
            blocked = clearance <= 0.3
            if not blocked.any():
                break
            position[blocked], target[blocked] = sample_navigation_route(
                speed[blocked],
                self._reset_rng,
                self.cfg.height_target_range if self.cfg.method == "diffnav" else (0.5, 2.5),
            )
        else:
            raise RuntimeError("Could not sample collision-free 18v route endpoints after 64 attempts")
        self.target_pos[env_ids] = target
        direction = target - position
        self.target_dir[env_ids] = direction
        zero = torch.zeros(count, device=self.device)
        quaternion = quat_from_euler_xyz(zero, zero, torch.atan2(direction[:, 1], direction[:, 0]))
        self.drone.write_root_pose_to_sim_index(root_pose=torch.cat((position, quaternion), dim=-1), env_ids=env_ids)
        self.drone.write_root_velocity_to_sim_index(root_velocity=position.new_zeros(count, 6), env_ids=env_ids)
        self.drone.write_joint_position_to_sim_index(
            position=self.drone.data.default_joint_pos.torch[env_ids], env_ids=env_ids
        )
        self.controller.reset(env_ids)
        self.controller.apply(self.drone)
        self.height_range[env_ids] = torch.stack(
            (torch.minimum(position[:, 2], target[:, 2]), torch.maximum(position[:, 2], target[:, 2])), dim=-1
        )
        self.previous_velocity[env_ids] = 0
        self.command[env_ids] = 0
        self.previous_actions[env_ids] = 0
        self.first_reach_step[env_ids] = 0
        self.arrived[env_ids] = False
        self.arrive_time[env_ids] = 0
        self._episode_failure[env_ids] = 0
        self.episode_return[env_ids] = 0
        self.episode_id[env_ids] += 1
        self._terms = None

    def reset(self, *args, **kwargs):
        self._stepping = False
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
            self.physics_dt if self.cfg.method == "diffnav" else self.step_dt,
            self.cfg.dynamic_speed_range,
        )
        self.cfg.num_dynamic_obstacles = count
        if getattr(getattr(self, "drone", None), "__backend_name__", None) == "newton":
            self._bind_shared_dynamic_geometry()
        if self.dynamic_collection is not None:
            self._dynamic_pose[:, :, :3] = self._dynamic_pose.new_tensor((0.0, 0.0, -1000.0))
        self._write_dynamic_obstacles()

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
        self._write_dynamic_obstacles()

    def _write_dynamic_obstacles(self):
        if self.dynamic_collection is None:
            return
        self._dynamic_pose[:, self._active_obstacle_ids, :3] = self.motion.position
        # Kinematic poses are authoritative; visual motion never changes the collision rule.
        self.dynamic_collection.write_body_pose_to_sim_index(body_poses=self._dynamic_pose)
        self.dynamic_collection.write_data_to_sim()

    def nearest_obstacle_signed_distance(self, query):
        return nearest_obstacle_signed_distance(
            query,
            self.terrain_distance.signed_distance(query),
            self.motion.position,
            self.motion.size,
            dynamic_backend=getattr(self.terrain_distance, "dynamic_signed_distance", None),
        )

    def _compute_first_events(self):
        from aerialnav.diffnav.termination import first_events

        state = self.root_state()
        radius = 0.3  # Shared task sphere; training safety margins are separate.
        return first_events(
            state,
            self.target_pos,
            self.nearest_obstacle_signed_distance(state[:, :3]) - radius,
            height_bounds_reset=getattr(self.cfg, "height_bounds_reset", True),
        )

    def _compute_terms(self):
        state = self.root_state()
        hits = self.lidar.data.ray_hits_w.torch
        origin = self.lidar.data.pos_w.torch
        distances = (hits - origin[:, None]).norm(dim=-1).clamp_max(4).reshape(self.num_envs, 1, 36, 4)
        self._terms = navigation_terms(
            state[:, :3],
            state[:, 7:10],
            self.target_pos,
            self.target_dir,
            distances,
            self.motion,
            self.previous_velocity,
            self.height_range,
        )
        self._terms.update(self._compute_first_events())
        return self._terms

    def _get_dones(self):
        # Match Navigation's order: PhysX -> global obstacle motion -> observations.
        self._update_dynamic_obstacles()
        terms = self._finish_policy_terms(self._compute_terms())
        self.nonfinite_events += terms["invalid"].sum()
        terminated = self._termination_mask(terms)
        elapsed_s = self.episode_length_buf.float() * self.step_dt
        record_arrival(self.arrived, self.arrive_time, terms["reached"], elapsed_s)
        wait_expired = arrival_wait_timeout(
            terms["reached"], self.arrive_time, elapsed_s, self.cfg.wait_before_truncate
        )
        episode_timeout = self.episode_length_buf >= self.max_episode_length
        timed_out = episode_timeout | wait_expired
        outcome = terminal_outcomes(terms, terminated | timed_out, self._episode_failure)
        first = self._first_arrival_mask(terms, terminated)
        self.first_reach_step.copy_(torch.where(first, self.episode_length_buf, self.first_reach_step))
        state = self.root_state()
        # These values belong to the completed transition, before same-step autoreset.
        self.extras["transition"] = {
            "position": state[:, :3].clone(),
            "quaternion": state[:, 3:7].clone(),
            "velocity": state[:, 7:10].clone(),
            "angular_velocity": state[:, 10:13].clone(),
            "target": self.target_pos.clone(),
            "episode_id": self.episode_id.clone(),
            "height_target": self.target_pos[:, 2].clone(),
            "height_range": self.height_range.clone(),
            "step": self.episode_length_buf.clone(),
            "time_s": self.episode_length_buf.float() * self.step_dt,
            "reached": terms["reached"].clone(),
            "collision": terms["collision"].clone(),
            "bounds": terms["bounds"].clone(),
            "invalid": terms["invalid"].clone(),
            "timeout": timed_out.clone(),
            "terminated": terminated.clone(),
            "arrived": self.arrived.clone(),
            "arrive_time": self.arrive_time.clone(),
            "arrival_wait_timeout": wait_expired.clone(),
            "episode_timeout": episode_timeout.clone(),
            "terminal_event_code": outcome.clone(),
            "executed_action": self.executed_actions.clone(),
            "action_delay_steps": self.executed_action_lags.clone(),
        }
        if self.cfg.method == "diffnav" and self.cfg.capture_terminal_observation:
            # Capture before DirectRLEnv autoresets sensors, targets and randomization.
            transition = self.extras["transition"]
            transition["depth_features"] = (
                self.scene["depth_camera"].data.output["policy_depth"].torch.permute(0, 3, 1, 2).clone()
            )
            transition["input_delay_steps"] = self.perturbation.observation_buffer.time_lags.clone()
            transition["critic_parameters"] = {k: v.detach().clone() for k, v in self.diffnav_parameters.items()}
        return terminated, timed_out

    def _termination_mask(self, terms):
        return terms["collision"] | terms["bounds"] | terms["invalid"]

    def _first_arrival_mask(self, terms, terminated):
        return terms["reached"] & ~terminated & (self.first_reach_step == 0)

    def _finish_policy_terms(self, terms):
        return terms

    def _get_rewards(self):
        reward = self._terms["reward"]
        cfg = self.cfg.sim_to_real
        if self.cfg.method == "ppo" and cfg.enabled and cfg.action_difference:
            reward = reward - cfg.action_difference_weight * self.action_difference
        self.episode_return += reward
        self.extras["transition"]["episode_return"] = self.episode_return.clone()
        self.previous_velocity.copy_(self.drone.data.root_lin_vel_w.torch)
        return reward

    def _get_observations(self):
        def sample():
            terms = self._compute_terms()
            if self.cfg.method == "ppo" and self.perturbation.noisy:
                state = self.root_state()
                position, velocity, _ = self.perturbation.state(state[:, :3], state[:, 7:10], state[:, 3:7])
                distance = self.perturbation.normal(4.0 - terms["lidar"], self.cfg.sim_to_real.lidar_std).clamp(0, 4)
                terms = navigation_terms(
                    position,
                    velocity,
                    self.target_pos,
                    self.target_dir,
                    distance,
                    self.motion,
                    self.previous_velocity,
                    self.height_range,
                )
            return torch.cat((terms["state"], terms["lidar"].flatten(1), terms["dynamic_obstacle"].flatten(1)), dim=-1)

        if self.cfg.method == "diffnav":
            return {"policy": sample()}
        return {"policy": self.perturbation.observe(self.common_step_counter, sample)}

    def get_diffnav_observation(self):
        state = self.root_state()
        camera = self.scene["depth_camera"]
        features = camera.data.output["policy_depth"].torch
        depth = features.permute(0, 3, 1, 2).clone()
        measured_depth = camera.data.output["distance_to_image_plane"].torch.permute(0, 3, 1, 2)

        def sample():
            position, velocity, quaternion = self.perturbation.state(state[:, :3], state[:, 7:10], state[:, 3:7])
            return torch.cat(
                (position, velocity, quaternion, self.target_pos, depth.flatten(1), measured_depth.flatten(1)), dim=-1
            )

        policy = self.perturbation.observe(self.common_step_counter, sample)
        # The fast odometry channel models an onboard state estimate, with the
        # configured measurement noise; only the delivered depth frame is delayed.
        current = self.perturbation.latest_observation
        age = self.perturbation.observation_age_s
        policy_depth = policy[:, 13:205].reshape(self.num_envs, 1, 12, 16)
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
                policy[:, 205:].reshape_as(measured_depth),
                camera.data.intrinsic_matrices.torch,
                source_rotation,
                source_position,
                target_rotation,
                target_position,
            )
            policy_depth = torch.where(
                (age > 0)[:, None, None, None], compensated_depth_features(warped, known), policy_depth
            )
            known_fraction = torch.where(age > 0, known.float().mean((1, 2, 3)), known_fraction)
        return {
            "depth_features": depth,
            "position": state[:, :3],
            "quaternion": state[:, 3:7],
            "velocity": state[:, 7:10],
            "target": self.target_pos.clone(),
            "episode_id": self.episode_id.clone(),
            "height_target": self.target_pos[:, 2].clone(),
            "height_range": self.height_range.clone(),
            "policy_position": current[:, :3],
            "policy_velocity": current[:, 3:6],
            "policy_quaternion": current[:, 6:10],
            "policy_target": current[:, 10:13],
            "policy_depth_features": policy_depth,
            "depth_age_s": age.clone(),
            "depth_known_fraction": known_fraction,
            "input_delay_steps": self.perturbation.observation_buffer.time_lags.clone(),
        }


class DiffNavEnv(NavigationEnv):
    """The same navigation world with unrestricted world-acceleration actions."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._configure_controller()
        self._event_code = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
        self._event_time = torch.zeros(self.num_envs, device=self.device)
        self._substep_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._collision_position = torch.zeros(self.num_envs, 3, device=self.device)
        self._collision_velocity = torch.zeros_like(self._collision_position)
        self._collision_substep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._dynamic_positions_by_substep = (
            self.motion.position.new_zeros((self.cfg.decimation, *self.motion.position.shape))
            if self.cfg.variant == "terminal"
            else None
        )
        self._dynamic_velocities_by_substep = (
            torch.zeros_like(self._dynamic_positions_by_substep) if self.cfg.variant == "terminal" else None
        )
        self._substep_bounds = torch.zeros_like(self._substep_collision)
        self._substep_invalid = torch.zeros_like(self._substep_collision)
        self.filtered_command = torch.zeros(self.num_envs, 3, device=self.device)

    def _configure_controller(self):
        from aerialnav.diffnav.controller import GraphedLeeRotorController

        self._rescale_motor_response()
        state = self.root_state()
        self.controller = GraphedLeeRotorController(
            self.controller, (state, state[:, 7:10], self.command, self.command[:, :1])
        )

    def _rescale_motor_response(self):
        if self.physics_dt != 0.01:
            # Preserve the baseline motor time constant when changing its sample period.
            for name in ("tau_up", "tau_down"):
                tau = self.controller.rotor_parameters[name]
                tau.copy_(1 - (1 - tau).pow(self.physics_dt / 0.01))

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.filtered_command[env_ids] = 0
        reset_first_event(self._event_code, self._event_time, env_ids)
        self._substep_collision[env_ids] = False
        self._collision_position[env_ids] = 0
        self._collision_velocity[env_ids] = 0
        self._collision_substep[env_ids] = 0
        self._substep_bounds[env_ids] = False
        self._substep_invalid[env_ids] = False

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

    def apply_diffnav_camera_randomization(self, env_ids):
        camera = self.scene["depth_camera"]
        pattern = camera.cfg.pattern_cfg
        scale = self.diffnav_parameters["camera_fov_scale"][env_ids].reshape(-1)
        matrices = torch.zeros(len(env_ids), 3, 3, device=self.device)
        base_fx = pattern.width * pattern.focal_length / pattern.horizontal_aperture
        base_fy = pattern.height * pattern.focal_length / pattern.vertical_aperture
        matrices[:, 0, 0], matrices[:, 1, 1] = base_fx / scale, base_fy / scale
        matrices[:, 0, 2] = pattern.horizontal_aperture_offset * base_fx + pattern.width / 2
        matrices[:, 1, 2] = pattern.vertical_aperture_offset * base_fy + pattern.height / 2
        matrices[:, 2, 2] = 1
        camera.set_intrinsic_matrices(matrices, env_ids=env_ids)
        camera.set_mount_pitch(self.diffnav_parameters["camera_pitch"][env_ids].reshape(-1), env_ids)

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
        if self.cfg.variant == "terminal":
            shape = (self.cfg.decimation, *self.motion.position.shape)
            if self._dynamic_positions_by_substep.shape != shape:
                # The legacy curriculum can change the shared obstacle count.
                self._dynamic_positions_by_substep = self.motion.position.new_zeros(shape)
                self._dynamic_velocities_by_substep = torch.zeros_like(self._dynamic_positions_by_substep)

    def _record_substep(self, terms):
        # A 10 Hz policy can cross the collision band between observations.
        # Use the same sphere clearance at every completed PhysX step.
        elapsed_s = self._policy_start_time + self._physical_substep * self.physics_dt
        record_arrival(self.arrived, self.arrive_time, terms["reached"], elapsed_s)
        latch_first_event(self._event_code, self._event_time, terms, elapsed_s)
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
        if self.cfg.variant == "terminal":
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

    def _get_dones(self):
        terminated, timeout = super()._get_dones()
        timed_out_first = (self._event_code == 0) & timeout
        self._event_code[timed_out_first] = 5
        self._event_time.copy_(
            torch.where(timed_out_first, self.episode_length_buf.float() * self.step_dt, self._event_time)
        )
        self.extras["transition"]["first_event_code"] = self._event_code.clone()
        self.extras["transition"]["first_event_time_s"] = self._event_time.clone()
        self.extras["transition"]["collision_position"] = self._collision_position.clone()
        self.extras["transition"]["collision_substep"] = self._collision_substep.clone()
        self.extras["transition"]["collision_velocity"] = self._collision_velocity.clone()
        if self.cfg.variant == "terminal":
            self.extras["transition"]["dynamic_positions_by_substep"] = self._dynamic_positions_by_substep.clone()
            self.extras["transition"]["dynamic_velocities_by_substep"] = self._dynamic_velocities_by_substep.clone()
        self.extras["transition"]["actuator"] = self.filtered_command.clone()
        callback = getattr(self, "pre_reset_loss_callback", None)
        if callback is not None:
            self.pre_reset_loss_callback = None
            callback(self.extras["transition"])
        return terminated, timeout

    @torch.no_grad()
    def _apply_action(self):
        from aerialnav.diffnav.dynamics import disturbance_acceleration

        if self._physical_substep:
            self._update_dynamic_obstacles()
            self._record_substep(self._compute_first_events())
        self._physical_substep += 1
        state = self.root_state()
        alpha = torch.exp(
            -self.diffnav_parameters["pitch_ctl_delay"]
            * self.diffnav_parameters["control_gain_scale"]
            * self.physics_dt
        )
        self.filtered_command.mul_(alpha).add_(self.command * (1 - alpha))
        forward = quat_apply(state[:, 3:7], state.new_tensor((1.0, 0.0, 0.0)).expand(self.num_envs, -1))
        desired_forward = torch.nn.functional.normalize(5.0 * forward + self.target_pos - state[:, :3], dim=-1)
        yaw_alpha = torch.exp(-self.diffnav_parameters["yaw_ctl_delay"] * self.physics_dt)
        yaw_vector = yaw_alpha * forward + (1.0 - yaw_alpha) * desired_forward
        yaw = torch.atan2(yaw_vector[:, 1], yaw_vector[:, 0]).unsqueeze(-1)
        self.controller.compute(state, state[:, 7:10], acceleration_world=self.filtered_command, yaw_world=yaw)
        self.controller.apply(
            self.drone,
            disturbance_acceleration(state[:, 7:10], state[:, 3:7], self.diffnav_parameters),
        )
