"""Shared NavRL world with official PhysX stepping and two action semantics."""

from __future__ import annotations

import numpy as np
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from isaaclab_navrl.control import LeeRotorController

from .curriculum import TrainingCurriculum, active_obstacle_indices
from .randomization import PolicyPerturbation
from .task_logic import DynamicObstacleMotion, goal_frame, navigation_terms


class NavRLEnv(DirectRLEnv):
    """The original NavRL observation/reward task on Isaac Lab 3.0."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        if cfg.evaluation:
            cfg.sim_to_real.enabled = False
        cfg.configure_scene()
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.drone = self.scene["drone"]
        self.lidar = self.scene["lidar"]
        self.controller = LeeRotorController(self.num_envs, self.device, dt=self.physics_dt)
        self.curriculum = TrainingCurriculum(self.num_envs, self.device, stage=cfg.training_stage) if cfg.training_curriculum else None
        self.perturbation = PolicyPerturbation(
            cfg.sim_to_real, self.num_envs, self.device, cfg.seed,
            nominal_action_delay=1 if cfg.method == "zhang" else 0,
        )
        self.mass_scale = torch.ones(self.num_envs, 1, device=self.device)
        self.inertia_scale = torch.ones_like(self.mass_scale)
        if self.curriculum is not None:
            cfg.num_dynamic_obstacles = self.curriculum.active_obstacles
        self.motion = DynamicObstacleMotion(
            cfg.num_dynamic_obstacles, cfg.terrain_seed, self.device,
            self.physics_dt if cfg.method == "zhang" else self.step_dt, cfg.dynamic_speed_range,
        )
        self.dynamic_collection = self.scene["dynamic_obstacles"] if cfg.num_dynamic_obstacles else None
        if self.dynamic_collection is not None:
            if self.dynamic_collection.data.body_pose_w.torch.shape[0] != 1:
                raise RuntimeError("The global dynamic obstacles were unexpectedly replicated.")
            self._dynamic_pose = self.dynamic_collection.data.body_pose_w.torch.clone()
            self.set_active_obstacles(cfg.num_dynamic_obstacles)
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_dir = torch.zeros_like(self.target_pos)
        self.height_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.previous_velocity = torch.zeros_like(self.target_pos)
        self.command = torch.zeros_like(self.target_pos)
        self.executed_actions = torch.zeros_like(self.command)
        self.previous_actions = torch.zeros_like(self.command)
        self.action_difference = torch.zeros(self.num_envs, device=self.device)
        self.first_reach_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.nonfinite_events = torch.zeros((), dtype=torch.long, device=self.device)
        self.episode_id = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._reset_rng = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self._side_rng = np.random.RandomState(cfg.seed)
        self._terms = None
        self._stepping = False

    def _setup_scene(self):
        self.scene.filter_collisions(global_prim_paths=["/World/ground", "/World/defaultGroundPlane"])

    def root_state(self):
        """World position, XYZW orientation, linear and angular velocity."""
        data = self.drone.data
        return torch.cat(
            (data.root_pos_w.torch, data.root_quat_w.torch, data.root_lin_vel_w.torch, data.root_ang_vel_w.torch),
            dim=-1,
        )

    def _sample_border(self, count):
        masks = self.target_pos.new_tensor(((1, 0, 1), (1, 0, 1), (0, 1, 1), (0, 1, 1)))
        shifts = self.target_pos.new_tensor(((0, 24, 0), (0, -24, 0), (24, 0, 0), (-24, 0, 0)))
        sides = self._side_rng.randint(0, 4, count)
        sample = torch.rand(count, 3, generator=self._reset_rng, device=self.device) * 48 - 24
        sample[:, 2] = 0.5 + 2 * torch.rand(count, generator=self._reset_rng, device=self.device)
        return sample * masks[sides] + shifts[sides]

    def _reset_idx(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.int32)
        if self._stepping:
            self.extras["log"] = {
                "Episode/return": self.episode_return[env_ids].mean(),
                "Episode/length": self.episode_length_buf[env_ids].float().mean(),
                "Episode/first_arrival": (self.first_reach_step[env_ids] > 0).float().mean(),
                "Episode/reach_goal": self._terms["reached"][env_ids].float().mean(),
                "Episode/collision": self._terms["collision"][env_ids].float().mean(),
            }
        # No shared-world object or trajectory is reset here.
        super()._reset_idx(env_ids)
        count = len(env_ids)
        if self.cfg.evaluation:
            position = self.target_pos.new_zeros(count, 3)
            position[:, 0] = (env_ids.float() / self.num_envs - 0.5) * 32
            position[:, 1], position[:, 2] = 24.0, 2.0
            target = position.clone()
            target[:, 0] = torch.linspace(-16, 16, self.num_envs, device=self.device)[env_ids]
            target[:, 1] = -24.0
        else:
            target = self._sample_border(count)
            position = self._sample_border(count)
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
        result = super().step(actions)
        if self.curriculum is not None:
            self.curriculum.observe(result[-1]["transition"])
        return result

    def set_active_obstacles(self, count: int):
        """Change the shared difficulty without respawning USD or copying worlds."""
        self._active_obstacle_ids = torch.tensor(
            active_obstacle_indices(count, self.cfg.dynamic_capacity()), device=self.device, dtype=torch.long
        )
        self.motion = DynamicObstacleMotion(
            count, self.cfg.terrain_seed, self.device,
            self.physics_dt if self.cfg.method == "zhang" else self.step_dt, self.cfg.dynamic_speed_range,
        )
        self.cfg.num_dynamic_obstacles = count
        self._dynamic_pose[0, :, :3] = self._dynamic_pose.new_tensor((0.0, 0.0, -1000.0))
        self._write_dynamic_obstacles()

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
        self._dynamic_pose[0, self._active_obstacle_ids, :3] = self.motion.position
        # Kinematic poses are authoritative; visual motion never changes the collision rule.
        self.dynamic_collection.write_body_pose_to_sim_index(body_poses=self._dynamic_pose)
        self.dynamic_collection.write_data_to_sim()

    def _compute_terms(self):
        state = self.root_state()
        hits = self.lidar.data.ray_hits_w.torch
        origin = self.lidar.data.pos_w.torch
        distances = (hits - origin[:, None]).norm(dim=-1).clamp_max(4).reshape(self.num_envs, 1, 36, 4)
        self._terms = navigation_terms(
            state[:, :3], state[:, 7:10], self.target_pos, self.target_dir, distances,
            self.motion, self.previous_velocity, self.height_range,
        )
        self._terms["invalid"] = ~torch.isfinite(state).all(dim=-1)
        return self._terms

    def _get_dones(self):
        # Match NavRL's order: PhysX -> global obstacle motion -> observations.
        self._update_dynamic_obstacles()
        terms = self._finish_policy_terms(self._compute_terms())
        self.nonfinite_events += terms["invalid"].sum()
        terminated = terms["collision"] | terms["bounds"] | terms["invalid"]
        timed_out = self.episode_length_buf >= self.max_episode_length
        first = terms["reached"] & ~terminated & (self.first_reach_step == 0)
        self.first_reach_step[first] = self.episode_length_buf[first]
        state = self.root_state()
        # These values belong to the completed transition, before same-step autoreset.
        self.extras["transition"] = {
            "position": state[:, :3].clone(), "quaternion": state[:, 3:7].clone(),
            "velocity": state[:, 7:10].clone(), "angular_velocity": state[:, 10:13].clone(),
            "target": self.target_pos.clone(), "episode_id": self.episode_id.clone(),
            "step": self.episode_length_buf.clone(),
            "time_s": self.episode_length_buf.float() * self.step_dt,
            "reached": terms["reached"].clone(), "collision": terms["collision"].clone(),
            "bounds": terms["bounds"].clone(), "invalid": terms["invalid"].clone(),
            "timeout": timed_out.clone(), "terminated": terminated.clone(),
            "executed_action": self.executed_actions.clone(),
            "action_delay_steps": self.executed_action_lags.clone(),
        }
        return terminated, timed_out

    def _finish_policy_terms(self, terms):
        return terms

    def _get_rewards(self):
        reward = self._terms["reward"]
        cfg = self.cfg.sim_to_real
        if self.cfg.method == "navrl" and cfg.enabled and cfg.action_difference:
            reward = reward - cfg.action_difference_weight * self.action_difference
        self.episode_return += reward
        self.extras["transition"]["episode_return"] = self.episode_return.clone()
        self.previous_velocity.copy_(self.drone.data.root_lin_vel_w.torch)
        return reward

    def _get_observations(self):
        def sample():
            terms = self._compute_terms()
            if self.cfg.method == "navrl" and self.perturbation.noisy:
                state = self.root_state()
                position, velocity, _ = self.perturbation.state(state[:, :3], state[:, 7:10], state[:, 3:7])
                distance = self.perturbation.normal(4.0 - terms["lidar"], self.cfg.sim_to_real.lidar_std).clamp(0, 4)
                terms = navigation_terms(position, velocity, self.target_pos, self.target_dir, distance,
                                         self.motion, self.previous_velocity, self.height_range)
            return torch.cat((terms["state"], terms["lidar"].flatten(1),
                              terms["dynamic_obstacle"].flatten(1)), dim=-1)

        if self.cfg.method == "zhang":
            return {"policy": sample()}
        return {"policy": self.perturbation.observe(self.common_step_counter, sample)}

    def get_zhang_observation(self):
        state = self.root_state()
        features = self.scene["depth_camera"].data.output["policy_depth"].torch
        depth = features.permute(0, 3, 1, 2).clone()

        def sample():
            position, velocity, quaternion = self.perturbation.state(state[:, :3], state[:, 7:10], state[:, 3:7])
            return torch.cat((position, velocity, quaternion, self.target_pos, depth.flatten(1)), dim=-1)

        policy = self.perturbation.observe(self.common_step_counter, sample)
        return {
            "depth_features": depth, "position": state[:, :3],
            "quaternion": state[:, 3:7], "velocity": state[:, 7:10],
            "target": self.target_pos.clone(), "episode_id": self.episode_id.clone(),
            "policy_position": policy[:, :3], "policy_velocity": policy[:, 3:6],
            "policy_quaternion": policy[:, 6:10], "policy_target": policy[:, 10:13],
            "policy_depth_features": policy[:, 13:].reshape(self.num_envs, 1, 12, 16),
            "input_delay_steps": self.perturbation.observation_buffer.time_lags.clone(),
        }


class ZhangNavRLEnv(NavRLEnv):
    """The same navigation world with unrestricted world-acceleration actions."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        from isaaclab_navrl.zhang.controller import GraphedLeeRotorController

        from .geometry import TerrainDistance

        state = self.root_state()
        self.controller = GraphedLeeRotorController(
            self.controller, (state, state[:, 7:10], self.command, self.command[:, :1])
        )
        self.terrain_distance = TerrainDistance.from_stage(self.sim.stage, self.device)
        self._event_code = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
        self._event_time = torch.zeros(self.num_envs, device=self.device)
        self._substep_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._substep_bounds = torch.zeros_like(self._substep_collision)
        self._substep_invalid = torch.zeros_like(self._substep_collision)
        self.filtered_command = torch.zeros(self.num_envs, 3, device=self.device)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.filtered_command[env_ids] = 0

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

    def nearest_obstacle_vector(self, query):
        from .geometry import nearest_obstacle_vector

        return nearest_obstacle_vector(
            query, self.terrain_distance.closest_points(query), self.motion.position, self.motion.size
        )

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
        self._event_code.zero_()
        self._substep_collision.zero_()
        self._substep_bounds.zero_()
        self._substep_invalid.zero_()

    def _compute_first_events(self):
        from isaaclab_navrl.zhang.termination import first_events

        state = self.root_state()
        hits = self.lidar.data.ray_hits_w.torch
        origin = self.lidar.data.pos_w.torch
        distances = (hits - origin[:, None]).norm(dim=-1).clamp_max(4).reshape(self.num_envs, 1, 36, 4)
        return first_events(state, self.target_pos, distances, self.motion)

    def _record_substep(self, terms):
        # A 10 Hz policy can cross the collision band between observations.
        # Evaluate the unchanged NavRL rules at every completed PhysX step.
        code = torch.zeros_like(self._event_code)
        code[terms["reached"]] = 1
        code[terms["bounds"]] = 3
        code[terms["collision"]] = 2
        code[terms["invalid"]] = 4
        first = (self._event_code == 0) & (code != 0)
        self._event_code[first] = code[first]
        self._event_time[first] = self._policy_start_time[first] + self._physical_substep * self.physics_dt
        arrival = first & (code == 1) & (self.first_reach_step == 0)
        self.first_reach_step[arrival] = (self._policy_start_time[arrival] / self.step_dt).round().long() + 1
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
        self._event_time[timed_out_first] = self.episode_length_buf[timed_out_first].float() * self.step_dt
        self.extras["transition"]["first_event_code"] = self._event_code.clone()
        self.extras["transition"]["first_event_time_s"] = self._event_time.clone()
        self.extras["transition"]["actuator"] = self.filtered_command.clone()
        return terminated, timeout

    @torch.no_grad()
    def _apply_action(self):
        from isaaclab_navrl.zhang.dynamics import disturbance_acceleration

        if self._physical_substep:
            self._update_dynamic_obstacles()
            self._record_substep(self._compute_first_events())
        self._physical_substep += 1
        state = self.root_state()
        alpha = torch.exp(-self.zhang_parameters["pitch_ctl_delay"]
                          * self.zhang_parameters["control_gain_scale"] * self.physics_dt)
        self.filtered_command.mul_(alpha).add_(self.command * (1 - alpha))
        forward = quat_apply(state[:, 3:7], state.new_tensor((1.0, 0.0, 0.0)).expand(self.num_envs, -1))
        desired_forward = torch.nn.functional.normalize(5.0 * forward + self.target_pos - state[:, :3], dim=-1)
        yaw_alpha = torch.exp(-self.zhang_parameters["yaw_ctl_delay"] * self.physics_dt)
        yaw_vector = yaw_alpha * forward + (1.0 - yaw_alpha) * desired_forward
        yaw = torch.atan2(yaw_vector[:, 1], yaw_vector[:, 0]).unsqueeze(-1)
        self.controller.compute(
            state, state[:, 7:10], acceleration_world=self.filtered_command, yaw_world=yaw
        )
        self.controller.apply(self.drone)
        self.drone.set_disturbance_acceleration(
            disturbance_acceleration(state[:, 7:10], state[:, 3:7], self.zhang_parameters)
        )
