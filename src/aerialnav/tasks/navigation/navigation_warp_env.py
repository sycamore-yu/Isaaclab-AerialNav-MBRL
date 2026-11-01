"""Official Warp frontend for the shared navigation tasks.

Lee/motor dynamics, navigation observations/rewards, event predicates and step
counters use fixed Warp buffers and CUDA graphs. Simulator IO, sensors, random
sampling, delay histories, partial resets and episode reporting stay outside
capture. Keeping these boundaries explicit preserves the Torch task's reset
and interval events, which DirectRLEnvWarp does not currently execute itself.
"""

from __future__ import annotations

from functools import wraps

import torch
import warp as wp
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.utils.video_recorder import VideoRecorder
from isaaclab_experimental.envs import DirectRLEnvWarp
from isaaclab_experimental.envs.interactive_scene_warp import InteractiveSceneWarp
from isaaclab_experimental.utils.warp_graph_cache import WarpGraphCache

from aerialnav.training import MAP_SIZE
from aerialnav.warp_control import WarpLeeRotorController

from .navigation_env import DiffNavEnv, NavigationEnv
from .scene import SharedNavigationScene
from .warp_task_kernels import WarpFirstEvents, WarpNavigationTerms


def _on_warp_stream(method):
    """Share one registered non-default CUDA stream across Torch and Warp."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if not str(self.device).startswith("cuda"):
            return method(self, *args, **kwargs)
        caller = torch.cuda.current_stream(self.device)
        if caller.cuda_stream == self._torch_warp_stream.cuda_stream:
            with wp.ScopedStream(self._warp_stream):
                return method(self, *args, **kwargs)
        self._torch_warp_stream.wait_stream(caller)
        try:
            with torch.cuda.stream(self._torch_warp_stream), wp.ScopedStream(self._warp_stream):
                return method(self, *args, **kwargs)
        finally:
            caller.wait_stream(self._torch_warp_stream)

    return wrapped


@wp.kernel(enable_backward=False)
def _increment_episode(length: wp.array(dtype=wp.int64)):
    env = wp.tid()
    length[env] += wp.int64(1)


@wp.kernel(enable_backward=False)
def _combine_resets(
    terminated: wp.array(dtype=wp.bool),
    timeout: wp.array(dtype=wp.bool),
    reset: wp.array(dtype=wp.bool),
):
    env = wp.tid()
    reset[env] = terminated[env] or timeout[env]


class SharedNavigationWarpScene(SharedNavigationScene, InteractiveSceneWarp):
    """Index-based episode resets preserve the shared moving obstacle world.

    Reset sampling and sensor invalidation intentionally run outside capture.
    The inherited shared reset only touches selected drones and their sensors.
    """


class _GraphedWarpController(WarpLeeRotorController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graphs = WarpGraphCache()

    @torch.no_grad()
    def compute(self, root_state_xyzw, velocity_world, acceleration_world=None, yaw_world=None):
        self._copy_inputs(root_state_xyzw, velocity_world, acceleration_world, yaw_world)
        self.advance_rng()
        arguments = {"use_acceleration": acceleration_world is not None, "use_yaw": yaw_world is not None}
        if str(self.device).startswith("cuda"):
            stage = f"control_{arguments['use_acceleration']}_{arguments['use_yaw']}"
            self.graphs.capture_or_replay(stage, self.launch, kwargs=arguments)
        else:
            self.launch(**arguments)
        return self.commands


class _NavigationWarpLoop(DirectRLEnvWarp):
    """Warp stepping with the task's complete event and partial-reset contract."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        if str(self.device).startswith("cuda"):
            # CUDA graph capture cannot begin on the legacy default stream.
            self._warp_stream = wp.Stream(self.device)
            self._torch_warp_stream = wp.stream_to_torch(self._warp_stream)
        # Public episode snapshots and trainer bookkeeping use int64. The
        # backing storage remains Warp-owned and is incremented by a kernel.
        self._episode_length_buf_wp = wp.zeros(self.num_envs, dtype=wp.int64, device=self.device)
        self._episode_length_buf_torch = wp.to_torch(self._episode_length_buf_wp)
        self.torch_episode_length_buf = self.episode_length_buf
        self.torch_reset_terminated = wp.to_torch(self.reset_terminated)
        self.torch_reset_time_outs = wp.to_torch(self.reset_time_outs)
        self._torch_reset_buf = wp.to_torch(self.reset_buf)
        self._reward_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.torch_reward_buf = wp.to_torch(self._reward_wp)
        self.render_enabled = True
        self.has_rtx_sensors = self.sim.get_setting("/isaaclab/render/rtx_sensors")
        self.video_recorders = [VideoRecorder(recorder, self) for recorder in self.cfg.video_recorders]

    @_on_warp_stream
    def _run_warp(self, stage, function):
        if str(self.device).startswith("cuda"):
            return self._graph_cache.capture_or_replay(stage, function)
        return function()

    def _increment_episode(self):
        wp.launch(_increment_episode, dim=self.num_envs, inputs=[self._episode_length_buf_wp], device=self.device)

    def _combine_resets(self):
        wp.launch(
            _combine_resets,
            dim=self.num_envs,
            inputs=[self.reset_terminated, self.reset_time_outs],
            outputs=[self.reset_buf],
            device=self.device,
        )

    def _reset_idx(self, env_ids):
        # This host boundary retains mass/inertia/policy event randomization,
        # noise history reset, and only the selected drones' sensor state.
        DirectRLEnv._reset_idx(self, env_ids)

    @_on_warp_stream
    def reset(self, seed=None, options=None):
        observation, extras = DirectRLEnv.reset(self, seed=seed, options=options)
        self.torch_obs_buf = observation["policy"]
        return observation, extras

    @torch.no_grad()
    @_on_warp_stream
    def step(self, action):
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)
        self._pre_physics_step(action)
        is_rendering = self.sim.is_rendering
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            # Public composer, actuators and sensor updates are outside graphs,
            # matching the official Warp frontend's simulator IO boundary.
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.physics_dt)

        self._run_warp("episode_counter", self._increment_episode)
        self.common_step_counter += 1
        terminated, timeout = self._get_dones()
        self.torch_reset_terminated.copy_(terminated)
        self.torch_reset_time_outs.copy_(timeout)
        self._run_warp("reset_flags", self._combine_resets)
        # Copy before observation kernels overwrite their fixed reward views.
        self.torch_reward_buf.copy_(self._get_rewards())
        reset_ids = self._torch_reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        if len(reset_ids):
            if self.cfg.compute_final_obs:
                self.extras["final_obs"] = self._compute_observations()
            self._reset_idx(reset_ids)
            if self.render_enabled and is_rendering and self.has_rtx_sensors:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        if self.sim.consume_reset_request():
            manual_ids = (~self._torch_reset_buf).nonzero(as_tuple=False).squeeze(-1).int()
            if len(manual_ids):
                self.torch_reset_terminated[manual_ids] = True
                self._reset_idx(manual_ids)
        if self.cfg.events and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        for recorder in self.video_recorders:
            recorder.step()
        self.obs_buf = self._compute_observations()
        self.torch_obs_buf = self.obs_buf["policy"]
        return (
            self.obs_buf,
            self.torch_reward_buf,
            self.torch_reset_terminated,
            self.torch_reset_time_outs,
            self.extras,
        )

    def close(self):
        for recorder in getattr(self, "video_recorders", []):
            recorder.close()
        super().close()


class _WarpNavigation:
    """Fixed-buffer task algebra shared by both official Warp registrations."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        cfg.scene.class_type = SharedNavigationWarpScene
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._warp_term_bindings = {}
        self._warp_obstacle_count = len(self.motion.position)
        self._warp_events = WarpFirstEvents(
            root_state=torch.zeros((self.num_envs, 13), device=self.device),
            target=self.target_pos,
            collision_gap=torch.zeros(self.num_envs, device=self.device),
            collision_terminates=getattr(self.cfg, "variant", None) != "continuous",
            bounds_terminates=getattr(self.cfg, "variant", None) != "continuous",
        )

    def _create_controller(self):
        return _GraphedWarpController(self.num_envs, self.device, dt=self.physics_dt)

    def _configure_controller(self):
        self._rescale_motor_response()

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

    def _compute_first_events(self):
        state = self.root_state()
        self._warp_events.update(
            root_state=state,
            collision_gap=self.nearest_obstacle_signed_distance(state[:, :3]) - 0.3,
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

    def _get_observations(self):
        def sample():
            terms = self._compute_terms()
            binding = self._warp_term_bindings["navigation_terms"]
            if self.cfg.method == "ppo" and self.perturbation.noisy:
                state = self.root_state()
                position, velocity, _ = self.perturbation.state(state[:, :3], state[:, 7:10], state[:, 3:7])
                distances = self.perturbation.normal(4.0 - terms["lidar"], self.cfg.sim_to_real.lidar_std).clamp(0, 4)
                binding = self._navigation_terms(position, velocity, distances, "noisy_navigation_terms")
            # Match Torch's fresh observation storage: future graph replays must
            # not mutate policy inputs already retained by the trainer/history.
            return binding.policy_tensor.clone()

        if self.cfg.method == "diffnav":
            return {"policy": sample()}
        return {"policy": self.perturbation.observe(self.common_step_counter, sample)}


class NavigationWarpEnv(_WarpNavigation, NavigationEnv, _NavigationWarpLoop):
    """Navigation velocity-command task using the official Newton/Warp frontend."""


class DiffNavWarpEnv(_WarpNavigation, DiffNavEnv, _NavigationWarpLoop):
    """DiffNav acceleration-command task using the official Newton/Warp frontend."""
