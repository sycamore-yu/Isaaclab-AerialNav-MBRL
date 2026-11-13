"""Newton/Warp stepping, same-step autoreset and host event integration."""

from functools import wraps

import torch
import warp as wp
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.utils.video_recorder import VideoRecorder
from isaaclab_experimental.envs import DirectRLEnvWarp


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
