"""Simulator-independent owner of navigation targets and episode bookkeeping."""

import torch

from isaaclab_navrl.zhang.termination import (
    arrival_wait_timeout,
    latch_first_event,
    record_arrival,
    terminal_outcomes,
)


class NavigationTask:
    """Mutable per-drone task state; shared obstacle motion is not owned here.

    All writes are in place so Warp kernels may retain these tensor addresses.
    Physical execution, controller state and sensor history remain with their
    existing owners. This class neither steps nor resets a simulator.
    """

    def __init__(self, num_envs: int, device: str | torch.device):
        self.target_pos = torch.zeros(num_envs, 3, device=device)
        self.target_speed = torch.zeros(num_envs, 1, device=device)
        self.target_dir = torch.zeros_like(self.target_pos)
        self.height_range = torch.zeros(num_envs, 2, device=device)
        self.previous_velocity = torch.zeros_like(self.target_pos)
        self.first_reach_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.arrived = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.arrive_time = torch.zeros(num_envs, device=device)
        self.episode_failure = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.episode_return = torch.zeros(num_envs, device=device)
        self.episode_id = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.first_event_code = torch.zeros(num_envs, dtype=torch.int8, device=device)
        self.first_event_time = torch.zeros(num_envs, device=device)

    @torch.no_grad()
    def reset(self, env_ids, position, target, speed):
        """Install an already sampled route; reset only the selected rows."""
        ids = torch.as_tensor(env_ids, device=self.target_pos.device, dtype=torch.long)
        self.target_pos[ids] = target
        self.target_speed[ids] = speed
        self.target_dir[ids] = target - position
        self.height_range[ids] = torch.stack(
            (torch.minimum(position[:, 2], target[:, 2]), torch.maximum(position[:, 2], target[:, 2])), dim=-1
        )
        for value in (
            self.previous_velocity,
            self.first_reach_step,
            self.arrived,
            self.arrive_time,
            self.episode_failure,
            self.episode_return,
            self.first_event_code,
            self.first_event_time,
        ):
            value.index_fill_(0, ids, 0)
        self.episode_id[ids] += 1

    @torch.no_grad()
    def record_substep(self, terms, elapsed_s):
        record_arrival(self.arrived, self.arrive_time, terms["reached"], elapsed_s)
        latch_first_event(self.first_event_code, self.first_event_time, terms, elapsed_s)

    @torch.no_grad()
    def advance(
        self, terms, step, *, policy_dt, max_episode_length, wait_before_truncate, terminated, first_arrival=None
    ):
        """Score one policy boundary without changing the termination policy."""
        elapsed_s = step.float() * policy_dt
        record_arrival(self.arrived, self.arrive_time, terms["reached"], elapsed_s)
        wait_expired = arrival_wait_timeout(terms["reached"], self.arrive_time, elapsed_s, wait_before_truncate)
        episode_timeout = step >= max_episode_length
        timeout = episode_timeout | wait_expired
        if first_arrival is None:
            first_arrival = terms["reached"] & ~terminated & (self.first_reach_step == 0)
        self.first_reach_step.copy_(torch.where(first_arrival, step, self.first_reach_step))
        return {
            "timeout": timeout,
            "arrival_wait_timeout": wait_expired,
            "episode_timeout": episode_timeout,
            "terminal_event_code": self.result(terms, terminated | timeout),
        }

    def result(self, terms, done):
        """Final success uses current goal membership and the latched failure."""
        return terminal_outcomes(terms, done, self.episode_failure)

    @torch.no_grad()
    def finish_first_events(self, timeout, elapsed_s):
        first = (self.first_event_code == 0) & timeout
        self.first_event_code[first] = 5
        self.first_event_time.copy_(torch.where(first, elapsed_s, self.first_event_time))

    @torch.no_grad()
    def record_reward(self, reward, velocity):
        self.episode_return.add_(reward)
        self.previous_velocity.copy_(velocity)
