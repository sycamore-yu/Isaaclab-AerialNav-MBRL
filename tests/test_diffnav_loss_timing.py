"""Original depth CNN/GRU and dynamic DCOL timing parity through real trainer code."""

import torch
from test_diffnav_contract import AnalyticEnv

from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.diffnav.runner import DiffNavTrainer


class DepthEnv(AnalyticEnv):
    physics_dt = step_dt = 0.05

    def __init__(self):
        super().__init__()
        self.cfg.variant = "terminal"
        self.cfg.perception = "depth"
        self.terrain_distance.signed_distance = lambda point: point[..., 0] * 0 + 5
        self.before_reset_calls = 0

    def step(self, command):
        command = self.perturbation.execute(command)
        self.total_steps += 1
        self.velocity += command * self.step_dt
        self.position += self.velocity * self.step_dt
        done = torch.tensor([self.total_steps % 2 == 0, False])
        timeout = torch.zeros_like(done)
        transition = {
            "position": self.position.clone(),
            "velocity": self.velocity.clone(),
            "executed_action": command.clone(),
            "actuator": command.clone(),
            "bounds": torch.zeros_like(done),
            "collision": done.clone(),
            "terminated": done.clone(),
            "timeout": timeout,
            "collision_position": self.position.clone(),
            "collision_velocity": self.velocity.clone(),
            "collision_substep": done.long(),
            "dynamic_positions_by_substep": torch.empty(1, 0, 3),
            "first_event_code": done.to(torch.int8) * 2,
        }
        callback = getattr(self, "pre_reset_loss_callback", None)
        if callback is not None:
            self.pre_reset_loss_callback = None
            callback(transition)
            self.before_reset_calls += 1
        if done[0]:
            self.position[0] = torch.tensor([0.0, 0.0, 2.0])
            self.velocity[0] = 0
            self.episode_id[0] += 1
            self.perturbation.reset(torch.tensor([0]))
        self.depth.fill_(1 + self.total_steps / 100)
        return {}, torch.zeros(2), done, timeout, {"transition": transition}


def test_depth_policy_two_updates_preserve_weights_and_callbacks_precede_reset(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        weights = []
        for timing in ("deferred", "pre_reset"):
            torch.manual_seed(7)
            env = DepthEnv()
            cfg = DiffNavTrainCfg(loss_timing=timing, timesteps=4, max_iterations=2, velocity_window_s=0.1)
            trainer = DiffNavTrainer(env, cfg, tmp_path / timing)
            assert type(trainer.model).__name__ == "DiffNavPolicy" and trainer.model.fc.weight.shape[0] == 6
            trainer.run()
            assert env.before_reset_calls == (8 if timing == "pre_reset" else 0)
            assert trainer.completed_updates == 2
            weights.append(trainer.model.state_dict())
        for name in weights[0]:
            torch.testing.assert_close(weights[0][name], weights[1][name], rtol=1e-5, atol=1e-6)
    finally:
        torch.set_num_threads(previous_threads)
