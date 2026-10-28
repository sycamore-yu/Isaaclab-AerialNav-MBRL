"""BPTT optimization against measured PhysX trajectories, without PPO."""

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from .dynamics import PointMassState, align_transition, policy_transition
from .losses import zhang_losses
from .model import ZhangPolicy, decode_action, policy_state


def cosine_learning_rate(update, cfg):
    """Five full cosine decay/restart cycles within the configured update budget."""
    cycle_progress = (update * cfg.cosine_cycles / cfg.max_iterations) % 1.0
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * (1 + math.cos(math.pi * cycle_progress)) / 2)


class ZhangTrainer:
    def __init__(self, env, cfg, log_dir):
        cfg.validate()
        if abs(env.physics_dt - cfg.physics_dt) > 1e-8 or abs(env.step_dt - cfg.policy_dt) > 1e-8:
            raise ValueError("The environment and differentiable model use different time steps")
        self.env, self.cfg = env, cfg
        self.log_dir = Path(log_dir).resolve()
        self.model = ZhangPolicy().to(env.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.nonfinite_grad_events = 0
        self.completed_updates = 0
        self.elapsed_s = 0.0
        self.velocity_history = None

    def _checkpoint(self, path):
        payload = {
            "format_version": 1, "method": "zhang", "fresh": True,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_cfg": asdict(self.cfg), "completed_updates": self.completed_updates,
            "num_envs": self.env.num_envs, "elapsed_s": self.elapsed_s,
            "nonfinite_grad_events": self.nonfinite_grad_events,
            "environment": {"seed": self.env.cfg.seed, "terrain_seed": self.env.cfg.terrain_seed,
                            "static_obstacles": self.env.cfg.num_static_obstacles,
                            "dynamic_obstacles": self.env.cfg.num_dynamic_obstacles,
                            "physics_dt": self.env.physics_dt, "policy_dt": self.env.step_dt},
            "torch_rng_state": torch.get_rng_state(),
        }
        torch.save(payload, path)

    def _result(self, status, checkpoint=None, error=None):
        result = {
            "method": "zhang", "status": status, "fresh": True,
            "num_envs": self.env.num_envs, "completed_updates": self.completed_updates,
            "max_updates": self.cfg.max_iterations, "rollout_steps": self.cfg.timesteps,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "elapsed_s": self.elapsed_s, "nonfinite_grad_events": self.nonfinite_grad_events,
            "nonfinite_events": int(self.env.nonfinite_events),
            "error": error,
        }
        (self.log_dir / "training_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result

    def run(self):
        self.env.reset()
        observation = self.env.get_zhang_observation()
        zero = torch.zeros_like(observation["velocity"])
        state = PointMassState(observation["position"].detach(), observation["velocity"].detach(), zero, zero)
        hidden = zero.new_zeros(self.env.num_envs, 192)
        queued_command = zero
        previous_command = zero
        lookahead = torch.linspace(0, self.cfg.policy_dt, 10, device=self.env.device)[None, :, None]
        writer = SummaryWriter(str(self.log_dir / "tensorboard"))
        log = (self.log_dir / "updates.jsonl").open("w")
        started = time.perf_counter()
        try:
            for update in range(self.cfg.max_iterations):
                update_started = time.perf_counter()
                state = state.detach()
                hidden, queued_command = hidden.detach(), queued_command.detach()
                previous_command = previous_command.detach()
                history = defaultdict(list)
                terminated_count = torch.zeros((), device=self.env.device, dtype=torch.long)
                for _ in range(self.cfg.timesteps):
                    # EventManager resets mutate its tensors; the rollout graph owns immutable snapshots.
                    parameters = {name: value.detach().clone() for name, value in self.env.zhang_parameters.items()}
                    features, heading, target_velocity, rotation = policy_state(
                        state.position, state.velocity, observation["quaternion"], observation["target"],
                        parameters["max_speed"], parameters["margin"],
                    )
                    # Camera output is a reused buffer and must not overwrite saved CNN inputs.
                    output, hidden = self.model(observation["depth_features"].detach().clone(), features, hidden)
                    command, velocity_prediction = decode_action(output, heading, parameters["thr_est_error"])
                    # Source main_cuda records p + v * sub_div before env.run, including terminating steps.
                    query = state.position[:, None] + state.velocity[:, None] * lookahead
                    history["obstacle_vector"].append(self.env.nearest_obstacle_vector(query))
                    history["margin"].append(parameters["margin"])
                    history["target_velocity"].append(target_velocity)
                    history["velocity_prediction"].append(velocity_prediction)
                    history["observed_velocity"].append(state.velocity.detach())
                    history["episode_id"].append(observation["episode_id"].clone())
                    history["command"].append(command)
                    history["previous_command"].append(previous_command)
                    prediction = policy_transition(state, queued_command, rotation, parameters, self.cfg)
                    with torch.no_grad():
                        _, _, terminated, timeout, info = self.env.step(queued_command.detach())
                        done = terminated | timeout
                        next_observation = self.env.get_zhang_observation()
                    transition = info["transition"]
                    terminal_state, state = align_transition(
                        prediction, transition["position"], transition["velocity"], state.velocity, done,
                        next_observation["position"], next_observation["velocity"], self.cfg.policy_dt,
                    )
                    # Source velocity history is post-step; preserve it before same-step autoreset.
                    history["velocity"].append(terminal_state.velocity)
                    hidden = torch.where(done[:, None], torch.zeros_like(hidden), hidden)
                    queued_command = torch.where(done[:, None], torch.zeros_like(command), command)
                    previous_command = queued_command
                    observation = next_observation
                    terminated_count += done.sum()
                losses, self.velocity_history = zhang_losses(history, self.cfg, self.velocity_history)
                if not bool(torch.isfinite(losses["total"])):
                    raise FloatingPointError("Non-finite Zhang rollout loss")
                learning_rate = cosine_learning_rate(update, self.cfg)
                for group in self.optimizer.param_groups:
                    group["lr"] = learning_rate
                self.optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm,
                                                             error_if_nonfinite=True)
                except RuntimeError:
                    self.nonfinite_grad_events += 1
                    raise
                self.optimizer.step()
                self.completed_updates = update + 1
                self.elapsed_s = time.perf_counter() - started
                metrics = {name: float(value.detach()) for name, value in losses.items()}
                metrics.update(update=self.completed_updates, learning_rate=learning_rate, grad_norm=float(grad_norm),
                               elapsed_s=self.elapsed_s, update_s=time.perf_counter() - update_started,
                               completed_episodes=int(terminated_count))
                log.write(json.dumps(metrics, allow_nan=False) + "\n")
                log.flush()
                for name, value in metrics.items():
                    writer.add_scalar(name, value, self.completed_updates)
                print("ZHANG_UPDATE " + json.dumps(metrics), flush=True)
                if self.completed_updates % 100 == 0:
                    self._checkpoint(self.log_dir / f"checkpoint_{self.completed_updates:04d}.pt")
            checkpoint = self.log_dir / "checkpoint_final.pt"
            self._checkpoint(checkpoint)
            result = self._result("completed", checkpoint)
            print("TRAINING_RESULT " + json.dumps(result), flush=True)
            return result
        except BaseException as error:
            self.elapsed_s = time.perf_counter() - started
            self._result("failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            log.close()
            writer.close()
