"""BPTT optimization against measured PhysX trajectories, without PPO."""

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from itertools import count
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from isaaclab_navrl.tasks.navigation.curriculum import STAGE_CONFIGS
from isaaclab_navrl.tasks.navigation.randomization import DifferentiableDelay
from isaaclab_navrl.training import record_curriculum, write_json

from .dynamics import PointMassState, align_state, align_transition, policy_transition
from .losses import zhang_losses
from .model import ZhangPolicy, decode_action, policy_state


def cosine_learning_rate(update, cfg):
    """Repeat the fixed cosine period independently of the total run length."""
    cycle_progress = (update % cfg.cosine_period_updates) / cfg.cosine_period_updates
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
        self.environment_steps = 0
        self.stage_updates = 0
        self.schedule_updates = 0
        self.parent_checkpoint = None
        self.elapsed_s = 0.0
        self.velocity_history = None

    def load_stage_checkpoint(self, path):
        """Continue the same experiment after rebuilding its stage-specific terrain."""
        payload = torch.load(path, map_location=self.env.device, weights_only=True)
        if payload["format_version"] != 2 or payload["method"] != "zhang":
            raise ValueError("Stage transitions require this experiment's Zhang checkpoint")
        stages = list(STAGE_CONFIGS)
        index = stages.index(self.env.cfg.training_stage)
        if index == 0 or payload["environment"]["training_stage"] != stages[index - 1]:
            raise ValueError("The checkpoint must come from the immediately preceding training stage")
        if self.cfg.max_iterations is None and not payload["curriculum"]["stage_passed"]:
            raise ValueError("Formal training requires a checkpoint that passed the preceding stage")
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.completed_updates = payload["completed_updates"]
        self.environment_steps = payload["environment_steps"]
        self.elapsed_s = payload["elapsed_s"]
        self.nonfinite_grad_events = payload["nonfinite_grad_events"]
        self.schedule_updates = 0 if self.env.cfg.training_stage == "finetune_s1" else payload["schedule_updates"]
        self.parent_checkpoint = str(Path(path).resolve())
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if "cuda_rng_state" in payload and torch.device(self.env.device).type == "cuda":
            torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu(), self.env.device)

    def _checkpoint(self, path):
        payload = {
            "format_version": 2, "method": "zhang", "fresh": self.parent_checkpoint is None,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_cfg": asdict(self.cfg), "completed_updates": self.completed_updates,
            "environment_steps": self.environment_steps,
            "num_envs": self.env.num_envs, "elapsed_s": self.elapsed_s,
            "nonfinite_grad_events": self.nonfinite_grad_events,
            "stage_updates": self.stage_updates, "schedule_updates": self.schedule_updates,
            "parent_checkpoint": self.parent_checkpoint,
            "environment": {"seed": self.env.cfg.seed, "terrain_seed": self.env.cfg.terrain_seed,
                            "static_obstacles": self.env.cfg.num_static_obstacles,
                            "dynamic_obstacles": self.env.cfg.num_dynamic_obstacles,
                            "physics_dt": self.env.physics_dt, "policy_dt": self.env.step_dt,
                            "training_stage": self.env.cfg.training_stage,
                            "sim_to_real": asdict(self.env.cfg.sim_to_real)},
            "torch_rng_state": torch.get_rng_state(),
            "curriculum": self.env.curriculum.state_dict() if self.env.curriculum is not None else None,
        }
        if torch.device(self.env.device).type == "cuda":
            payload["cuda_rng_state"] = torch.cuda.get_rng_state(self.env.device)
        torch.save(payload, path)

    def _result(self, status, checkpoint=None, error=None):
        result = {
            "method": "zhang", "status": status, "fresh": self.parent_checkpoint is None,
            "stage": self.env.cfg.training_stage, "stage_updates": self.stage_updates,
            "parent_checkpoint": self.parent_checkpoint,
            "num_envs": self.env.num_envs, "completed_updates": self.completed_updates,
            "max_updates": self.cfg.max_iterations, "rollout_steps": self.cfg.timesteps,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "elapsed_s": self.elapsed_s, "nonfinite_grad_events": self.nonfinite_grad_events,
            "nonfinite_events": int(self.env.nonfinite_events),
            "environment_steps": self.environment_steps,
            "torch_peak_allocated_mb": (
                torch.cuda.max_memory_allocated(self.env.device) / 2**20
                if torch.device(self.env.device).type == "cuda" else 0.0
            ),
            "curriculum": self.env.curriculum.state_dict() if self.env.curriculum is not None else None,
            "acceptance_passed": bool(
                self.env.curriculum is not None and self.env.curriculum.acceptance_passed
                and self.env.num_envs >= 1024 and self.env.cfg.num_static_obstacles == 400
                and self.env.cfg.num_dynamic_obstacles == 100 and self.env.cfg.sim_to_real.enabled
            ),
            "stage_passed": bool(self.env.curriculum is not None and self.env.curriculum.stage_passed),
            "error": error,
        }
        write_json(self.log_dir / "training_result.json", result)
        return result

    def run(self):
        self.env.reset()
        observation = self.env.get_zhang_observation()
        zero = torch.zeros_like(observation["velocity"])
        state = PointMassState(observation["position"].detach(), observation["velocity"].detach(), zero, zero)
        hidden = zero.new_zeros(self.env.num_envs, 192)
        previous_command = zero
        action_history = DifferentiableDelay(self.env.perturbation.action_buffer.history_length)
        observation_history = DifferentiableDelay(self.env.perturbation.observation_buffer.history_length)
        lookahead = torch.linspace(0, self.cfg.policy_dt, 10, device=self.env.device)[None, :, None]
        writer = SummaryWriter(str(self.log_dir / "tensorboard"))
        log = (self.log_dir / "updates.jsonl").open("w")
        started = time.perf_counter()
        previous_elapsed_s = self.elapsed_s
        initial_updates = self.completed_updates
        self._result("running")
        try:
            updates = count() if self.cfg.max_iterations is None else range(self.cfg.max_iterations)
            for update in updates:
                update_started = time.perf_counter()
                state = state.detach()
                hidden = hidden.detach()
                action_history.detach()
                observation_history.detach()
                previous_command = previous_command.detach()
                history = defaultdict(list)
                terminated_count = torch.zeros((), device=self.env.device, dtype=torch.long)
                execution_error = torch.zeros((), device=self.env.device)
                for _ in range(self.cfg.timesteps):
                    # EventManager resets mutate its tensors; the rollout graph owns immutable snapshots.
                    parameters = {name: value.detach().clone() for name, value in self.env.zhang_parameters.items()}
                    _, heading, target_velocity, rotation = policy_state(
                        state.position, state.velocity, observation["quaternion"], observation["target"],
                        parameters["max_speed"], parameters["margin"],
                    )
                    delayed_state = observation_history.push(
                        torch.cat((state.position, state.velocity), dim=-1), observation["input_delay_steps"]
                    )
                    features, _, _, _ = policy_state(
                        align_state(delayed_state[:, :3], observation["policy_position"]),
                        align_state(delayed_state[:, 3:], observation["policy_velocity"]),
                        observation["policy_quaternion"], observation["policy_target"],
                        parameters["max_speed"], parameters["margin"],
                    )
                    # Camera output is a reused buffer and must not overwrite saved CNN inputs.
                    output, hidden = self.model.forward_checkpointed(
                        observation["policy_depth_features"].detach().clone(), features, hidden
                    )
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
                    applied_command = action_history.push(
                        command, self.env.perturbation.action_buffer.time_lags.clone(), zero_until_ready=True
                    )
                    with torch.no_grad():
                        _, _, terminated, timeout, info = self.env.step(command.detach())
                        done = terminated | timeout
                        next_observation = self.env.get_zhang_observation()
                    transition = info["transition"]
                    execution_error = torch.maximum(
                        execution_error, (applied_command.detach() - transition["executed_action"]).abs().max()
                    )
                    applied_command = align_state(applied_command, transition["executed_action"])
                    prediction = policy_transition(state, applied_command, rotation, parameters, self.cfg)
                    terminal_state, state = align_transition(
                        prediction, transition["position"], transition["velocity"], state.velocity, done,
                        next_observation["position"], next_observation["velocity"], self.cfg.policy_dt,
                        transition["actuator"],
                    )
                    # Source velocity history is post-step; preserve it before same-step autoreset.
                    history["velocity"].append(terminal_state.velocity)
                    hidden = torch.where(done[:, None], torch.zeros_like(hidden), hidden)
                    previous_command = torch.where(done[:, None], torch.zeros_like(command), command)
                    action_history.reset(done)
                    observation_history.reset(done)
                    observation = next_observation
                    terminated_count += done.sum()
                episode_ids = torch.stack(history["episode_id"])
                if self.velocity_history is not None:
                    episode_ids = torch.cat((self.velocity_history["episode_id"], episode_ids))
                if len(episode_ids) > self.cfg.velocity_window_steps:
                    windows = episode_ids.unfold(0, self.cfg.velocity_window_steps, 1)[1:]
                    valid_velocity_windows = (windows == windows[..., :1]).all(dim=-1).float().mean()
                else:
                    valid_velocity_windows = episode_ids.new_zeros((), dtype=torch.float)
                losses, self.velocity_history = zhang_losses(history, self.cfg, self.velocity_history)
                if not bool(torch.isfinite(losses["total"])):
                    raise FloatingPointError("Non-finite Zhang rollout loss")
                if float(execution_error) > 1e-6:
                    raise RuntimeError(f"PhysX and differentiable model executed different delayed actions: {execution_error}")
                learning_rate = cosine_learning_rate(self.schedule_updates, self.cfg)
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
                self.stage_updates = update + 1
                self.completed_updates = initial_updates + self.stage_updates
                self.environment_steps += self.cfg.timesteps * self.env.num_envs
                self.schedule_updates += 1
                self.elapsed_s = previous_elapsed_s + time.perf_counter() - started
                metrics = {name: float(value.detach()) for name, value in losses.items()}
                metrics.update(update=self.completed_updates, learning_rate=learning_rate, grad_norm=float(grad_norm),
                               elapsed_s=self.elapsed_s, update_s=time.perf_counter() - update_started,
                               completed_episodes=int(terminated_count),
                               velocity_window_valid_fraction=float(valid_velocity_windows),
                               executed_action_max_error=float(execution_error))
                log.write(json.dumps(metrics, allow_nan=False) + "\n")
                log.flush()
                for name, value in metrics.items():
                    writer.add_scalar(name, value, self.completed_updates)
                print("ZHANG_UPDATE " + json.dumps(metrics), flush=True)
                record_curriculum(self.env.curriculum, self.completed_updates, self.log_dir, writer, self._checkpoint)
                if self.completed_updates == 1 or self.completed_updates % 100 == 0:
                    self._checkpoint(self.log_dir / "checkpoint_latest.pt")
                if self.env.curriculum is not None and self.env.curriculum.ready_to_advance:
                    self.env.advance_curriculum()
                    observation = self.env.get_zhang_observation()
                    zero = torch.zeros_like(observation["velocity"])
                    state = PointMassState(observation["position"].detach(), observation["velocity"].detach(), zero, zero)
                    hidden = zero.new_zeros(self.env.num_envs, 192)
                    previous_command = zero
                    action_history = DifferentiableDelay(self.env.perturbation.action_buffer.history_length)
                    observation_history = DifferentiableDelay(self.env.perturbation.observation_buffer.history_length)
                    self.velocity_history = None
                    print(f"CURRICULUM -> {self.env.cfg.num_dynamic_obstacles} at update {self.completed_updates}", flush=True)
                result = self._result("running")
                if self.cfg.stop_after_acceptance and result["stage_passed"]:
                    break
            if self.env.curriculum is not None and self.env.curriculum.stage_passed:
                self._checkpoint(self.log_dir / "checkpoint_stage.pt")
            checkpoint = self.log_dir / "checkpoint_final.pt"
            self._checkpoint(checkpoint)
            result = self._result("completed", checkpoint)
            print("TRAINING_RESULT " + json.dumps(result), flush=True)
            return result
        except BaseException as error:
            self.elapsed_s = previous_elapsed_s + time.perf_counter() - started
            self._result("failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            log.close()
            writer.close()
