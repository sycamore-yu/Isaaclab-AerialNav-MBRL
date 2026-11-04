"""BPTT and cost actor-critic optimization of measured simulator trajectories."""

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from itertools import count
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from aerialnav.physics import runtime_metadata
from aerialnav.tasks.navigation.depth_history import OBSERVATION_CONTRACT
from aerialnav.tasks.navigation.randomization import DifferentiableDelay
from aerialnav.training import (
    EPISODE_END_CONTRACT,
    STAGE_CONFIG,
    record_curriculum,
    tensorboard_directory,
    write_json,
    write_training_scalars,
)

from .actor_critic import ActorCriticSettings, CostActorCritic, short_horizon_cost_objective
from .collision import height_bounded_gap, metric_navigation_gap
from .critic_state import CriticHistory, critic_observation
from .diagnostics import GradientDiagnostic, collision_records, critic_vertical_probe
from .dynamics import PointMassState, align_state, align_transition, policy_transition
from .losses import VelocityEMA, diffnav_transition_costs, risk_speed_scale
from .model import DiffNavPolicy, decode_action, policy_state
from .normalization import PolicyObservationNormalizer
from .optimization import OptimizerStepMonitor, RecoveryConfig, RecoveryController, make_plateau_scheduler
from .projection import ActionUpdateProbe, height_priority_gradient
from .streaming import PreResetLoss


def cosine_learning_rate(update, cfg):
    """Repeat the fixed cosine period independently of the total run length."""
    cycle_progress = (update % cfg.cosine_period_updates) / cfg.cosine_period_updates
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * (1 + math.cos(math.pi * cycle_progress)) / 2)


@torch.no_grad()
def rollout_diagnostics(history):
    """Describe measured trajectories without changing losses or acceptance counts."""
    position = torch.stack(history["position"])
    height_target = torch.stack(history["height_target"])
    velocity = torch.stack(history["velocity"])
    command = torch.stack(history["executed_command"])
    delta = torch.stack(history["model_velocity_delta"])
    bounds = torch.stack(history["bounds"])
    low = bounds & (position[..., 2] < 0.2)
    high = bounds & (position[..., 2] > 4.0)
    return {
        "mean_altitude_m": float(position[..., 2].mean()),
        "mean_target_altitude_m": float(height_target.mean()),
        "mean_abs_target_altitude_error_m": float((position[..., 2] - height_target).abs().mean()),
        "mean_horizontal_speed_mps": float(velocity[..., :2].norm(dim=-1).mean()),
        "mean_vertical_speed_mps": float(velocity[..., 2].mean()),
        "mean_abs_vertical_speed_mps": float(velocity[..., 2].abs().mean()),
        "mean_executed_vertical_acceleration_mps2": float(command[..., 2].mean()),
        "mean_abs_executed_vertical_acceleration_mps2": float(command[..., 2].abs().mean()),
        "terminal_lower_bound_count": int(low.sum()),
        "terminal_upper_bound_count": int(high.sum()),
        "terminal_bound_recovered_count": int((bounds & ~low & ~high).sum()),
        "surrogate_velocity_error_mps": float(delta.norm(dim=-1).mean()),
        "surrogate_vertical_velocity_bias_mps": float(delta[..., 2].mean()),
        "depth_age_s": float(torch.stack(history["depth_age_s"]).mean()),
        "depth_known_fraction": float(torch.stack(history["depth_known_fraction"]).mean()),
    }


class DiffNavTrainer:
    method = "diffnav"
    checkpoint_methods = ("diffnav",)
    model_type = DiffNavPolicy
    observation_contract = OBSERVATION_CONTRACT
    checkpoint_observation_contracts = (OBSERVATION_CONTRACT,)

    def checkpoint_metadata(self):
        return {}

    def validate_checkpoint(self, payload):
        pass

    def transition_costs(self, history, cfg, velocity_history=None):
        if cfg.loss_timing == "pre_reset":
            if self.step_loss.steps != len(history["velocity"]):
                raise RuntimeError("Expected exactly one pre-reset loss per transition")
            return self.step_loss.finish()
        return diffnav_transition_costs(history, cfg, velocity_history)

    def begin_rollout_loss(self):
        self.pre_reset_collision = None
        self.step_loss = (
            PreResetLoss(self.cfg, diffnav_transition_costs, self.velocity_history)
            if self.cfg.loss_timing == "pre_reset"
            else None
        )

    def _collision_endpoint(self, predicted_positions, terminal_position, transition):
        substep = transition["collision_substep"]
        index = (substep - 1).clamp_min(0)
        rows = torch.arange(self.env.num_envs, device=self.env.device)
        collision_position = align_state(predicted_positions[rows, index], transition["collision_position"])
        collision_position = torch.where(substep[:, None] > 0, collision_position, terminal_position)
        pose_index = torch.where(substep > 0, index, predicted_positions.shape[1] - 1)
        return collision_position, transition["dynamic_positions_by_substep"][pose_index]

    def prepare_transition_loss(self, history, state, command, rotation, parameters):
        if self.step_loss is None:
            return
        sample = {name: values[-1] for name, values in history.items() if values}

        def before_reset(transition):
            with torch.enable_grad():
                applied = align_state(command, transition["executed_action"])
                if self.cfg.variant == "terminal":
                    prediction, positions = policy_transition(
                        state, applied, rotation, parameters, self.cfg, record_positions=True
                    )
                    self.pre_reset_prediction = (prediction, positions)
                else:
                    prediction = policy_transition(state, applied, rotation, parameters, self.cfg)
                    self.pre_reset_prediction = prediction
                terminal, _ = align_transition(
                    prediction,
                    transition["position"],
                    transition["velocity"],
                    state.velocity,
                    transition["terminated"] | transition["timeout"],
                    transition["position"],
                    transition["velocity"],
                    self.cfg.policy_dt,
                    transition["actuator"],
                )
                scored = sample | {"position": terminal.position, "velocity": terminal.velocity}
                if self.cfg.variant == "terminal":
                    position, obstacles = self._collision_endpoint(positions, terminal.position, transition)
                    gap = self._collision_gap(position, parameters["margin"], obstacles)
                    self.pre_reset_collision = (position, obstacles, gap)
                    scored["collision_gap"] = torch.cat((sample["collision_lookahead"], gap[:, None]), dim=1)
                if self.cfg.failure_terminal_cost:
                    code = transition.get("terminal_event_code", transition["first_event_code"])
                    scored["failure_terminal"] = transition["terminated"] & (code >= 2) & (code <= 4)
                self.step_loss.append(scored)
                if not torch.equal(sample["episode_id"], self.env.episode_id):
                    raise RuntimeError("Loss was evaluated after episode reset")

        self.env.pre_reset_loss_callback = before_reset

    def __init__(self, env, cfg, log_dir):
        cfg.validate()
        if abs(env.physics_dt - cfg.physics_dt) > 1e-8 or abs(env.step_dt - cfg.policy_dt) > 1e-8:
            raise ValueError("The environment and differentiable model use different time steps")
        if env.cfg.variant != cfg.variant:
            raise ValueError("The environment and trainer must use the same DiffNav variant")
        if tuple(env.cfg.height_target_range) != tuple(cfg.height_target_range):
            raise ValueError("The environment and trainer must use the same episode target height range")
        self.env, self.cfg = env, cfg
        self.log_dir = Path(log_dir).resolve()
        self.model = self.model_type(cfg.recurrent_model).to(env.device)
        optimizer_type = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
        self.optimizer = optimizer_type(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.normalizer = PolicyObservationNormalizer(
            cfg.observation_normalization, cfg.normalization_variance_floor, device=env.device
        )
        self.optimizer_monitor = OptimizerStepMonitor(self.model)
        self.scheduler = make_plateau_scheduler(
            self.optimizer,
            cfg.lr_schedule,
            factor=cfg.plateau_factor,
            patience=cfg.plateau_patience,
            threshold=cfg.plateau_threshold,
            cooldown=cfg.plateau_cooldown,
            min_lr=cfg.minimum_lr,
        )
        self.recovery = RecoveryController(
            env.cfg.training_stage or f"dynamic_{env.cfg.num_dynamic_obstacles}",
            RecoveryConfig(enabled=cfg.recovery_enabled, min_lr=cfg.minimum_lr),
        )
        self.lr_scale = 1.0
        self.lr_cap = cfg.lr
        self.actor_critic = None
        self.critic_monitor = None
        self._pending_critic_state = None
        self.horizon_controller = None
        if cfg.algorithm == "ahac":
            from .actor_critic import AdaptiveHorizonController, AHACSettings

            self.horizon_controller = AdaptiveHorizonController(
                AHACSettings(
                    initial_horizon=cfg.ahac_initial_horizon,
                    min_horizon=cfg.ahac_min_horizon,
                    max_horizon=cfg.ahac_max_horizon,
                    risk_clearance=cfg.ahac_risk_clearance,
                    risk_scale=cfg.ahac_risk_scale,
                    risk_budget=cfg.ahac_risk_budget,
                    dual_lr=cfg.ahac_dual_lr,
                )
            )
        self.stop_requested = None
        self.last_rollout_steps = None
        self.nonfinite_grad_events = 0
        self.completed_updates = 0
        self.environment_steps = 0
        self.stage_updates = 0
        self.schedule_updates = 0
        self.parent_checkpoint = None
        self.elapsed_s = 0.0
        self.process_started_at = time.monotonic()
        self.velocity_history = None

    def request_stop(self, reason="operator"):
        """Finish one atomic training update before exporting a stopped checkpoint."""
        self.stop_requested = str(reason)

    def _learning_rate(self):
        if self.cfg.lr_schedule == "cosine":
            value = cosine_learning_rate(self.schedule_updates, self.cfg) * self.lr_scale
            value = max(self.cfg.minimum_lr, min(value, self.lr_cap))
            for group in self.optimizer.param_groups:
                group["lr"] = value
        return self.optimizer.param_groups[0]["lr"]

    def _initialize_rollout(self):
        self.env.reset()
        observation = self.env.get_diffnav_observation()
        zero = torch.zeros_like(observation["velocity"])
        state = PointMassState(observation["position"].detach(), observation["velocity"].detach(), zero, zero)
        hidden = zero.new_zeros(self.env.num_envs, self.model.recurrent_state_size)
        action_history = DifferentiableDelay(self.env.perturbation.action_buffer.history_length)
        observation_history = DifferentiableDelay(self.env.perturbation.observation_buffer.history_length)
        critic_history = CriticHistory(self.cfg.velocity_window_steps, zero)
        self.velocity_ema = VelocityEMA(zero, self.cfg.seed + 31415) if self.cfg.velocity_loss_mode == "ema" else None
        self.velocity_history = None
        return observation, zero, state, hidden, zero, action_history, observation_history, critic_history

    def _critic_observation(
        self, state, observation, hidden, previous_command, critic_history, action_history, observation_history
    ):
        features = critic_observation(
            state,
            observation,
            hidden,
            previous_command,
            self.env.diffnav_parameters,
            critic_history,
            action_history,
            observation_history,
            self.env.episode_length_buf,
            self.env.max_episode_length,
            self.env.perturbation.action_buffer.time_lags,
        )
        return self._with_velocity_memory(features)

    def _with_velocity_memory(self, features):
        if self.cfg.velocity_loss_mode == "ema":
            return torch.cat((features, self.velocity_ema.value, self.velocity_ema.alpha), dim=-1)
        return features

    def _initialize_critic(self, example):
        if self.cfg.algorithm == "bptt" or self.actor_critic is not None:
            return
        settings = ActorCriticSettings(
            algorithm=self.cfg.algorithm,
            discount=self.cfg.discount,
            td_lambda=self.cfg.td_lambda,
            critic_lr=self.cfg.critic_lr,
            critic_iterations=self.cfg.critic_iterations,
            critic_max_iterations=self.cfg.critic_max_iterations,
            seed=self.cfg.seed + 2718,
            critic_min_iterations=min(6, self.cfg.critic_max_iterations),
            value_transform=self.cfg.critic_value_transform,
            target_tau=self.cfg.critic_target_tau,
        )
        self.actor_critic = CostActorCritic(example.shape[-1], settings, device=self.env.device)
        self.critic_monitor = OptimizerStepMonitor(self.actor_critic.critics)
        if self._pending_critic_state is not None:
            self.actor_critic.load_state_dict(self._pending_critic_state)
            self._pending_critic_state = None

    def _collision_gap(self, query, radius, obstacle_positions=None, *, static_distance=None):
        """Metric SDF minus the safety radius, at the existing sample times."""
        from aerialnav.tasks.navigation.geometry import nearest_obstacle_signed_distance

        radius = radius.reshape(-1, *([1] * (query.ndim - 2))).expand(query.shape[:-1])
        static_distance = (
            self.env.terrain_distance.signed_distance(query) if static_distance is None else static_distance
        )
        obstacle_positions = self.env.motion.position if obstacle_positions is None else obstacle_positions
        return (
            nearest_obstacle_signed_distance(
                query,
                static_distance,
                obstacle_positions,
                self.env.motion.size,
                dynamic_backend=getattr(self.env.terrain_distance, "dynamic_signed_distance", None),
            )
            - radius
        )

    def load_stage_checkpoint(self, path, *, resume=False, allow_num_envs_change=False):
        """Load the preceding stage or explicitly resume saved same-stage learning state."""
        payload = torch.load(path, map_location=self.env.device, weights_only=True)
        if payload["format_version"] != 3 or payload["method"] not in self.checkpoint_methods:
            raise ValueError("Stage transitions require this experiment's DiffNav checkpoint")
        if payload["environment"].get("observation_contract") not in self.checkpoint_observation_contracts:
            raise ValueError("Observation timing changed; resume the old experiment with its frozen source")
        self.validate_checkpoint(payload)
        # Version-3 persisted configs predate optional failure-cost and target-rate fields.
        saved_config = asdict(
            type(self.cfg)(
                **(
                    payload["train_cfg"]
                    | {
                        "loss_timing": payload["train_cfg"].get("loss_timing", "deferred"),
                        "position_integration": payload["train_cfg"].get("position_integration", "original"),
                        "geometry_backend": payload["train_cfg"].get("geometry_backend", "native"),
                        "avoidance_gradient": payload["train_cfg"].get("avoidance_gradient", "original"),
                    }
                )
            )
        )
        allowed_stage_changes = {"lr", "max_iterations", "stop_after_acceptance", "gradient_diagnostic_updates"}
        allowed_stage_changes.update(getattr(self.cfg, "resume_runtime_fields", ()))
        if resume:
            allowed_stage_changes.remove("lr")
        for name, value in asdict(self.cfg).items():
            if name not in allowed_stage_changes and saved_config[name] != value:
                raise ValueError(f"Stage transitions cannot change {name}")
        stages = list(STAGE_CONFIG)
        index = stages.index(self.env.cfg.training_stage)
        if resume:
            saved_env = payload["environment"]
            expected = {
                "training_stage": self.env.cfg.training_stage,
                "static_obstacles": self.env.cfg.num_static_obstacles,
                "dynamic_obstacles": self.env.cfg.num_dynamic_obstacles,
                "seed": self.env.cfg.seed,
                "terrain_seed": self.env.cfg.terrain_seed,
                "sim_to_real": asdict(self.env.cfg.sim_to_real),
                "target_speed_range": getattr(self.env.cfg, "target_speed_range", None),
            }
            if (not allow_num_envs_change and payload["num_envs"] != self.env.num_envs) or any(
                saved_env[key] != value for key, value in expected.items()
            ):
                raise ValueError("Same-stage continuation requires matching environment, seed and num_envs")
        else:
            if index == 0 or payload["environment"]["training_stage"] != stages[index - 1]:
                raise ValueError("The checkpoint must come from the immediately preceding training stage")
            if self.cfg.max_iterations is None and not payload["curriculum"]["stage_passed"]:
                raise ValueError("Formal training requires a checkpoint that passed the preceding stage")
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.normalizer.load_state_dict(payload["normalizer_state_dict"], strict=True)
        self._pending_critic_state = payload["actor_critic_state_dict"]
        if self.horizon_controller is not None:
            self.horizon_controller.load_state_dict(payload["horizon_controller_state_dict"])
        self.completed_updates = payload["completed_updates"]
        self.environment_steps = payload["environment_steps"]
        self.elapsed_s = payload["elapsed_s"]
        self.nonfinite_grad_events = payload["nonfinite_grad_events"]
        self.schedule_updates = (
            payload["schedule_updates"] if resume or self.env.cfg.training_stage != "finetune_s1" else 0
        )
        self.lr_scale = payload["lr_scale"]
        self.lr_cap = payload["lr_cap"]
        # A harder stage has a different report distribution. Start a fresh
        # Plateau/recovery report history; retain model/optimizer/normalization.
        if not resume and (self.env.cfg.training_stage == "finetune_s1" or self.cfg.lr != payload["train_cfg"]["lr"]):
            self.lr_scale, self.lr_cap = 1.0, self.cfg.lr
            for group in self.optimizer.param_groups:
                group["lr"] = self.cfg.lr
        self.parent_checkpoint = str(Path(path).resolve())
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if "cuda_rng_state" in payload and torch.device(self.env.device).type == "cuda":
            torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu(), self.env.device)
        if resume:
            self.stage_updates = payload["stage_updates"]
            if self.scheduler is not None:
                self.scheduler.load_state_dict(payload["scheduler_state_dict"])
            self.log_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                self.log_dir / "continuation.json",
                {
                    "checkpoint": self.parent_checkpoint,
                    "runtime_changes": {
                        name: {
                            "from": payload["train_cfg"].get(name, "legacy source default"),
                            "to": getattr(self.cfg, name),
                        }
                        for name in getattr(self.cfg, "resume_runtime_fields", ())
                        if name not in payload["train_cfg"] or saved_config[name] != getattr(self.cfg, name)
                    },
                    "previous_num_envs": payload["num_envs"],
                    "num_envs": self.env.num_envs,
                    "completed_updates": self.completed_updates,
                    "stage_updates": self.stage_updates,
                    "schedule_updates": self.schedule_updates,
                    "environment_steps": self.environment_steps,
                    "previous_episode_end_contract": payload["environment"].get(
                        "episode_end_contract", "arrival_latched_until_failure_or_timeout_v0"
                    ),
                    "episode_end_contract": EPISODE_END_CONTRACT,
                    "previous_wait_before_truncate_s": payload["environment"].get("wait_before_truncate_s"),
                    "wait_before_truncate_s": getattr(self.env.cfg, "wait_before_truncate", 5.0),
                    "restored": [
                        "model",
                        "optimizer",
                        "normalizer",
                        "scheduler",
                        "torch_cpu_cuda_rng",
                        "LR",
                        "counters",
                    ],
                    "reset": ["environment", "recurrent_state", "report_window", "best_reports", "recovery_reports"],
                },
            )

    def _checkpoint(self, path):
        payload = {
            "format_version": 3,
            "method": self.method,
            "physics": self.cfg.physics,
            **runtime_metadata(self.env),
            "variant": self.cfg.variant,
            "algorithm": self.cfg.algorithm,
            "stage_log_dir": str(self.log_dir),
            "fresh": self.parent_checkpoint is None,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "normalizer_state_dict": self.normalizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "recovery_state_dict": self.recovery.state_dict(),
            "lr_scale": self.lr_scale,
            "lr_cap": self.lr_cap,
            "actor_critic_state_dict": self.actor_critic.state_dict() if self.actor_critic is not None else None,
            "horizon_controller_state_dict": (
                self.horizon_controller.state_dict() if self.horizon_controller is not None else None
            ),
            "train_cfg": asdict(self.cfg),
            "completed_updates": self.completed_updates,
            "environment_steps": self.environment_steps,
            "num_envs": self.env.num_envs,
            "elapsed_s": self.elapsed_s,
            "nonfinite_grad_events": self.nonfinite_grad_events,
            "stage_updates": self.stage_updates,
            "schedule_updates": self.schedule_updates,
            "parent_checkpoint": self.parent_checkpoint,
            "environment": {
                "physics": self.cfg.physics,
                **runtime_metadata(self.env),
                "seed": self.env.cfg.seed,
                "terrain_seed": self.env.cfg.terrain_seed,
                "static_obstacles": self.env.cfg.num_static_obstacles,
                "dynamic_obstacles": self.env.cfg.num_dynamic_obstacles,
                "physics_dt": self.env.physics_dt,
                "policy_dt": self.env.step_dt,
                "depth_camera_batch_size": getattr(self.env.cfg, "depth_camera_batch_size", None),
                "height_target_range": self.env.cfg.height_target_range,
                "target_speed_range": getattr(self.env.cfg, "target_speed_range", None),
                "geometry_backend": self.cfg.geometry_backend,
                "training_stage": self.env.cfg.training_stage,
                "variant": self.cfg.variant,
                "sim_to_real": asdict(self.env.cfg.sim_to_real),
                "observation_contract": self.observation_contract,
                "episode_end_contract": EPISODE_END_CONTRACT,
                "wait_before_truncate_s": getattr(self.env.cfg, "wait_before_truncate", 5.0),
            },
            **self.checkpoint_metadata(),
            "torch_rng_state": torch.get_rng_state(),
            "curriculum": self.env.curriculum.state_dict() if self.env.curriculum is not None else None,
        }
        if torch.device(self.env.device).type == "cuda":
            payload["cuda_rng_state"] = torch.cuda.get_rng_state(self.env.device)
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def _restore_stable(self, path):
        """Restore learned state only; never rewind training/report counters or RNG."""
        path = Path(path).resolve()
        if not path.is_relative_to(self.log_dir):
            raise ValueError("Recovery checkpoint must belong to this stage's log directory")
        payload = torch.load(path, map_location=self.env.device, weights_only=True)
        if (
            payload.get("format_version") != 3
            or payload.get("method") != self.method
            or payload.get("stage_log_dir") != str(self.log_dir)
            or payload["environment"]["training_stage"] != self.env.cfg.training_stage
            or payload["environment"]["dynamic_obstacles"] != self.env.cfg.num_dynamic_obstacles
            or payload["train_cfg"] != asdict(self.cfg)
            or payload["environment"].get("observation_contract") != self.observation_contract
        ):
            raise ValueError("Stable recovery requires the exact same run, stage and configuration")
        self.validate_checkpoint(payload)
        current_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.normalizer.load_state_dict(payload["normalizer_state_dict"], strict=True)
        if self.actor_critic is not None:
            progress = self.actor_critic.state_dict()
            self.actor_critic.load_state_dict(
                {
                    **payload["actor_critic_state_dict"],
                    "completed_fits": progress["completed_fits"],
                    "shuffle_rng_state": progress["shuffle_rng_state"],
                }
            )
        if self.horizon_controller is not None:
            self.horizon_controller.load_state_dict(
                {
                    **payload["horizon_controller_state_dict"],
                    "completed_updates": self.horizon_controller.completed_updates,
                }
            )
        for group, value in zip(self.optimizer.param_groups, current_lrs, strict=True):
            group["lr"] = value
        # Keep the current scheduler, recovery decisions, LR scale/cap, RNG and
        # accumulated experience. The caller resets environment and all memory.

    def _report_controls(self, report, writer):
        if report is None:
            return False
        previous_lr = self.optimizer.param_groups[0]["lr"]
        if self.scheduler is not None:
            self.scheduler.step(report["success_rate"])
        scheduler_reduced = self.optimizer.param_groups[0]["lr"] < previous_lr
        if scheduler_reduced:
            self.lr_cap = min(self.lr_cap, self.optimizer.param_groups[0]["lr"])
        decision = self.recovery.observe({**report, "stage": report.get("stage", self.recovery.stage)})
        restored = False
        if decision.action == "save_stable":
            checkpoint = self.log_dir / f"checkpoint_stable_{self.recovery.stage}_u{self.completed_updates}.pt"
            if checkpoint.exists():
                raise FileExistsError(f"Stable checkpoint is immutable: {checkpoint}")
            self._checkpoint(checkpoint)
            checkpoints = {level: str(checkpoint) for level in decision.levels}
            self.recovery.acknowledge("save_stable", checkpoints=checkpoints)
        elif decision.action == "reduce_lr":
            # A Plateau reduction on this very report satisfies the first
            # recovery action. Do not lower the same LR twice on one report.
            if not scheduler_reduced:
                for group in self.optimizer.param_groups:
                    group["lr"] = max(decision.min_lr, group["lr"] * decision.lr_factor)
                self.lr_scale *= decision.lr_factor
            self.lr_cap = min(self.lr_cap, self.optimizer.param_groups[0]["lr"])
            self.recovery.acknowledge("reduce_lr")
        elif decision.action == "restore":
            self._restore_stable(decision.checkpoint)
            self.recovery.acknowledge("restore")
            restored = True
        event = {
            **asdict(decision),
            "lr_before": previous_lr,
            "lr_after": self.optimizer.param_groups[0]["lr"],
            "scheduler_reduced": scheduler_reduced,
            "state": self.recovery.state,
            "restore_count": self.recovery.restore_count,
            "stable_checkpoints": self.recovery.stable_checkpoints,
            "environment_steps": self.environment_steps,
        }
        with (self.log_dir / "optimization_events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        writer.add_scalar("Optimization/next_learning_rate", event["lr_after"], self.completed_updates)
        writer.add_scalar("Recovery/restore_count", self.recovery.restore_count, self.completed_updates)
        if self.recovery.reference_success is not None:
            writer.add_scalar("Recovery/reference_success", self.recovery.reference_success, self.completed_updates)
            writer.add_scalar(
                "Recovery/degradation_threshold", self.recovery.degradation_threshold, self.completed_updates
            )
            writer.add_scalar("Recovery/stable_level", max(self.recovery.stable_checkpoints), self.completed_updates)
        if decision.action != "none" or scheduler_reduced:
            print("OPTIM_CONTROL " + json.dumps(event), flush=True)
        return restored

    def _result(self, status, checkpoint=None, error=None):
        result = {
            "method": self.method,
            "episode_end_contract": EPISODE_END_CONTRACT,
            "wait_before_truncate_s": getattr(self.env.cfg, "wait_before_truncate", 5.0),
            "physics": self.cfg.physics,
            **runtime_metadata(self.env),
            "variant": self.cfg.variant,
            "algorithm": self.cfg.algorithm,
            "recurrent_model": self.cfg.recurrent_model,
            "collision_metric": self.cfg.collision_metric,
            "geometry_backend": self.cfg.geometry_backend,
            "avoidance_gradient": self.cfg.avoidance_gradient,
            "approach_mode": self.cfg.approach_mode,
            "lateral_avoidance": self.cfg.lateral_avoidance,
            "risk_speed_floor": self.cfg.risk_speed_floor,
            "dynamic_lookahead": "constant_velocity_same_query_time",
            "status": status,
            "velocity_loss_mode": self.cfg.velocity_loss_mode,
            "resolved_velocity_loss_mode": self.cfg.resolved_velocity_loss_mode,
            "height_loss_mode": self.cfg.height_loss_mode,
            "height_target_range": list(self.cfg.height_target_range),
            "static_obstacle_distance": getattr(
                self.env.terrain_distance, "geometry_mode", "signed_heightfield_occupancy_exit_proxy"
            ),
            "static_projection": getattr(
                self.env.terrain_distance, "projection_precision", "float64_selected_face_refinement_float32_storage"
            ),
            "optimizer": self.cfg.optimizer,
            "weight_decay": self.cfg.weight_decay,
            "gradient_projection": self.cfg.gradient_projection,
            "fresh": self.parent_checkpoint is None,
            "stage": self.env.cfg.training_stage,
            "stage_updates": self.stage_updates,
            "parent_checkpoint": self.parent_checkpoint,
            "num_envs": self.env.num_envs,
            "completed_updates": self.completed_updates,
            "max_updates": self.cfg.max_iterations,
            "rollout_steps": self.last_rollout_steps,
            "configured_rollout_steps": self.cfg.timesteps,
            "bootstrap_semantics": ("absorbing_zero_timeout_pre_reset_live_endpoint" if self.actor_critic else None),
            "adaptive_horizon": self.cfg.algorithm == "ahac",
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "elapsed_s": self.elapsed_s,
            "nonfinite_grad_events": self.nonfinite_grad_events,
            "nonfinite_events": int(self.env.nonfinite_events),
            "environment_steps": self.environment_steps,
            "torch_peak_allocated_mb": (
                torch.cuda.max_memory_allocated(self.env.device) / 2**20
                if torch.device(self.env.device).type == "cuda"
                else 0.0
            ),
            "curriculum": self.env.curriculum.state_dict() if self.env.curriculum is not None else None,
            "acceptance_passed": bool(
                self.env.curriculum is not None
                and self.env.curriculum.acceptance_passed
                and self.env.num_envs >= 1024
                and self.env.curriculum.min_episodes >= 1024
                and self.env.cfg.num_static_obstacles == STAGE_CONFIG["finetune_s2"]["static_obstacles"]
                and self.env.cfg.num_dynamic_obstacles == STAGE_CONFIG["finetune_s2"]["dynamic_obstacles"]
                and self.env.cfg.sim_to_real.enabled
            ),
            "stage_passed": bool(self.env.curriculum is not None and self.env.curriculum.stage_passed),
            "recovery_restore_count": self.recovery.restore_count,
            "stop_requested": self.stop_requested,
            "error": error,
        }
        write_json(self.log_dir / "training_result.json", result)
        return result

    def run(self):
        (observation, zero, state, hidden, previous_command, action_history, observation_history, critic_history) = (
            self._initialize_rollout()
        )
        if self.cfg.algorithm != "bptt":
            self._initialize_critic(
                self._critic_observation(
                    state, observation, hidden, previous_command, critic_history, action_history, observation_history
                )
            )
            write_json(
                self.log_dir / "actor_critic_protocol.json",
                {
                    "bootstrap": "absorbing_zero_timeout_pre_reset_live_endpoint",
                    "velocity_loss_mode": self.cfg.resolved_velocity_loss_mode,
                    "critic_settings": asdict(self.actor_critic.settings),
                    "horizon_controller": self.horizon_controller.state_dict() if self.horizon_controller else None,
                    "risk_metric": (
                        "metre_primitive_ground_height_clearance_static_verified_exit_proxy"
                        if self.horizon_controller
                        else None
                    ),
                    "risk_aggregate": "worst_trajectory_temporal_peak" if self.horizon_controller else None,
                    "stability_diagnostics": ["actor_gradient_clip_ratio", "surrogate_velocity_error_mps"],
                    "physx_contact_jacobian_verified": False,
                },
            )
        lookahead = torch.linspace(0, self.cfg.policy_dt, 10, device=self.env.device)[None, :, None]
        writer = SummaryWriter(str(tensorboard_directory(self.log_dir)))
        writer.add_text("Protocol/episode_end_contract", EPISODE_END_CONTRACT, self.completed_updates)
        if self.cfg.variant == "continuous":
            write_json(
                self.log_dir / "continuous_geometry.json",
                {
                    "ground_plane_z_m": 0.01,
                    "ground_clearance": "query_height_minus_ground_plane_minus_radius",
                    "static_geometry": getattr(
                        self.env.terrain_distance, "geometry_mode", "signed_heightfield_occupancy_exit_proxy"
                    ),
                    "static_projection": getattr(
                        self.env.terrain_distance,
                        "projection_precision",
                        "float64_selected_face_refinement_float32_storage",
                    ),
                    "applies_to": ["obstacle_avoidance", "collision"],
                    "query_timing": "drone_and_obstacles_at_same_constant_velocity_lookahead_time",
                    "static_penetration": "negative_distance_to_verified_original_mesh_exit",
                    "static_distance_is_exact_sdf": False,
                    "static_occupancy": "conservative_highest_vertical_mesh_projection",
                    "collision_or_bounds_reset": False,
                },
            )
        if self.cfg.variant == "terminal":
            write_json(
                self.log_dir / "collision_geometry.json",
                {
                    "scope": "hybrid_dynamic_" + self.cfg.collision_metric,
                    "dynamic_geometry": "sphere_vs_axis_aligned_box_or_finite_z_cylinder",
                    "dynamic_selection": "minimum_metric_over_all_dynamic_obstacles",
                    "dynamic_metric": (
                        "radius_times_alpha_minus_one_metre_scaled_proxy_not_signed_distance"
                        if self.cfg.collision_metric == "dcol"
                        else "primitive_signed_distance_minus_radius"
                    ),
                    "lookahead_obstacles": "position_plus_velocity_times_query_time",
                    "static_geometry": "signed_heightfield_occupancy_exit_proxy",
                    "static_projection": "float64_selected_face_refinement_float32_storage",
                    "static_penetration": "negative_distance_to_verified_original_mesh_exit",
                    "static_distance_is_exact_sdf": False,
                    "static_occupancy": "conservative_highest_vertical_mesh_projection",
                    "ground_gap": "query_height_minus_0.01_minus_radius",
                    "terminal_sample": "first_collision_substep_position_and_obstacles_or_policy_terminal_position",
                    "collision_gradient": "aligned_surrogate_at_the_same_physics_substep_before_reset",
                    "physx_contact_jacobian": False,
                },
            )
        log = (self.log_dir / "updates.jsonl").open("w")
        started = time.perf_counter()
        previous_elapsed_s = self.elapsed_s
        initial_updates = self.completed_updates
        initial_stage_updates = self.stage_updates
        self._result("running")
        try:
            updates = count() if self.cfg.max_iterations is None else range(self.cfg.max_iterations)
            for update in updates:
                update_started = time.perf_counter()
                diagnostic_update = initial_updates + update + 1 in self.cfg.gradient_diagnostic_updates
                fixed_action_probe = None
                fixed_action_probe_metadata = None
                state = state.detach()
                hidden = hidden.detach()
                action_history.detach()
                observation_history.detach()
                critic_history.detach()
                if self.velocity_ema is not None:
                    self.velocity_ema.detach()
                previous_command = previous_command.detach()
                normalization_snapshot = self.normalizer.begin_rollout()
                rollout_steps = (
                    self.horizon_controller.horizon if self.horizon_controller is not None else self.cfg.timesteps
                )
                self.last_rollout_steps = rollout_steps
                history = defaultdict(list)
                self.begin_rollout_loss()
                collisions = []
                value_probes = []
                terminated_count = torch.zeros((), device=self.env.device, dtype=torch.long)
                execution_error = torch.zeros((), device=self.env.device)
                for rollout_step in range(rollout_steps):
                    # EventManager resets mutate its tensors; the rollout graph owns immutable snapshots.
                    parameters = {name: value.detach().clone() for name, value in self.env.diffnav_parameters.items()}
                    if self.actor_critic is not None:
                        history["critic_observation"].append(
                            self._critic_observation(
                                state,
                                observation,
                                hidden,
                                previous_command,
                                critic_history,
                                action_history,
                                observation_history,
                            ).detach()
                        )
                    _, heading, target_velocity, rotation = policy_state(
                        state.position,
                        state.velocity,
                        observation["quaternion"],
                        observation["target"],
                        parameters["max_speed"],
                        parameters["margin"],
                    )
                    observation_history.push(
                        torch.cat((state.position, state.velocity), dim=-1), observation["input_delay_steps"]
                    )
                    features, policy_heading, _, _ = policy_state(
                        align_state(state.position, observation["policy_position"]),
                        align_state(state.velocity, observation["policy_velocity"]),
                        observation["policy_quaternion"],
                        observation["policy_target"],
                        parameters["max_speed"],
                        parameters["margin"],
                        body_frame=getattr(self.model, "body_frame", False),
                    )
                    # Camera output is a reused buffer and must not overwrite saved CNN inputs.
                    depth = (
                        observation["policy_points" if self.cfg.perception == "lidar" else "policy_depth_features"]
                        .detach()
                        .clone()
                    )
                    normalized_depth, normalized_features = self.normalizer.normalize(
                        depth, features, normalization_snapshot
                    )
                    self.normalizer.observe(depth, features.detach())
                    # Source look-ahead queries belong to the pre-step state.
                    query = state.position[:, None] + state.velocity[:, None] * lookahead
                    from aerialnav.tasks.navigation.geometry import nearest_obstacle_signed_distance

                    if self.cfg.approach_mode == "relative" or self.cfg.avoidance_gradient == "reshaped":
                        query.requires_grad_(True)
                    # shortcut: constant obstacle velocity within one policy step; turns are not predicted.
                    obstacle_positions = self.env.motion.position + self.env.motion.velocity * lookahead[..., None]
                    static_distance = self.env.terrain_distance.signed_distance(query)
                    geometry = nearest_obstacle_signed_distance(
                        query,
                        static_distance,
                        obstacle_positions,
                        self.env.motion.size,
                        dynamic_velocity=self.env.motion.velocity if self.cfg.approach_mode == "relative" else None,
                        dynamic_backend=getattr(self.env.terrain_distance, "dynamic_signed_distance", None),
                    )
                    if self.cfg.approach_mode == "relative":
                        obstacle_distance, obstacle_velocity = geometry
                        normal = torch.autograd.grad(obstacle_distance.sum(), query, retain_graph=True)[0].detach()
                        normal = torch.nn.functional.normalize(normal, dim=-1)
                        relative_velocity = state.velocity[:, None] - obstacle_velocity
                        history["relative_velocity"].append(relative_velocity[:, 1:])
                        history["obstacle_normal"].append(normal[:, 1:])
                        speed_scale = risk_speed_scale(
                            -(relative_velocity * normal).sum(-1),
                            obstacle_distance - parameters["margin"][:, None],
                            parameters["max_speed"],
                            self.cfg.risk_speed_floor,
                        )
                        target_velocity = target_velocity * speed_scale
                        history["target_speed_scale"].append(speed_scale)
                    else:
                        obstacle_distance = geometry
                    if diagnostic_update and rollout_step == rollout_steps - 1:
                        # Snapshot before the recurrent call; the same observations
                        # isolate the command change caused by one optimizer step.
                        clearance = (obstacle_distance.detach() - parameters["margin"][:, None]).amin(dim=1)
                        probe_ids = clearance.topk(min(16, self.env.num_envs), largest=False).indices
                        fixed_action_probe = {
                            name: value[probe_ids].detach().clone()
                            for name, value in {
                                "depth": normalized_depth,
                                "features": normalized_features,
                                "hidden": hidden,
                                "heading": policy_heading,
                                "thrust_error": parameters["thr_est_error"],
                            }.items()
                        }
                        fixed_action_probe_metadata = {
                            "selection": "smallest_minimum_pre_step_lookahead_clearance",
                            "environment_ids": probe_ids.tolist(),
                            "clearance_m": clearance[probe_ids].tolist(),
                        }
                    output, hidden = self.model.forward_checkpointed(normalized_depth, normalized_features, hidden)
                    command, velocity_prediction = decode_action(output, policy_heading, parameters["thr_est_error"])
                    # Source main_cuda records p + v * sub_div before env.run, including terminating steps.
                    history["obstacle_signed_distance"].append(obstacle_distance)
                    if self.cfg.avoidance_gradient == "reshaped":
                        from .gradient_field import reshape_avoidance_distance

                        history["avoidance_signed_distance"].append(
                            reshape_avoidance_distance(
                                obstacle_distance,
                                query,
                                state.velocity.detach(),
                                observation["target"] - state.position.detach(),
                            )
                        )
                    if self.cfg.variant == "terminal":
                        collision_lookahead = self._collision_gap(
                            query[:, 1:-1],
                            parameters["margin"],
                            obstacle_positions[:, 1:-1],
                            static_distance=static_distance[:, 1:-1],
                        )
                    if self.cfg.variant == "terminal":
                        history["collision_lookahead"].append(collision_lookahead)
                    history["margin"].append(parameters["margin"])
                    history["target_velocity"].append(target_velocity)
                    history["height_target"].append(observation["height_target"].detach().clone())
                    if self.cfg.height_loss_mode == "goal_height_band":
                        history["height_range"].append(observation["height_range"].detach().clone())
                    history["velocity_prediction"].append(velocity_prediction)
                    history["observed_velocity"].append(state.velocity.detach())
                    history["depth_age_s"].append(observation["depth_age_s"])
                    history["depth_known_fraction"].append(observation["depth_known_fraction"])
                    history["episode_id"].append(observation["episode_id"].clone())
                    history["command"].append(command)
                    history["previous_command"].append(previous_command)
                    applied_command = action_history.push(
                        command, self.env.perturbation.action_buffer.time_lags.clone(), zero_until_ready=True
                    )
                    self.prepare_transition_loss(history, state, applied_command, rotation, parameters)
                    with torch.no_grad():
                        _, _, terminated, timeout, info = self.env.step(command.detach())
                        # DirectRLEnv reuses these buffers on the next simulation step.
                        terminated, timeout = terminated.clone(), timeout.clone()
                        done = terminated | timeout
                        next_observation = self.env.get_diffnav_observation()
                    transition = info["transition"]
                    if diagnostic_update and self.cfg.variant == "terminal":
                        collisions.extend(collision_records(self.env, transition))
                    execution_error = torch.maximum(
                        execution_error, (applied_command.detach() - transition["executed_action"]).abs().max()
                    )
                    applied_command = align_state(applied_command, transition["executed_action"])
                    if self.cfg.variant == "terminal":
                        cached = getattr(self, "pre_reset_prediction", None)
                        if cached is None:
                            prediction, predicted_positions = policy_transition(
                                state, applied_command, rotation, parameters, self.cfg, record_positions=True
                            )
                        else:
                            prediction, predicted_positions = cached
                        self.pre_reset_prediction = None
                    else:
                        prediction = getattr(self, "pre_reset_prediction", None)
                        if prediction is None:
                            prediction = policy_transition(state, applied_command, rotation, parameters, self.cfg)
                        self.pre_reset_prediction = None
                    history["bounds"].append(transition["bounds"])
                    history["done"].append(done.clone())
                    history["collision"].append(transition["collision"])
                    if self.cfg.failure_terminal_cost:
                        code = transition.get("terminal_event_code", transition["first_event_code"])
                        history["failure_terminal"].append(terminated & (code >= 2) & (code <= 4))
                    if self.cfg.resolved_velocity_loss_mode == "truncated_window":
                        history["timeout"].append(transition["timeout"])
                        history["invalid"].append(transition["invalid"])
                    if self.cfg.resolved_velocity_loss_mode == "truncated_window" or self.method == "pointnet":
                        history["episode_step"].append(transition["step"])
                    history["executed_command"].append(transition["executed_action"])
                    history["model_velocity_delta"].append(prediction.velocity.detach() - transition["velocity"])
                    terminal_state, state = align_transition(
                        prediction,
                        transition["position"],
                        transition["velocity"],
                        state.velocity,
                        done,
                        next_observation["position"],
                        next_observation["velocity"],
                        self.cfg.policy_dt,
                        transition["actuator"],
                    )
                    # Source velocity history is post-step; preserve it before same-step autoreset.
                    history["position"].append(terminal_state.position)
                    history["velocity"].append(terminal_state.velocity)
                    if self.velocity_ema is not None:
                        history["velocity_ema"].append(self.velocity_ema.update(terminal_state.velocity))
                    if self.cfg.variant == "terminal":
                        # Use the first collision's actual physical substep, including its
                        # obstacle poses. A later policy-end position can already have passed
                        # the object. Its aligned surrogate graph survives same-step reset.
                        if self.pre_reset_collision is None:
                            collision_position, collision_obstacles = self._collision_endpoint(
                                predicted_positions, terminal_state.position, transition
                            )
                            terminal_gap = self._collision_gap(
                                collision_position, parameters["margin"], collision_obstacles
                            )
                        else:
                            collision_position, collision_obstacles, terminal_gap = self.pre_reset_collision
                            self.pre_reset_collision = None
                        history["collision_gap"].append(torch.cat((collision_lookahead, terminal_gap[:, None]), dim=1))
                        if self.horizon_controller is not None:
                            endpoint_clearance = metric_navigation_gap(
                                collision_position,
                                self.env.terrain_distance.signed_distance(collision_position),
                                collision_obstacles,
                                self.env.motion.size,
                            )
                            history["risk_clearance"].append(
                                torch.cat(
                                    (height_bounded_gap(query, obstacle_distance - 0.3), endpoint_clearance[:, None]),
                                    dim=1,
                                )
                            )
                    if self.actor_critic is not None:
                        # DiffAero: absorbing terminals have no future cost; timeouts
                        # and the ordinary horizon use the pre-reset endpoint graph.
                        critic_history.push(terminal_state.velocity, target_velocity, torch.zeros_like(done))
                        endpoint = critic_observation(
                            terminal_state,
                            transition,
                            hidden,
                            command,
                            transition["critic_parameters"],
                            critic_history,
                            action_history,
                            observation_history,
                            transition["step"],
                            self.env.max_episode_length,
                            transition["action_delay_steps"],
                        )
                        endpoint = self._with_velocity_memory(endpoint)
                        live_value = self.actor_critic.actor_value(endpoint)
                        if diagnostic_update:
                            value_probes.append(
                                critic_vertical_probe(
                                    self.actor_critic,
                                    endpoint,
                                    terminal_state.position[:, 2],
                                    terminated,
                                )
                            )
                        actor_next_value = torch.where(terminated, torch.zeros_like(live_value), live_value)
                        history["actor_next_value"].append(actor_next_value)
                        history["critic_next_value"].append(actor_next_value.detach())
                        history["absorbing_terminal"].append(terminated)
                        history["time_truncated"].append(timeout & ~terminated)
                        critic_history.reset(done)
                    if self.velocity_ema is not None:
                        self.velocity_ema.reset(done)
                    hidden = torch.where(done[:, None], torch.zeros_like(hidden), hidden)
                    previous_command = torch.where(done[:, None], torch.zeros_like(command), command)
                    action_history.reset(done)
                    observation_history.reset(done)
                    observation = next_observation
                    terminated_count += done.sum()
                losses, self.velocity_history, transition_cost = self.transition_costs(
                    history, self.cfg, self.velocity_history
                )
                actor_loss = losses["total"]
                algorithm_metrics = {}
                if self.actor_critic is not None:
                    actor_terms = short_horizon_cost_objective(
                        transition_cost,
                        torch.stack(history["actor_next_value"]),
                        torch.stack(history["done"]),
                        discount=self.cfg.discount,
                    )
                    actor_loss = actor_terms["objective"]
                    algorithm_metrics.update(
                        {f"ActorCritic/{name}": float(value.detach()) for name, value in actor_terms.items()}
                    )
                    if self.actor_critic is not None:
                        absorbing = torch.stack(history["absorbing_terminal"])
                        values = torch.stack(history["critic_next_value"])
                        algorithm_metrics["ActorCritic/absorbing_terminal_count"] = float(absorbing.sum())
                        algorithm_metrics["ActorCritic/absorbing_bootstrap_abs_sum"] = float(
                            values.masked_select(absorbing).abs().sum()
                        )
                        algorithm_metrics["ActorCritic/pre_reset_timeout_count"] = float(
                            torch.stack(history["time_truncated"]).sum()
                        )
                risk = None
                if self.horizon_controller is not None:
                    risk = self.horizon_controller.risk(torch.stack(history["risk_clearance"]))
                    risk_penalty = self.horizon_controller.penalty(risk)
                    actor_loss = actor_loss + risk_penalty
                    algorithm_metrics["AHAC/constraint_penalty"] = float(risk_penalty.detach())
                algorithm_metrics["actor_objective"] = float(actor_loss.detach())
                if not bool(torch.isfinite(losses["total"])):
                    raise FloatingPointError("Non-finite DiffNav rollout loss")
                if not bool(torch.isfinite(actor_loss)):
                    raise FloatingPointError("Non-finite DiffNav actor objective")
                if float(execution_error) > 1e-6:
                    raise RuntimeError(
                        f"PhysX and differentiable model executed different delayed actions: {execution_error}"
                    )
                learning_rate = self._learning_rate()
                self.optimizer.zero_grad(set_to_none=True)
                gradient_diagnostic = None
                height_gradient = None
                trainable = tuple(parameter for parameter in self.model.parameters() if parameter.requires_grad)
                if diagnostic_update:
                    components = None
                    if self.actor_critic is not None:
                        components = {name: actor_terms[name] for name in ("discounted_cost", "bootstrap")}
                        if risk is not None:
                            components["risk_penalty"] = risk_penalty
                    gradient_diagnostic = GradientDiagnostic(self.model, losses, self.cfg, components=components)
                    if self.actor_critic is None:
                        height_gradient = gradient_diagnostic.component_gradient("height_boundary")
                elif self.cfg.gradient_projection == "height_priority":
                    components = torch.autograd.grad(
                        losses["height_boundary"],
                        trainable,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    height_gradient = torch.cat(
                        [
                            (torch.zeros_like(parameter) if gradient is None else gradient.detach()).reshape(-1)
                            for parameter, gradient in zip(trainable, components, strict=True)
                        ]
                    )
                actor_loss.backward()
                if gradient_diagnostic is not None:
                    gradient_diagnostic.check_total_gradient()
                action_probe = None
                if (
                    diagnostic_update and self.actor_critic is None
                ) or self.cfg.gradient_projection == "height_priority":
                    total_gradient = torch.cat(
                        [
                            (
                                torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach()
                            ).reshape(-1)
                            for parameter in trainable
                        ]
                    )
                    if diagnostic_update:
                        action_probe = ActionUpdateProbe(
                            self.model,
                            self.optimizer,
                            total_flat=total_gradient,
                            height_flat=height_gradient,
                            height_weight=self.cfg.coef_height_boundary,
                            max_grad_norm=self.cfg.max_grad_norm,
                            probe=fixed_action_probe,
                        )
                    if self.cfg.gradient_projection == "height_priority":
                        projected, projection_metrics = height_priority_gradient(
                            total_gradient,
                            height_gradient,
                            self.cfg.coef_height_boundary,
                        )
                        offset = 0
                        for parameter in trainable:
                            end = offset + parameter.numel()
                            if parameter.grad is not None:
                                parameter.grad.copy_(projected[offset:end].view_as(parameter))
                            offset = end
                        algorithm_metrics.update(
                            {
                                f"Projection/{name}": float(value)
                                for name, value in projection_metrics.items()
                                if isinstance(value, (bool, int, float))
                            }
                        )
                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True
                    )
                except RuntimeError:
                    self.nonfinite_grad_events += 1
                    raise
                self.optimizer_monitor.before_step()
                self.optimizer.step()
                optimizer_metrics = self.optimizer_monitor.after_step()
                gradient_record = None
                if gradient_diagnostic is not None:
                    gradient_record = gradient_diagnostic.finish(self.optimizer, grad_norm, self.cfg.max_grad_norm)
                    gradient_record["applied_gradient_projection"] = self.cfg.gradient_projection
                    if action_probe is not None:
                        gradient_record["optimizer_action_probe"] = action_probe.finish(
                            self.model,
                            applied_projection=self.cfg.gradient_projection,
                        )
                        gradient_record["optimizer_action_probe"]["probe_selection"] = fixed_action_probe_metadata
                        gradient_record["diagnostic_s"] += action_probe.diagnostic_s
                        del action_probe
                    gradient_record["collision_events"] = collisions
                    gradient_record["critic_vertical_probes"] = value_probes
                    del gradient_diagnostic
                if self.actor_critic is not None:
                    self.critic_monitor.before_step()
                    critic_metrics = self.actor_critic.fit(
                        torch.stack(history["critic_observation"]),
                        transition_cost.detach(),
                        torch.stack(history["critic_next_value"]),
                        torch.stack(history["done"]),
                    )
                    algorithm_metrics.update({f"ActorCritic/{name}": value for name, value in critic_metrics.items()})
                    algorithm_metrics.update(
                        {f"Critic/{name}": value for name, value in self.critic_monitor.after_step().items()}
                    )
                rollout_metrics = rollout_diagnostics(history)
                if self.horizon_controller is not None:
                    terminal_fraction = float(torch.stack(history["absorbing_terminal"]).any(dim=0).float().mean())
                    algorithm_metrics.update(
                        self.horizon_controller.update(
                            risk.detach(),
                            terminal_fraction=terminal_fraction,
                            gradient_ratio=float(grad_norm) / self.cfg.max_grad_norm,
                            model_velocity_error=rollout_metrics["surrogate_velocity_error_mps"],
                        )
                    )
                self.stage_updates = initial_stage_updates + update + 1
                self.completed_updates = initial_updates + update + 1
                self.environment_steps += rollout_steps * self.env.num_envs
                self.schedule_updates += 1
                self.elapsed_s = previous_elapsed_s + time.perf_counter() - started
                metrics = {name: float(value.detach()) for name, value in losses.items()}
                metrics.update(optimizer_metrics)
                metrics.update(algorithm_metrics)
                metrics.update(rollout_metrics)
                if hasattr(self.model, "execution_metrics"):
                    metrics.update(self.model.execution_metrics())
                metrics.update(
                    update=self.completed_updates,
                    learning_rate=learning_rate,
                    grad_norm=float(grad_norm),
                    elapsed_s=self.elapsed_s,
                    process_elapsed_s=time.monotonic() - self.process_started_at,
                    update_s=time.perf_counter() - update_started,
                    completed_episodes=int(terminated_count),
                    rollout_steps=rollout_steps,
                    environment_steps=self.environment_steps,
                    executed_action_max_error=float(execution_error),
                )
                if torch.device(self.env.device).type == "cuda":
                    for name, read_value in (
                        ("allocated", torch.cuda.memory_allocated),
                        ("reserved", torch.cuda.memory_reserved),
                        ("peak_allocated", torch.cuda.max_memory_allocated),
                        ("peak_reserved", torch.cuda.max_memory_reserved),
                    ):
                        metrics[f"Memory/torch_{name}_mb"] = read_value(self.env.device) / 2**20
                if gradient_record is not None:
                    heights = torch.stack(history["position"]).detach()[..., 2]
                    gradient_record.update(
                        update=self.completed_updates,
                        stage=self.env.cfg.training_stage,
                        variant=self.cfg.variant,
                        environment_steps=self.environment_steps,
                        num_envs=self.env.num_envs,
                        train_cfg=asdict(self.cfg),
                        rollout=metrics.copy(),
                        upper_boundary_active_fraction=float(
                            (heights > self.cfg.height_bounds[1] - self.cfg.boundary_margin).float().mean()
                        ),
                        lower_boundary_active_fraction=float(
                            (heights < self.cfg.height_bounds[0] + self.cfg.boundary_margin).float().mean()
                        ),
                    )
                    with (self.log_dir / "gradient_diagnostics.jsonl").open("a") as diagnostic_log:
                        diagnostic_log.write(json.dumps(gradient_record, allow_nan=False) + "\n")
                    metrics["gradient_diagnostic_s"] = gradient_record["diagnostic_s"]
                log.write(json.dumps(metrics, allow_nan=False) + "\n")
                log.flush()
                write_training_scalars(writer, metrics, self.completed_updates)
                print("DIFFNAV_UPDATE " + json.dumps(metrics), flush=True)
                report = record_curriculum(
                    self.env.curriculum, self.completed_updates, self.log_dir, writer, self._checkpoint
                )
                if self._report_controls(report, writer):
                    (
                        observation,
                        zero,
                        state,
                        hidden,
                        previous_command,
                        action_history,
                        observation_history,
                        critic_history,
                    ) = self._initialize_rollout()
                if self.completed_updates == 1 or self.completed_updates % 100 == 0:
                    self._checkpoint(self.log_dir / "checkpoint_latest.pt")
                if self.env.curriculum is not None and self.env.curriculum.ready_to_advance:
                    self.env.advance_curriculum()
                    self.recovery.reset_stage(f"dynamic_{self.env.cfg.num_dynamic_obstacles}")
                    (
                        observation,
                        zero,
                        state,
                        hidden,
                        previous_command,
                        action_history,
                        observation_history,
                        critic_history,
                    ) = self._initialize_rollout()
                    print(
                        f"CURRICULUM -> {self.env.cfg.num_dynamic_obstacles} at update {self.completed_updates}",
                        flush=True,
                    )
                result = self._result("running")
                if self.stop_requested is not None:
                    checkpoint = self.log_dir / "checkpoint_stopped.pt"
                    self._checkpoint(checkpoint)
                    result = self._result("stopped", checkpoint)
                    print("TRAINING_RESULT " + json.dumps(result), flush=True)
                    return result
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
