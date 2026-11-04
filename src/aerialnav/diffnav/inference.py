"""Checkpoint loading and recurrent policy inference."""

import torch

from aerialnav.tasks.navigation.depth_history import OBSERVATION_CONTRACT

from .config import DiffNavTrainCfg
from .model import DiffNavPolicy, decode_action, policy_state
from .normalization import PolicyObservationNormalizer


class DiffNavInference:
    method = "diffnav"
    checkpoint_methods = ("diffnav",)
    checkpoint_observation_contracts = (OBSERVATION_CONTRACT,)
    config_type = DiffNavTrainCfg
    model_type = DiffNavPolicy

    def validate_checkpoint(self, payload):
        pass

    def __init__(self, env, checkpoint):
        payload = torch.load(checkpoint, map_location=env.device, weights_only=True)
        if payload["method"] not in self.checkpoint_methods or payload["format_version"] not in (2, 3):
            raise ValueError("Not a supported DiffNav checkpoint")
        if (
            env.cfg.sim_to_real.enabled
            and env.cfg.sim_to_real.input_delay
            and payload.get("environment", {}).get("observation_contract") not in self.checkpoint_observation_contracts
        ):
            raise ValueError("Observation timing changed; evaluate the old delayed-input policy with its frozen source")
        training = payload["train_cfg"]
        self.checkpoint_train_cfg = training
        self.training_collision_metric = training.get("collision_metric", "dcol")
        self.training_geometry_backend = training.get("geometry_backend", "native")
        runtime = training | {
            "collision_metric": "sdf",
            "position_integration": training.get("position_integration", "original"),
            "geometry_backend": self.training_geometry_backend,
        }
        self.cfg = self.config_type(**runtime)
        self.cfg.validate()
        self.env = env
        self.validate_checkpoint(payload)
        self.model = self.model_type(self.cfg.recurrent_model).to(env.device).eval()
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.normalizer = PolicyObservationNormalizer(
            self.cfg.observation_normalization, self.cfg.normalization_variance_floor, device=env.device
        ).eval()
        if payload["format_version"] == 3:
            self.normalizer.load_state_dict(payload["normalizer_state_dict"], strict=True)
        self.normalization_snapshot = self.normalizer.begin_rollout()
        self.hidden = torch.zeros(env.num_envs, self.model.recurrent_state_size, device=env.device)
        self.episode_id = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    @torch.no_grad()
    def __call__(self, observation=None):
        observation = self.env.get_diffnav_observation()
        reset = observation["episode_id"] != self.episode_id
        self.hidden[reset] = 0
        self.episode_id.copy_(observation["episode_id"])
        parameters = self.env.diffnav_parameters
        state, heading, _, _ = policy_state(
            observation["policy_position"],
            observation["policy_velocity"],
            observation["policy_quaternion"],
            observation["policy_target"],
            parameters["max_speed"],
            parameters["margin"],
            body_frame=getattr(self.model, "body_frame", False),
        )
        depth, state = self.normalizer.normalize(
            observation["policy_points" if self.cfg.perception == "lidar" else "policy_depth_features"],
            state,
            self.normalization_snapshot,
        )
        output, self.hidden = self.model(depth, state, self.hidden)
        command, _ = decode_action(output, heading, parameters["thr_est_error"])
        return command


def load_diffnav_policy(env, checkpoint):
    return DiffNavInference(env, checkpoint)
