"""Independent checkpoint loading for the shared first-event evaluator."""

import torch

from isaaclab_navrl.tasks.navigation.depth_history import (
    DEPTH_EVENT_OBSERVATION_CONTRACT,
    OBSERVATION_CONTRACT,
    TEMPORAL_DEPTH_EVENT_OBSERVATION_CONTRACT,
)

from .config import ZhangTrainCfg
from .model import ZhangPolicy, decode_action, policy_state
from .normalization import PolicyObservationNormalizer


class ZhangInference:
    method = "zhang"
    checkpoint_methods = ("zhang",)
    checkpoint_observation_contracts = (OBSERVATION_CONTRACT,)
    config_type = ZhangTrainCfg
    model_type = ZhangPolicy

    def validate_checkpoint(self, payload):
        pass

    def __init__(self, env, checkpoint):
        payload = torch.load(checkpoint, map_location=env.device, weights_only=True)
        if payload["method"] not in self.checkpoint_methods or payload["format_version"] not in (2, 3):
            raise ValueError("Not a supported Zhang checkpoint")
        self.checkpoint_train_cfg = payload["train_cfg"]
        self.training_collision_metric = payload["train_cfg"].get("collision_metric", "dcol")
        self.training_geometry_backend = payload["train_cfg"].get("geometry_backend", "native")
        self.cfg = self.config_type(
            **(
                payload["train_cfg"]
                | {
                    "collision_metric": "sdf",
                    "goal_speed_profile": payload["train_cfg"].get("goal_speed_profile", "linear"),
                    "position_integration": payload["train_cfg"].get("position_integration", "original"),
                    "geometry_backend": self.training_geometry_backend,
                    "depth_event_pose_source": payload["train_cfg"].get("depth_event_pose_source", "estimated"),
                }
            )
        )
        self.cfg.validate()
        if bool(getattr(env.cfg, "depth_event_channel", False)) != self.cfg.depth_event_channel:
            raise ValueError("Evaluation environment must match checkpoint event input contract")
        if (
            self.cfg.depth_event_channel
            and getattr(env.cfg, "depth_event_history_frames", 1) != self.cfg.depth_event_history_frames
        ):
            raise ValueError("Evaluation environment must match event history length")
        if self.cfg.depth_event_channel:
            self.checkpoint_observation_contracts = (
                (TEMPORAL_DEPTH_EVENT_OBSERVATION_CONTRACT,)
                if self.cfg.depth_event_history_frames > 1
                else (DEPTH_EVENT_OBSERVATION_CONTRACT,)
            )
        saved_contract = payload.get("environment", {}).get("observation_contract")
        if (
            self.cfg.depth_event_channel or (env.cfg.sim_to_real.enabled and env.cfg.sim_to_real.input_delay)
        ) and saved_contract not in self.checkpoint_observation_contracts:
            if self.cfg.depth_event_channel:
                raise ValueError("Checkpoint observation contract does not match the evaluation input")
            raise ValueError("Observation timing changed; evaluate the old delayed-input policy with its frozen source")
        self.env = env
        self.validate_checkpoint(payload)
        self.model = self.model_type(self.cfg.recurrent_model).to(env.device).eval()
        if self.cfg.depth_event_channel:
            self.model.enable_depth_event_channel()
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
        observation = self.env.get_zhang_observation()
        reset = observation.privileged["episode_id"] != self.episode_id
        self.hidden[reset] = 0
        self.episode_id.copy_(observation.privileged["episode_id"])
        parameters = self.env.zhang_parameters
        from .goal import final_approach_weight

        goal_weight = None
        if self.cfg.goal_speed_profile == "smooth":
            goal_weight = final_approach_weight(
                self.env,
                observation.privileged["position"],
                observation.privileged["target"],
                parameters["max_speed"],
                parameters["margin"],
                self.cfg.goal_approach_time_s,
            )
        state, heading, _, _ = policy_state(
            observation.policy["position"],
            observation.policy["velocity"],
            observation.policy["quaternion"],
            observation.policy["target"],
            parameters["max_speed"],
            parameters["margin"],
            body_frame=getattr(self.model, "body_frame", False),
            goal_speed_profile=self.cfg.goal_speed_profile,
            goal_approach_time_s=self.cfg.goal_approach_time_s,
            goal_clear_weight=goal_weight,
        )
        depth, state = self.normalizer.normalize(
            observation.policy["points" if self.cfg.perception == "lidar" else "depth_features"],
            state,
            self.normalization_snapshot,
        )
        output, self.hidden = self.model(depth, state, self.hidden)
        command, _ = decode_action(output, heading, parameters["thr_est_error"])
        return command


def load_zhang_policy(env, checkpoint):
    return ZhangInference(env, checkpoint)
