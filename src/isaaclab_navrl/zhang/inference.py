"""Independent checkpoint loading for the shared first-event evaluator."""

import torch

from .config import ZhangTrainCfg
from .model import ZhangPolicy, decode_action, policy_state


class ZhangInference:
    def __init__(self, env, checkpoint):
        payload = torch.load(checkpoint, map_location=env.device, weights_only=True)
        if payload["method"] != "zhang" or payload["format_version"] != 2:
            raise ValueError("Not a supported Zhang checkpoint")
        self.cfg = ZhangTrainCfg(**payload["train_cfg"])
        self.cfg.validate()
        self.env = env
        self.model = ZhangPolicy().to(env.device).eval()
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.hidden = torch.zeros(env.num_envs, 192, device=env.device)
        self.episode_id = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    @torch.no_grad()
    def __call__(self, observation=None):
        observation = self.env.get_zhang_observation()
        reset = observation["episode_id"] != self.episode_id
        self.hidden[reset] = 0
        self.episode_id.copy_(observation["episode_id"])
        parameters = self.env.zhang_parameters
        state, heading, _, _ = policy_state(
            observation["policy_position"], observation["policy_velocity"], observation["policy_quaternion"],
            observation["policy_target"], parameters["max_speed"], parameters["margin"],
        )
        _, heading, _, _ = policy_state(
            observation["position"], observation["velocity"], observation["quaternion"],
            observation["target"], parameters["max_speed"], parameters["margin"],
        )
        output, self.hidden = self.model(observation["policy_depth_features"], state, self.hidden)
        command, _ = decode_action(output, heading, parameters["thr_est_error"])
        return command


def load_zhang_policy(env, checkpoint):
    return ZhangInference(env, checkpoint)
