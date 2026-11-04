"""Exact preset/sensor checks, plus the verified corrected-sensor historical source."""

import json
from pathlib import Path

from .config import CHECKPOINT_METHODS, OBSERVATION_CONTRACTS

VERIFIED_LEGACY_SOURCE = "f8d42aa279e80b3b208f0ce4785f3c4307fa9c73"


def sensor_contract(sensor, outputs=3, actions=3):
    return {
        "preset": "pointnet",
        "observation_contract": OBSERVATION_CONTRACTS[0],
        "point_features": ["body_x", "body_y", "body_z", "validity_mask"],
        "learned_point_channels": 3,
        "state_dim": 10,
        "policy_output_dim": outputs,
        "action_dim": actions,
        "sensor": {
            "pattern": type(sensor.pattern_cfg).__name__,
            "horizontal_fov": sensor.pattern_cfg.horizontal_fov,
            "horizontal_res": sensor.pattern_cfg.horizontal_res,
            "vertical_ray_angles": list(sensor.pattern_cfg.vertical_ray_angles),
            "ray_alignment": sensor.ray_alignment,
            "max_distance": sensor.max_distance,
            "update_period": sensor.update_period,
            "offset_position": list(sensor.offset.pos),
            "offset_rotation_xyzw": list(sensor.offset.rot),
        },
    }


def policy_contract(env):
    cfg = env.cfg
    if cfg.perception_cfg.mode != "pointnet" or cfg.scene.point_lidar is None:
        raise ValueError("PointNet checkpoint requires presets=pointnet")
    if cfg.perception_cfg.policy_outputs != 3 or cfg.action_space != 3:
        raise ValueError("PointNet requires three policy outputs and three actions")
    return sensor_contract(cfg.scene.point_lidar, cfg.perception_cfg.policy_outputs, cfg.action_space)


def validate_checkpoint(payload, env):
    current = policy_contract(env)
    saved = payload.get("policy_contract")
    if saved is None and payload.get("method") == CHECKPOINT_METHODS[1]:
        from .presets import NavigationPerceptionPresets

        receipt = Path(payload.get("stage_log_dir", "")).parent / "source/source_identity.json"
        try:
            source = json.loads(receipt.read_text())["commit"]
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError("Historical checkpoint needs its verified corrected-sensor source receipt") from exc
        if (
            source != VERIFIED_LEGACY_SOURCE
            or payload.get("environment", {}).get("observation_contract") != OBSERVATION_CONTRACTS[1]
        ):
            raise ValueError("Historical checkpoint is not from the verified corrected-sensor source")
        saved = sensor_contract(NavigationPerceptionPresets().pointnet.point_lidar)
    if saved != current:
        raise ValueError("Checkpoint preset/observation/action/sensor configuration does not match presets=pointnet")
