"""Shared experiment argument grammar; imports without starting a simulator."""

import math
from dataclasses import asdict
from typing import get_args

from .config import DiffNavTrainCfg


def parse_binary_flag(raw: str) -> bool:
    """Use the same explicit 0/1 grammar as train_diffnav.py."""
    if raw not in ("0", "1"):
        raise ValueError("expected 0 or 1")
    return raw == "1"


def parse_gradient_diagnostic_updates(raw: str) -> tuple[int, ...]:
    """Validate the same opt-in update-list grammar as train_diffnav.py."""
    if raw == "":
        return ()
    parts = raw.split(",")
    if any(not part.isascii() or not part.isdecimal() for part in parts):
        raise ValueError("gradient_diagnostic_updates requires comma-separated positive integers without empty entries")
    updates = tuple(int(part) for part in parts)
    if any(update < 1 for update in updates) or len(set(updates)) != len(updates):
        raise ValueError("gradient_diagnostic_updates must contain unique positive integers")
    return updates


TrainArgValue = str | int | float | bool | tuple[int, ...]
TRAIN_ARG_CHOICES = {
    name: get_args(field.type)
    for name, field in DiffNavTrainCfg.__dataclass_fields__.items()
    if get_args(field.type) and isinstance(get_args(field.type)[0], str)
}
TRAIN_ARG_CHOICES["velocity_loss_mode"] = get_args(get_args(DiffNavTrainCfg.__annotations__["velocity_loss_mode"])[0])
TRAIN_ARG_CHOICES["timing"] = ("baseline", "unified_20hz", "unified_50hz")
TRAIN_ARG_TYPES = {
    "timing": str,
    "num_dynamic_obstacles": int,
    "target_speed_min": float,
    "target_speed_max": float,
    "variant": str,
    "algorithm": str,
    "recurrent_model": str,
    "lr_schedule": str,
    "observation_normalization": parse_binary_flag,
    "normalization_variance_floor": float,
    "recovery_enabled": parse_binary_flag,
    "timesteps": int,
    "velocity_window_s": float,
    "velocity_loss_mode": str,
    "collision_metric": str,
    "approach_mode": str,
    "lateral_avoidance": float,
    "risk_speed_floor": float,
    "camera_batch_size": int,
    "lr": float,
    "optimizer": str,
    "weight_decay": float,
    "plateau_patience": int,
    "height_loss_mode": str,
    "height_target_min": float,
    "height_target_max": float,
    "gradient_projection": str,
    "max_grad_norm": float,
    "grad_decay": float,
    "coef_obj_avoidance": float,
    "coef_collide": float,
    "coef_d_acc": float,
    "coef_d_jerk": float,
    "coef_height_boundary": float,
    "boundary_margin": float,
    "boundary_scale": float,
    "cosine_period_updates": int,
    "critic_lr": float,
    "failure_terminal_cost": float,
    "critic_target_tau": float,
    "coef_v": float,
    "coef_v_pred": float,
    "critic_iterations": int,
    "critic_max_iterations": int,
    "critic_value_transform": str,
    "discount": float,
    "td_lambda": float,
    "ahac_min_horizon": int,
    "ahac_max_horizon": int,
    "ahac_initial_horizon": int,
    "ahac_risk_clearance": float,
    "ahac_risk_scale": float,
    "ahac_risk_budget": float,
    "ahac_dual_lr": float,
    "gradient_diagnostic_updates": parse_gradient_diagnostic_updates,
}


def parse_train_args(tokens: list[str]) -> dict[str, TrainArgValue]:
    """Accept only DiffNav hyperparameters; process and curriculum flags stay fixed."""
    values = {}
    index = 1 if tokens and tokens[0] == "--" else 0
    while index < len(tokens):
        option, separator, raw_value = tokens[index].partition("=")
        key = option.removeprefix("--")
        if not option.startswith("--") or key not in TRAIN_ARG_TYPES:
            raise ValueError(f"Unsupported --train_args option: {option}")
        if key in values:
            raise ValueError(f"Duplicate --train_args option: {option}")
        if not separator:
            index += 1
            if index == len(tokens):
                raise ValueError(f"Missing value for {option}")
            raw_value = tokens[index]
        try:
            value = TRAIN_ARG_TYPES[key](raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid value for {option}: {raw_value}") from exc
        if key in TRAIN_ARG_CHOICES:
            if value not in TRAIN_ARG_CHOICES[key]:
                raise ValueError(f"Unknown DiffNav {key}: {value}")
        elif key == "gradient_diagnostic_updates" or isinstance(value, bool):
            pass  # The converters already validate tuple and explicit 0/1 grammars.
        elif not math.isfinite(value):
            raise ValueError(f"{option} must be finite")
        elif key == "discount":
            if not 0 <= value < 1:
                raise ValueError("discount must be in [0, 1) for continuing cost learning")
        elif key == "td_lambda":
            if not 0 <= value <= 1:
                raise ValueError("td_lambda must be in [0, 1]")
        elif key in ("grad_decay", "critic_target_tau", "risk_speed_floor"):
            if not 0 < value <= 1:
                raise ValueError("grad_decay must be in (0, 1]")
        elif key.startswith("coef_") or key in (
            "boundary_margin",
            "ahac_risk_clearance",
            "weight_decay",
            "plateau_patience",
            "failure_terminal_cost",
            "lateral_avoidance",
            "num_dynamic_obstacles",
        ):
            if value < 0:
                raise ValueError(f"{option} must be nonnegative")
        elif value <= 0:
            raise ValueError(f"{option} must be positive")
        values[key] = value
        index += 1
    if values.get("algorithm", "bptt") == "bptt" and values.get("critic_value_transform", "linear") != "linear":
        raise ValueError("--critic_value_transform softplus requires --algorithm shac or ahac")
    if values.get("failure_terminal_cost", 0) and values.get("algorithm", "bptt") == "bptt":
        raise ValueError("failure_terminal_cost requires SHAC or AHAC")
    if values.get("algorithm", "bptt") != "bptt":
        if values.get("recurrent_model", "gru") == "sru_lstm_gated":
            raise ValueError("SRU-LSTM-Gated currently requires --algorithm bptt; SHAC/AHAC are not supported")
        if values.get("variant", "dcol") != "dcol":
            raise ValueError("SHAC/AHAC comparisons require --variant dcol")
        if values.get("gradient_projection", "none") != "none":
            raise ValueError("Height-priority projection requires --algorithm bptt")
    if values.get("timesteps", 150) < 2:
        raise ValueError("timesteps must be at least 2")
    if (
        not 2
        <= values.get("ahac_min_horizon", 16)
        <= values.get("ahac_initial_horizon", 32)
        <= values.get("ahac_max_horizon", 300)
    ):
        raise ValueError("AHAC requires 2 <= min <= initial <= max horizon")
    if not 0.2 < values.get("height_target_min", 0.5) < values.get("height_target_max", 2.5) < 4.0:
        raise ValueError("height target range must be increasing and inside (0.2, 4.0) metres")
    low, high = values.get("target_speed_min"), values.get("target_speed_max")
    if (low is None) != (high is None) or (low is not None and not 0 < low < high):
        raise ValueError("target speed requires both bounds with 0 < min < max")
    return values


def add_train_args(parser):
    """Use the same options and dataclass defaults for direct and queued training."""
    defaults = asdict(DiffNavTrainCfg())
    defaults.update(
        timing="unified_20hz",
        num_dynamic_obstacles=None,
        target_speed_min=None,
        target_speed_max=None,
        camera_batch_size=64,
        lr=None,
        velocity_loss_mode=None,
        height_target_min=defaults["height_target_range"][0],
        height_target_max=defaults["height_target_range"][1],
    )
    for name, converter in TRAIN_ARG_TYPES.items():
        parser.add_argument("--" + name, type=converter, default=defaults[name], choices=TRAIN_ARG_CHOICES.get(name))
