"""Export an inference-only DiffNav checkpoint without local training metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aerialnav.diffnav.config import DiffNavTrainCfg


def export_weights(input_path: Path, output_path: Path) -> None:
    """Save only the fields used by the public inference loader."""
    source = torch.load(input_path, map_location="cpu", weights_only=True)
    if source.get("method") != "diffnav" or source.get("format_version") != 3:
        raise ValueError("Expected a version-3 DiffNav training checkpoint")
    fields = ("model_state_dict", "normalizer_state_dict", "train_cfg")
    if any(field not in source for field in fields):
        raise ValueError("Checkpoint is missing required inference fields")
    # Validate only; keep legacy training metadata rather than relabelling it.
    validation = dict(source["train_cfg"])
    if validation.get("collision_metric") == "dcol":
        validation["collision_metric"] = "sdf"
    try:
        DiffNavTrainCfg(**validation).validate()
    except (TypeError, ValueError) as exc:
        raise ValueError("Unsupported public training configuration") from exc
    environment = source.get("environment", {})
    public_environment = {"observation_contract": environment.get("observation_contract")}
    for name in (
        "profile",
        "training_stage",
        "static_obstacles",
        "dynamic_obstacles",
        "target_speed_range",
        "wait_before_truncate_s",
        "episode_end_contract",
        "randomization",
    ):
        if name in environment:
            public_environment[name] = environment[name]
    output = {
        "format_version": 3,
        "method": "diffnav",
        "model_state_dict": source["model_state_dict"],
        "normalizer_state_dict": source["normalizer_state_dict"],
        "train_cfg": source["train_cfg"],
        "environment": public_environment,
    }
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing checkpoint: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_weights(args.input, args.output)
