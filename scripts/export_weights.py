"""Export an inference-only DiffNav checkpoint without local training metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export_weights(input_path: Path, output_path: Path) -> None:
    """Save only the fields used by the public inference loader."""
    source = torch.load(input_path, map_location="cpu", weights_only=True)
    if source.get("method") != "diffnav" or source.get("format_version") != 3:
        raise ValueError("Expected a version-3 DiffNav training checkpoint")
    fields = ("model_state_dict", "normalizer_state_dict", "train_cfg")
    if any(field not in source for field in fields):
        raise ValueError("Checkpoint is missing required inference fields")
    output = {
        "format_version": 3,
        "method": "diffnav",
        "model_state_dict": source["model_state_dict"],
        "normalizer_state_dict": source["normalizer_state_dict"],
        "train_cfg": source["train_cfg"],
        "environment": {
            "observation_contract": source.get("environment", {}).get("observation_contract"),
        },
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
