"""Convert one original NavRL policy checkpoint for comparison/evaluation only."""

import argparse
from pathlib import Path

import torch

from isaaclab_navrl.agents.checkpoint_conversion import convert_original_state_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error(f"Destination already exists: {args.destination}")
    original = torch.load(args.source, map_location="cpu", weights_only=True)
    converted = convert_original_state_dict(original)
    converted["infos"]["source"] = str(args.source.resolve())
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.destination)
    print(f"Converted {args.source} -> {args.destination}; original optimizer state was not migrated.")


if __name__ == "__main__":
    main()
