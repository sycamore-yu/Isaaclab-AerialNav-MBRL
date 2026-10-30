"""Select the same physics presets in custom scripts and the Isaac Lab CLI."""

from __future__ import annotations

import argparse
import sys

PHYSICS_PRESETS = ("isaacsim_physx", "newton_mjwarp")


def parse_physics_args(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    """Remove one canonical ``physics=NAME`` selector before parsing script options.

    Extract it before argparse handles ``--train_args`` so the backend remains a
    process option even when it follows the training hyperparameters.
    """
    selector_help = "Physics selector: physics=isaacsim_physx (default) or physics=newton_mjwarp."
    parser.epilog = f"{parser.epilog}\n{selector_help}" if parser.epilog else selector_help
    tokens = list(sys.argv[1:] if argv is None else argv)
    selectors = [token for token in tokens if token.startswith("physics=")]
    if len(selectors) > 1:
        parser.error("Specify physics=NAME only once")
    physics = selectors[0].partition("=")[2] if selectors else PHYSICS_PRESETS[0]
    if physics not in PHYSICS_PRESETS:
        parser.error(f"Unknown physics preset {physics!r}; choose one of {PHYSICS_PRESETS}")
    args = parser.parse_args([token for token in tokens if not token.startswith("physics=")])
    # The launcher reserves args.physics and would replace our resolved task preset.
    args.physics_preset = physics
    return args


def select_physics(cfg, name: str) -> None:
    """Resolve the official physics preset before creating the simulator."""
    from isaaclab_tasks.utils.hydra import resolve_presets

    if name not in PHYSICS_PRESETS:
        raise ValueError(f"Unknown physics preset {name!r}; choose one of {PHYSICS_PRESETS}")
    resolve_presets(cfg, selected=(name,))
