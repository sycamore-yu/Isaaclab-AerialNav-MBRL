"""Select physical backends and official Torch/Warp environment frontends."""

from __future__ import annotations

import argparse
import sys

PHYSICS_PRESETS = ("isaacsim_physx", "newton_mjwarp")
FRONTENDS = ("torch", "warp")
DEFAULT_PHYSICS = "isaacsim_physx"
DEFAULT_FRONTEND = "torch"


def validate_runtime(physics: str, frontend: str) -> None:
    """Reject unsupported combinations before allocating a simulation."""
    if physics not in PHYSICS_PRESETS:
        raise ValueError(f"physics must be one of {PHYSICS_PRESETS}; received {physics!r}")
    if frontend not in FRONTENDS:
        raise ValueError(f"frontend must be one of {FRONTENDS}; received {frontend!r}")
    if frontend == "warp" and physics != "newton_mjwarp":
        raise ValueError("--frontend warp requires physics=newton_mjwarp")


def parse_physics_args(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse runtime selectors even when they follow ``--train_args``.

    Keep runtime options out of the training-parameter remainder. A separate
    ``physics_preset`` attribute avoids the launcher's generic physics override.
    """
    selector_help = (
        f"Physics selector: physics=NAME, choices {PHYSICS_PRESETS}; default {DEFAULT_PHYSICS}. "
        f"Frontend selector: --frontend {{torch,warp}}; default {DEFAULT_FRONTEND}."
    )
    parser.epilog = f"{parser.epilog}\n{selector_help}" if parser.epilog else selector_help
    tokens = list(sys.argv[1:] if argv is None else argv)
    physics_selectors, frontend_selectors, remaining = [], [], []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("physics="):
            physics_selectors.append(token.partition("=")[2])
        elif token == "--frontend":
            index += 1
            if index == len(tokens):
                parser.error("--frontend requires torch or warp")
            frontend_selectors.append(tokens[index])
        elif token.startswith("--frontend="):
            frontend_selectors.append(token.partition("=")[2])
        else:
            remaining.append(token)
        index += 1
    if len(physics_selectors) > 1 or len(frontend_selectors) > 1:
        parser.error("Specify physics=NAME and --frontend at most once each")
    physics = physics_selectors[0] if physics_selectors else DEFAULT_PHYSICS
    args = parser.parse_args(remaining)
    # An explicit PhysX physics preset selects its only supported frontend.
    # Explicit incompatible combinations still fail validation below.
    args.frontend = (
        frontend_selectors[0] if frontend_selectors else ("torch" if physics == "isaacsim_physx" else DEFAULT_FRONTEND)
    )
    try:
        validate_runtime(physics, args.frontend)
    except ValueError as exc:
        parser.error(str(exc))
    # The launcher reserves args.physics and would replace our resolved task preset.
    args.physics_preset = physics
    return args


def select_physics(cfg, name: str) -> None:
    """Resolve the official physics preset before creating the simulator."""
    from isaaclab_tasks.utils.hydra import resolve_presets

    if name not in PHYSICS_PRESETS:
        raise ValueError(f"Unknown physics preset {name!r}; choose one of {PHYSICS_PRESETS}")
    resolve_presets(cfg, selected=(name,))


def runtime_metadata(env) -> dict[str, str]:
    """Record the actual class without importing the optional Warp runtime on Torch."""
    env_type = type(env)
    is_warp = any(
        base.__module__ == "isaaclab_experimental.envs.direct_rl_env_warp" and base.__name__ == "DirectRLEnvWarp"
        for base in env_type.__mro__
    )
    return {
        "frontend": "warp" if is_warp else "torch",
        "environment_class": f"{env_type.__module__}.{env_type.__qualname__}",
    }


def create_navigation_env(task: str, cfg, args):
    """Construct the selected implementation through the official RL entrypoint."""
    from isaaclab_rl.entrypoints.common import create_isaaclab_env

    validate_runtime(args.physics_preset, args.frontend)
    env = create_isaaclab_env(task, cfg, args, convert_marl_to_single_agent=False).unwrapped
    metadata = runtime_metadata(env)
    if metadata["frontend"] != args.frontend:
        env.close()
        raise RuntimeError(
            f"Requested {args.frontend} frontend but constructed {metadata['environment_class']} "
            f"({metadata['frontend']})"
        )
    return env
