"""The fixed Newton/MJWarp + Warp navigation runtime."""

from __future__ import annotations

import argparse
import sys

DEFAULT_PHYSICS = "newton_mjwarp"
DEFAULT_FRONTEND = "warp"


def validate_runtime(physics: str, frontend: str) -> None:
    """Reject checkpoints or configurations from unsupported runtimes."""
    if physics != DEFAULT_PHYSICS:
        raise ValueError(f"physics must be {DEFAULT_PHYSICS!r}; received {physics!r}")
    if frontend != DEFAULT_FRONTEND:
        raise ValueError(f"frontend must be {DEFAULT_FRONTEND!r}; received {frontend!r}")


def parse_runtime_args(
    parser: argparse.ArgumentParser, argv: list[str] | None = None, *, allow_presets=False
) -> argparse.Namespace:
    """Separate domain presets from entrypoint arguments and training remainders."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    domain_tokens, remaining = [], []
    for token in tokens:
        if token in ("--physics", "--frontend") or token.startswith(("physics=", "--physics=", "--frontend=")):
            parser.error("Navigation uses Newton/MJWarp + Warp; runtime selectors are not supported")
        if token.startswith("presets="):
            domain_tokens.append(token)
        else:
            remaining.append(token)
    if domain_tokens and not allow_presets:
        parser.error("This entry point does not support domain presets; use the preset-aware BPTT or evaluation entry")
    args = parser.parse_args(remaining)
    args.physics_preset = DEFAULT_PHYSICS
    args.frontend = DEFAULT_FRONTEND
    args.preset_overrides = domain_tokens
    return args


def runtime_metadata(env) -> dict[str, str]:
    """Identify the actual environment; construction rejects unsupported implementations."""
    env_type = type(env)
    is_warp = any(
        base.__module__ == "isaaclab_experimental.envs.direct_rl_env_warp" and base.__name__ == "DirectRLEnvWarp"
        for base in env_type.__mro__
    )
    return {
        "physics": DEFAULT_PHYSICS,
        "frontend": DEFAULT_FRONTEND if is_warp else "unsupported",
        "environment_class": f"{env_type.__module__}.{env_type.__qualname__}",
    }


def create_navigation_env(task: str, cfg, args):
    """Construct the registered Warp environment through Isaac Lab's entrypoint."""
    from isaaclab_rl.entrypoints.common import create_isaaclab_env

    validate_runtime(args.physics_preset, args.frontend)
    env = create_isaaclab_env(task, cfg, args, convert_marl_to_single_agent=False).unwrapped
    metadata = runtime_metadata(env)
    if metadata["frontend"] != DEFAULT_FRONTEND:
        env.close()
        raise RuntimeError(f"Expected a Warp environment, got {metadata['environment_class']}")
    return env


def resolve_navigation_config(task, args, *, training=False):
    """Compose navigation and optional perception presets through Isaac Lab."""
    from isaaclab_tasks.utils.hydra import resolve_task_config

    return resolve_task_config(
        task,
        "bptt_cfg_entry_point" if training else None,
        overrides=args.preset_overrides,
    )
