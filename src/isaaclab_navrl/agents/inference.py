"""Load a NavRL checkpoint through the official RSL-RL runner."""

from pathlib import Path

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from .rsl_rl_cfg import NavRLPpoRunnerCfg


def load_navrl_policy(env, checkpoint: str | Path):
    """Return a deterministic callable accepting the raw env observation dict.

    The official wrapper resets the environment during construction. The caller
    must collect episode starts/goals and initial observations after this call.
    """
    cfg = NavRLPpoRunnerCfg()
    cfg.device = str(env.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    runner = OnPolicyRunner(wrapped, cfg.to_dict(), log_dir=None, device=cfg.device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False},
        map_location=cfg.device,
    )
    model = runner.get_inference_policy(device=cfg.device)

    @torch.inference_mode()
    def policy(observations):
        return model(observations)

    return policy
