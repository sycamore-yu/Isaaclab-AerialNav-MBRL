"""Independent task registrations for PPO and differentiable training."""

import gymnasium as gym

gym.register(
    id="Isaac-NavRL-Hummingbird-Direct-v0",
    entry_point="isaaclab_navrl.tasks.navigation.navrl_env:NavRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:NavRLEnvCfg",
        "warp_entry_point": "isaaclab_navrl.tasks.navigation.navrl_env:NavRLEnv",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": "isaaclab_navrl.agents.rsl_rl_cfg:NavRLPpoRunnerCfg",
    },
)
gym.register(
    id="Isaac-NavRL-Zhang-Hummingbird-Direct-v0",
    entry_point="isaaclab_navrl.tasks.navigation.navrl_env:ZhangNavRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ZhangNavRLEnvCfg",
        "bptt_cfg_entry_point": "isaaclab_navrl.pointnet.presets:training_presets",
        "warp_entry_point": "isaaclab_navrl.tasks.navigation.navrl_env:ZhangNavRLEnv",
    },
)
