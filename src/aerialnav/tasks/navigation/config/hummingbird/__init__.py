"""Independent task registrations for PPO and differentiable training."""

import gymnasium as gym

gym.register(
    id="Isaac-AerialNav-Static-PPO-v0",
    entry_point="aerialnav.tasks.navigation.navigation_env:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:NavigationEnvCfg",
        "warp_entry_point": "aerialnav.tasks.navigation.navigation_warp_env:NavigationWarpEnv",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": "aerialnav.agents.rsl_rl_cfg:NavigationPpoRunnerCfg",
    },
)
gym.register(
    id="Isaac-AerialNav-Static-Depth-v0",
    entry_point="aerialnav.tasks.navigation.navigation_env:DiffNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DiffNavEnvCfg",
        "warp_entry_point": "aerialnav.tasks.navigation.navigation_warp_env:DiffNavWarpEnv",
    },
)
