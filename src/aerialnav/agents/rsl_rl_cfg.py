"""RSL-RL PPO configuration for the optional aerial navigation baseline."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class NavigationModelCfg:
    class_name: str = "aerialnav.agents.models:NavigationActor"
    distribution_cfg: dict | None = {
        "class_name": "rsl_rl.modules.distribution:BetaDistribution",
        "action_range": (-2.0, 2.0),
    }
    stochastic: bool = False  # Consumed by Isaac Lab's official config migration.


@configclass
class NavigationPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 0
    num_steps_per_env = 32
    max_iterations = None
    init_at_random_ep_len = False
    save_interval = 100
    experiment_name = "ppo"
    logger = "tensorboard"
    clip_actions = None
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = NavigationModelCfg()
    critic = NavigationModelCfg(class_name="aerialnav.agents.models:NavigationCritic", distribution_cfg=None)
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="rsl_rl.algorithms:PPO",
        num_learning_epochs=4,
        num_mini_batches=16,
        learning_rate=0.0005,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.001,
        desired_kl=None,
        max_grad_norm=5.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        normalize_advantage_per_mini_batch=False,
        share_cnn_encoders=True,
    )
