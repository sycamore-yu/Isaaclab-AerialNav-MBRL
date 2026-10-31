"""Run upstream PPO updates continuously while retaining its logging state."""

from pathlib import Path

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.logger import Logger

from aerialnav.training import tensorboard_directory


class ContinuousLogger(Logger):
    def init_logging_writer(self):
        if self.writer is None:
            super().init_logging_writer()

    def stop_logging_writer(self):
        # The outer training process, not an individual update, owns the writer.
        pass


class ContinuousNavigationRunner(OnPolicyRunner):
    """Keep native learn/update code; the caller controls curriculum boundaries."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self.logger = ContinuousLogger(
            log_dir=str(tensorboard_directory(log_dir)) if log_dir is not None else None,
            cfg=self.cfg,
            env_cfg=env.cfg,
            num_envs=env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=device,
        )

    def save(self, path, infos=None):
        # Native learn(1) saves at the end of every call. The training driver
        # saves explicit periodic, best-stage and final checkpoints instead.
        if Path(path).name.startswith("model_"):
            return
        super().save(str(path), infos)
