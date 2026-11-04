"""Load release-synchronized PointNet checkpoints for recurrent XYZ inference."""

from aerialnav.diffnav.inference import DiffNavInference

from .checkpoint import validate_checkpoint
from .config import CHECKPOINT_METHODS, OBSERVATION_CONTRACTS, PointNetTrainCfg
from .model import PointNetPolicy


class PointNetInference(DiffNavInference):
    method = CHECKPOINT_METHODS[0]
    checkpoint_methods = CHECKPOINT_METHODS
    checkpoint_observation_contracts = OBSERVATION_CONTRACTS
    config_type = PointNetTrainCfg
    model_type = PointNetPolicy

    def validate_checkpoint(self, payload):
        validate_checkpoint(payload, self.env)

    def __init__(self, env, checkpoint):
        super().__init__(env, checkpoint)
        for name, legacy in (
            ("point_microbatch_envs", 16),
            ("point_precision", "fp32"),
            ("point_encoder_backend", "torch"),
        ):
            if name not in self.checkpoint_train_cfg:
                setattr(self.cfg, name, legacy)
        self.model.configure_execution(self.cfg)


def load_pointnet_policy(env, checkpoint):
    return PointNetInference(env, checkpoint)
