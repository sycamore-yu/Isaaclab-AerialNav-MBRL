"""Release simulator, optimization and logging with PointNet's point-cloud objective."""

from aerialnav.diffnav.runner import DiffNavTrainer

from .checkpoint import policy_contract, validate_checkpoint
from .config import CHECKPOINT_METHODS, OBSERVATION_CONTRACTS
from .losses import pointcloud_transition_costs
from .model import PointNetPolicy


class PointNetTrainer(DiffNavTrainer):
    method = CHECKPOINT_METHODS[0]
    checkpoint_methods = CHECKPOINT_METHODS
    model_type = PointNetPolicy
    observation_contract = OBSERVATION_CONTRACTS[0]
    checkpoint_observation_contracts = OBSERVATION_CONTRACTS

    def __init__(self, env, cfg, log_dir):
        super().__init__(env, cfg, log_dir)
        self.model.configure_execution(cfg)

    def checkpoint_metadata(self):
        return {"policy_contract": policy_contract(self.env)}

    def validate_checkpoint(self, payload):
        validate_checkpoint(payload, self.env)

    def transition_costs(self, history, cfg, velocity_history=None):
        return pointcloud_transition_costs(history, cfg, velocity_history)
