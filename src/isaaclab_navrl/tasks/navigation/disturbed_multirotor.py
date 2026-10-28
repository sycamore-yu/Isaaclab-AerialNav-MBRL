"""Multirotor extension applying Zhang's randomized aerodynamic forces."""

import torch
import warp as wp
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_contrib.assets import Multirotor


class DisturbedMultirotor(Multirotor):
    """Add a world force to the native rotor wrench before its PhysX write."""

    def _initialize_impl(self):
        super()._initialize_impl()
        self.disturbance_force_w = torch.zeros(self.num_instances, 3, device=self.device)
        masses = self.root_view.get_masses()
        masses = wp.to_torch(masses) if isinstance(masses, wp.array) else masses
        self.total_mass = masses.to(self.device).sum(-1, keepdim=True)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.disturbance_force_w[slice(None) if env_ids is None else env_ids] = 0

    def set_disturbance_acceleration(self, acceleration_world):
        self.disturbance_force_w.copy_(acceleration_world * self.total_mass)

    def _combine_thrusts(self):
        super()._combine_thrusts()
        self._internal_force_target_sim[:, 0] += quat_apply_inverse(
            self.data.root_quat_w.torch, self.disturbance_force_w
        )
