"""Historical collision metric only; all other training code is shared."""

import torch

from isaaclab_navrl.zhang.runner import ZhangTrainer

from .dcol import hybrid_collision_gap


class DcolReferenceTrainer(ZhangTrainer):
    def _collision_gap(self, query, radius, obstacle_positions=None, *, static_distance=None, query_time=0.0):
        radius = radius.reshape(-1, *([1] * (query.ndim - 2))).expand(query.shape[:-1])
        static_distance = (
            self.env.terrain_distance.signed_distance(query) if static_distance is None else static_distance
        )
        if obstacle_positions is None:
            obstacle_positions = self.env.motion.position
            if not (isinstance(query_time, (int, float)) and query_time == 0):
                offsets = torch.as_tensor(query_time, dtype=query.dtype, device=query.device)
                obstacle_positions = obstacle_positions + offsets[..., None, None] * self.env.motion.velocity
        return hybrid_collision_gap(query, radius, static_distance, obstacle_positions, self.env.motion.size)
