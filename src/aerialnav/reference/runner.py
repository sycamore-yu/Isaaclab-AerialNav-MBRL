"""Historical collision metric only; all other training code is shared."""

from aerialnav.diffnav.runner import DiffNavTrainer

from .dcol import hybrid_collision_gap


class DcolReferenceTrainer(DiffNavTrainer):
    def _collision_gap(self, query, radius, obstacle_positions=None, *, static_distance=None):
        radius = radius.reshape(-1, *([1] * (query.ndim - 2))).expand(query.shape[:-1])
        static_distance = (
            self.env.terrain_distance.signed_distance(query) if static_distance is None else static_distance
        )
        obstacle_positions = self.env.motion.position if obstacle_positions is None else obstacle_positions
        return hybrid_collision_gap(query, radius, static_distance, obstacle_positions, self.env.motion.size)
