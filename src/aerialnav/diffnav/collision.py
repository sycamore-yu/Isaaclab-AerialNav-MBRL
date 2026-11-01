"""SDF clearances shared by differentiable collision costs and AHAC."""

import torch


def height_bounded_gap(query, obstacle_gap, height_bounds=(0.2, 4.0)):
    """Metre clearance to obstacles and the environment's centre-height bounds."""
    return torch.minimum(
        obstacle_gap, torch.minimum(query[..., 2] - height_bounds[0], height_bounds[1] - query[..., 2])
    )


def metric_navigation_gap(query, static_signed_distance, positions, sizes, radius=0.3):
    """Actual primitive/ground/height clearances for AHAC, from the existing geometry SDF.

    Static terrain retains its documented verified-exit signed proxy. Dynamic
    selection is detached; only the selected primitive keeps a backward graph.
    Positions may be shared or broadcast per-query substep snapshots.
    """
    from aerialnav.tasks.navigation.geometry import nearest_obstacle_signed_distance

    return height_bounded_gap(
        query, nearest_obstacle_signed_distance(query, static_signed_distance, positions, sizes) - radius
    )
