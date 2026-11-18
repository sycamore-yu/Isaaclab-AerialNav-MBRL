from __future__ import annotations

import torch

from isaaclab_navrl.tasks.navigation.depth_history import (
    compensated_depth_features,
    reproject_depth,
)


def _intrinsics():
    return torch.tensor([[[8.0, 0.0, 8.0], [0.0, 8.0, 6.0], [0.0, 0.0, 1.0]]])


def test_known_plane_reprojection_and_disocclusion_never_fabricate_scene_data():
    depth = torch.full((1, 1, 12, 16), 4.0)
    rotation = torch.eye(3)[None]
    moved, known = reproject_depth(
        depth,
        _intrinsics(),
        rotation,
        torch.zeros(1, 3),
        rotation,
        torch.tensor([[0.5, 0.0, 0.0]]),
    )
    assert not known[..., -1].any()
    assert (moved[~known] == 0).all()
    assert torch.allclose(moved[known], torch.full_like(moved[known], 4.0))

    forward, known = reproject_depth(
        torch.full((1, 1, 12, 16), 3.0),
        _intrinsics(),
        rotation,
        torch.zeros(1, 3),
        rotation,
        torch.tensor([[0.0, 0.0, 1.0]]),
    )
    assert torch.allclose(forward[known], torch.full_like(forward[known], 2.0))

    # Unknown pixels stay zero in feature delta; they must not fabricate scene motion.
    features = compensated_depth_features(moved, known)
    assert features.shape == (1, 1, 12, 16)
