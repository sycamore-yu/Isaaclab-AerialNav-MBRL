"""The full-resolution closest pixel must survive the DiffNav preprocessing."""

import torch

from aerialnav.tasks.navigation.depth_processing import depth_features


def test_full_resolution_pixel_is_not_replaced_by_a_coarse_center_ray():
    depth = torch.full((1, 1, 120, 160), 24.0)
    depth[0, 0, 41, 81] = 0.5
    features = depth_features(depth)
    expected = torch.full((1, 1, 12, 16), -0.475)
    expected[0, 0, 4, 8] = 5.4
    torch.testing.assert_close(features, expected)


def test_noise_is_added_before_pooling_and_depth_is_clipped_before_inversion():
    depth = torch.full((1, 1, 48, 64), 30.0)
    noise = torch.zeros_like(depth)
    noise[0, 0, 1, 1] = 0.1
    depth[0, 0, 4, 4] = 0.0
    features = depth_features(depth, noise)
    torch.testing.assert_close(features[0, 0, 0, 0], torch.tensor(-0.375))
    torch.testing.assert_close(features[0, 0, 1, 1], torch.tensor(9.4))
