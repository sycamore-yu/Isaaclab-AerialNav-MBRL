"""Depth preprocessing and motion-compensated history."""

import torch

from aerialnav.tasks.navigation.depth_history import (
    compensated_depth_features,
    depth_event_observation,
    depth_motion_events,
    reproject_depth,
)
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


def test_known_plane_reprojection_and_disocclusion_never_fabricate_scene_data():
    depth = torch.full((1, 1, 12, 16), 4.0)
    intrinsics = torch.tensor([[[8.0, 0, 8.0], [0, 8.0, 6.0], [0, 0, 1.0]]])
    rotation = torch.eye(3)[None]
    origin = torch.zeros(1, 3)
    same, known = reproject_depth(depth, intrinsics, rotation, origin, rotation, origin)
    torch.testing.assert_close(same, depth)
    assert known.all()
    # Move right by exactly one pixel at this plane; the new right border is unseen.
    moved, known = reproject_depth(depth, intrinsics, rotation, origin, rotation, torch.tensor([[0.5, 0, 0]]))
    torch.testing.assert_close(moved[..., :-1], torch.full_like(moved[..., :-1], 4.0))
    assert known[..., :-1].all() and not known[..., -1].any()
    assert not moved[..., -1].any()
    assert not depth_motion_events(depth, moved, known).any()
    assert (compensated_depth_features(moved, known)[..., -1] == -1).all()
    # Forward ego translation changes static-plane depth, but is not object motion.
    forward, known = reproject_depth(depth, intrinsics, rotation, origin, rotation, torch.tensor([[0, 0, 1.0]]))
    torch.testing.assert_close(forward[known], torch.full_like(forward[known], 3.0))
    assert not depth_motion_events(torch.full_like(depth, 3.0), forward, known).any()


def test_motion_channel_marks_new_near_surface_not_vacated_background_or_unknown():
    previous = torch.full((1, 1, 12, 16), 5.0)
    current = previous.clone()
    current[..., 4:8, 6:10] = 3.0
    current[..., :2, :2] = 8.0
    known = torch.ones_like(current, dtype=torch.bool)
    known[..., 4, 6] = False
    events = depth_motion_events(current, previous, known)
    expected = torch.zeros_like(known)
    expected[..., 4:8, 6:10] = True
    expected[..., 4, 6] = False
    assert torch.equal(events, expected)


def test_pooled_event_belongs_to_nearest_depth_ray_not_a_different_moving_surface():
    depth = torch.full((1, 1, 120, 160), 5.0)
    events = torch.zeros_like(depth, dtype=torch.bool)
    events[..., 0, 0] = True
    depth[..., 0, 1] = 1.0
    observation = depth_event_observation(depth, events)
    assert observation.shape == (1, 2, 12, 16)
    assert observation[0, 1, 0, 0] == 0
    events[..., 0, 1] = True
    observation = depth_event_observation(depth, events)
    assert observation[0, 1, 0, 0] == 1
