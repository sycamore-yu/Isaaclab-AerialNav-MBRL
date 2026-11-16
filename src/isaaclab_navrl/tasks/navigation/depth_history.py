"""Depth history using only measured depth, calibration, and estimated camera poses."""

import torch
import torch.nn.functional as F

OBSERVATION_CONTRACT = "delayed_depth_current_odometry_ego_reprojection_v1"
DEPTH_EVENT_OBSERVATION_CONTRACT = "delayed_depth_signed_event_ego_reprojection_v1"
TEMPORAL_DEPTH_EVENT_OBSERVATION_CONTRACT = "delayed_depth_temporal_signed_event_region_pool_v2"


@torch.no_grad()
def reproject_depth(depth, intrinsics, source_rotation, source_position, target_rotation, target_position):
    """Forward-splat image-plane depth into a new optical frame with a z-buffer.

    Rotations map optical (+Z forward, +X right, +Y down) coordinates to world.
    Unobserved pixels remain invalid; this never fills them from scene geometry.
    """
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("depth must have shape (N, 1, H, W)")
    count, _, height, width = depth.shape
    if intrinsics.shape != (count, 3, 3):
        raise ValueError("intrinsics must have shape (N, 3, 3)")
    for rotation, position in ((source_rotation, source_position), (target_rotation, target_position)):
        if rotation.shape != (count, 3, 3) or position.shape != (count, 3):
            raise ValueError("poses must have shapes (N, 3, 3) and (N, 3)")
        if not torch.isfinite(rotation).all() or not torch.isfinite(position).all():
            raise ValueError("camera poses must be finite")
    if not torch.isfinite(intrinsics).all() or (intrinsics[:, (0, 1), (0, 1)] <= 0).any():
        raise ValueError("camera intrinsics must be finite with positive focal lengths")
    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype) + 0.5,
        torch.arange(width, device=depth.device, dtype=depth.dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x))).reshape(3, -1).expand(count, -1, -1)
    rays = torch.linalg.solve(intrinsics, pixels)
    valid_source = torch.isfinite(depth.flatten(2)) & (depth.flatten(2) > 0)
    points = rays * torch.where(valid_source, depth.flatten(2), 0)
    relative_rotation = target_rotation.transpose(1, 2) @ source_rotation
    translation = target_rotation.transpose(1, 2) @ (source_position - target_position).unsqueeze(-1)
    transformed = relative_rotation @ points + translation
    z = transformed[:, 2]
    projected = intrinsics @ transformed
    uv = projected[:, :2] / z[:, None].clamp_min(1e-6)
    u, v = uv.floor().long().unbind(1)
    valid = valid_source[:, 0] & (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    index = v.clamp(0, height - 1) * width + u.clamp(0, width - 1)
    output = depth.new_full((count, height * width), torch.inf)
    output.scatter_reduce_(1, index, torch.where(valid, z, torch.inf), reduce="amin", include_self=True)
    known = torch.isfinite(output).reshape(count, 1, height, width)
    return torch.where(known, output.reshape_as(depth), 0), known


def compensated_depth_features(depth, known):
    """Keep unknown cells distinct from observed far range without a new actor head."""
    inverse = torch.where(known, 3 / depth.clamp(0.3, 24) - 0.6, -1.0)
    return F.adaptive_max_pool2d(inverse, (12, 16))


@torch.no_grad()
def depth_motion_events(current, previous_in_current, known, *, threshold_m=0.08):
    """Cheap approaching/entering-surface cue, not the M-detector algorithm.

    shortcut: recession and newly exposed surfaces are not classified; use a
    multi-frame detector when those event types are required.
    """
    if threshold_m <= 0:
        raise ValueError("threshold_m must be positive")
    comparable = known & torch.isfinite(current) & (current > 0)
    threshold = threshold_m + 0.01 * previous_in_current
    return comparable & (current < previous_in_current - threshold)


def depth_event_observation(depth, events):
    """Two demo channels; each event bit belongs to the selected nearest depth ray."""
    if depth.shape != events.shape or events.dtype != torch.bool:
        raise ValueError("events must be a boolean mask matching depth")
    pooled, indices = F.adaptive_max_pool2d(3 / depth.clamp(0.3, 24) - 0.6, (12, 16), return_indices=True)
    selected = events.flatten(2).gather(2, indices.flatten(2)).reshape_as(pooled)
    return torch.cat((pooled, selected.to(pooled)), dim=1)


@torch.no_grad()
def pooled_signed_event_channel(depth, signed, known=None, *, region_pool=False):
    """Pool full-resolution event evidence, optionally preserving sparse motion edges.

    Legacy nearest-ray pooling is retained for v1 checkpoints. Region pooling
    means an event exists *somewhere* in the cell, not that its closest ray moved.
    """
    if depth.ndim != 4 or depth.shape[1] != 1 or signed.shape != depth.shape:
        raise ValueError("Expected depth and signed events [N,1,H,W]")
    inverse = 3 / depth.clamp(0.3, 24) - 0.6
    if known is not None:
        if known.shape != depth.shape:
            raise ValueError("known must match depth")
        inverse = torch.where(known, inverse, -1.0)
        signed = torch.where(known, signed, 0)
    if region_pool:
        positive = F.adaptive_max_pool2d(signed.clamp_min(0), (12, 16))
        negative = F.adaptive_max_pool2d((-signed).clamp_min(0), (12, 16))
        return torch.where(positive >= negative, positive, -negative)
    _, index = F.adaptive_max_pool2d(inverse, (12, 16), return_indices=True)
    return signed.flatten(2).gather(2, index.flatten(2)).reshape(depth.shape[0], 1, 12, 16)


@torch.no_grad()
def reproject_signed_event_channel(
    delivered_depth, delivered_event, intrinsic, source_rotation, source_position, target_rotation, target_position,
    aligned_depth, aligned_known,
):
    """Transport delayed events with their captured depth; do not infer any later motion.

    Reuses the same source and target estimates as delayed depth reprojection.
    Only the nearest (visible) event surface is used in the new view.
    """
    if delivered_event.shape != delivered_depth.shape:
        raise ValueError("Delayed event and depth must have identical image dimensions")
    images = []
    for sign in (1, -1):
        source = torch.where(delivered_event * sign > 0, delivered_depth, torch.inf)
        surface, valid = reproject_depth(
            source, intrinsic, source_rotation, source_position, target_rotation, target_position
        )
        aligned = valid & aligned_known & ((surface - aligned_depth).abs() < 0.15)
        images.append(aligned.to(delivered_depth.dtype))
    return images[0] - images[1]
