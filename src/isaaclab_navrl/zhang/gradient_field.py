"""Horizontal adaptation of arXiv:2604.10548v2, III-E, Eq. 27."""

import torch
from torch.nn import functional as F


def reshape_avoidance_distance(distance, query, velocity, goal_direction):
    """Keep signed metres; rotate the avoidance derivative in XY.

    Paper parameters: d_min=0.3, d_max=2.0, alpha=0.9. Vertical derivatives
    stay unchanged. At rest use the goal heading; head-on ties turn right.
    """
    normal = torch.autograd.grad(distance.sum(), query, retain_graph=True)[0].detach()
    with torch.no_grad():
        direction = torch.where(
            velocity[..., :2].norm(dim=-1, keepdim=True) > 1e-4, velocity[..., :2], goal_direction[..., :2]
        )
        direction = F.normalize(direction, dim=-1)[:, None]
        toward = -F.normalize(normal[..., :2], dim=-1)
        incidence = (direction * toward).sum(-1).clamp(-1, 1).acos()
        scaled_distance = (distance.detach().clamp(0.3, 2.0) - 0.3) / 1.7
        delta = (
            (torch.pi / 2 - incidence.clamp_max(torch.pi / 2)) * 0.9 * (1 - (-3 * (1 - scaled_distance).square()).exp())
        )
        cross = direction[..., 0] * toward[..., 1] - direction[..., 1] * toward[..., 0]
        delta *= torch.where(cross >= 0, 1.0, -1.0)
        c, s = delta.cos(), delta.sin()
        x, y = normal[..., 0].clone(), normal[..., 1].clone()
        normal[..., 0], normal[..., 1] = c * x - s * y, s * x + c * y
    return distance.detach() + ((query - query.detach()) * normal).sum(-1)
