"""DiffNav's depth transform, independent of simulator initialization."""

import torch
import torch.nn.functional as F


def depth_features(depth: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
    """Original inverse depth and pixel noise, followed by 12 x 16 max pooling."""
    image = 3.0 / depth.clamp(0.3, 24.0) - 0.6
    if noise is not None:
        image = image + noise
    return F.adaptive_max_pool2d(image, (12, 16))
