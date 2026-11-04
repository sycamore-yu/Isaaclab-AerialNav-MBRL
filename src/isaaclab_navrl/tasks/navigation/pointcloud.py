"""Body-frame XYZ with a non-learned validity mask from actual LiDAR hits."""

import torch
from isaaclab.utils.math import matrix_from_quat


def pack_body_points(hits, position, quaternion):
    valid = torch.isfinite(hits).all(-1)
    relative = torch.where(valid[..., None], hits - position[:, None], 0)
    points = torch.einsum("bpi,bij->bpj", relative, matrix_from_quat(quaternion))
    return torch.cat((points, valid[..., None]), -1).transpose(1, 2).contiguous()
