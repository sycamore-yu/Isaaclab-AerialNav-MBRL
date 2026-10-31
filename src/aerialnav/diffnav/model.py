"""DiffNav CNN policy with the original GRU or an SRU-LSTM-Gated memory.

The six outputs are interleaved [ax, vx, ay, vy, az, vz], not two contiguous
three-vectors. Inputs retain the ten-dimensional odometry configuration.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .sru import LSTMSRUGateCell


class DiffNavPolicy(nn.Module):
    def __init__(self, recurrent_model="gru"):
        super().__init__()
        if recurrent_model not in ("gru", "sru_lstm_gated"):
            raise ValueError("recurrent_model must be gru or sru_lstm_gated")
        self.recurrent_model = recurrent_model
        self.recurrent_state_size = 192 if recurrent_model == "gru" else 384
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128 * 2 * 4, 192, bias=False),
        )
        self.v_proj = nn.Linear(10, 192)
        if recurrent_model == "gru":
            self.gru = nn.GRUCell(192, 192)
        else:
            self.sru = LSTMSRUGateCell(192, 192)
        self.fc = nn.Linear(192, 6, bias=False)
        self.act = nn.LeakyReLU(0.05)
        with torch.no_grad():
            self.v_proj.weight.mul_(0.5)
            self.fc.weight.mul_(0.01)

    def forward(self, depth_features, state, hidden=None):
        """Return six outputs and complete [N, recurrent_state_size] memory.

        GRU memory is h; SRU memory packs [h, c]. Only h feeds the output head.
        """
        feature = self.act(self.stem(depth_features) + self.v_proj(state))
        if self.recurrent_model == "gru":
            h = self.gru(feature, hidden)
            hidden = h
        else:
            if hidden is None:
                hidden = feature.new_zeros(len(feature), self.recurrent_state_size)
            h, c = self.sru(feature, *hidden.split(192, dim=-1))
            # Pack both states so existing per-environment reset/detach/probes cover the full memory.
            hidden = torch.cat((h, c), dim=-1)
        return self.fc(self.act(h)), hidden

    def forward_checkpointed(self, depth_features, state, hidden):
        """Recompute neural activations during backward; never replay PhysX."""
        return checkpoint(self, depth_features, state, hidden, use_reentrant=False)


def quaternion_matrix(quaternion):
    """Body-to-world matrix for Isaac Lab 3.0 XYZW unit quaternions."""
    x, y, z, w = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def policy_state(position, velocity, quaternion, target, max_speed, margin):
    rotation = quaternion_matrix(quaternion.detach())
    forward = rotation[:, :, 0].clone()
    forward[:, 2] = 0
    forward = torch.nn.functional.normalize(forward, dim=-1)
    up = torch.zeros_like(forward)
    up[:, 2] = 1
    heading = torch.stack((forward, torch.cross(up, forward, dim=-1), up), dim=-1)
    target_raw = target.detach() - position.detach()
    target_norm = target_raw.norm(dim=-1, keepdim=True)
    target_velocity = target_raw / target_norm.clamp_min(1e-6) * torch.minimum(target_norm, max_speed)
    local_velocity = torch.einsum("bi,bij->bj", velocity, heading)
    local_target = torch.einsum("bi,bij->bj", target_velocity, heading)
    state = torch.cat((local_velocity, local_target, rotation[:, 2], margin[:, None]), dim=-1)
    return state, heading, target_velocity, rotation


def decode_action(output, heading, thrust_error):
    acceleration, velocity_prediction = (heading @ output.reshape(-1, 3, 2)).unbind(-1)
    gravity = output.new_tensor((0.0, 0.0, -9.80665))
    command = (acceleration - velocity_prediction - gravity) * thrust_error + gravity
    return command, velocity_prediction
