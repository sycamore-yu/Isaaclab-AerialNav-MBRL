"""Zhang CNN policy with the original GRU or an SRU-LSTM-Gated memory.

The six outputs are interleaved [ax, vx, ay, vy, az, vz], not two contiguous
three-vectors. Inputs retain the ten-dimensional odometry configuration.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .sru import LSTMSRUGateCell


class ZhangPolicy(nn.Module):
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
        self.dynamic_aux_head = None
        self.dynamic_aux_sectors = 0
        self.aux_from_perception = False
        self.kinematic_stem = None
        self.kinematic_proj = None
        with torch.no_grad():
            self.v_proj.weight.mul_(0.5)
            self.fc.weight.mul_(0.01)

    def enable_kinematic_residual_stream(self, kinematic_channels: int = 3):
        """Keep 1x12x16 inv-depth stem intact and add zero-initialized 24x32 kinematic branch."""
        if self.kinematic_stem is not None:
            raise ValueError("Kinematic residual stream can only be enabled once")
        ref = self.fc.weight
        # Preserve sparse event/looming peaks cleanly across 24x32 -> 12x16 -> 6x8 -> 192.
        self.kinematic_stem = nn.Sequential(
            nn.Conv2d(kinematic_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(64 * 4 * 6, 192, bias=False),
        ).to(device=ref.device, dtype=ref.dtype)
        self.kinematic_proj = nn.Linear(192, 192, bias=False).to(device=ref.device, dtype=ref.dtype)
        with torch.no_grad():
            self.kinematic_proj.weight.zero_()
        self.register_buffer("kinematic_residual_scale", torch.ones((), device=ref.device, dtype=ref.dtype))

    def enable_dynamic_aux_head(self, num_sectors: int = 8, *, from_perception: bool = False):
        """Attach training-only ego dynamic obstacle head on hidden or perception branch."""
        if num_sectors < 1:
            raise ValueError("num_sectors must be a positive integer")
        self.dynamic_aux_sectors = int(num_sectors)
        self.aux_from_perception = bool(from_perception)
        self.dynamic_aux_head = nn.Linear(192, self.dynamic_aux_sectors * 3).to(
            device=self.fc.weight.device, dtype=self.fc.weight.dtype
        )

    def predict_dynamic_aux(self, state_repr: torch.Tensor) -> torch.Tensor:
        """Predict [N, num_sectors, 3] dynamic obstacle features from recurrent or perception embedding."""
        if self.dynamic_aux_head is None:
            raise RuntimeError("dynamic_aux_head is not enabled on this policy")
        if self.aux_from_perception:
            emb = state_repr
        else:
            emb = state_repr if self.recurrent_model == "gru" else state_repr[:, :192]
        return self.dynamic_aux_head(self.act(emb)).reshape(len(emb), self.dynamic_aux_sectors, 3)

    def _encode_visual(self, depth_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.kinematic_stem is None:
            return self.stem(depth_features), None
        # ch0 in [N, 4, 24, 32] pools identically to 12x16 because 24x32 -> 12x16 is exact 2x2 max-pool.
        base_depth = nn.functional.adaptive_max_pool2d(depth_features[:, :1], (12, 16))
        base_embed = self.stem(base_depth)
        kin_embed = self.kinematic_stem(depth_features[:, 1:])
        return base_embed + self.kinematic_residual_scale * self.kinematic_proj(kin_embed), kin_embed + base_embed.detach()

    def _step_recurrent(self, depth_features, state, hidden):
        visual_embed, perception_embed = self._encode_visual(depth_features)
        feature = self.act(visual_embed + self.v_proj(state))
        if self.recurrent_model == "gru":
            h = self.gru(feature, hidden)
            hidden = h
        else:
            if hidden is None:
                hidden = feature.new_zeros(len(feature), self.recurrent_state_size)
            h, c = self.sru(feature, *hidden.split(192, dim=-1))
            # Pack both states so existing per-environment reset/detach/probes cover the full memory.
            hidden = torch.cat((h, c), dim=-1)
        return self.fc(self.act(h)), hidden, perception_embed

    def forward(self, depth_features, state, hidden=None):
        """Return six outputs and complete [N, recurrent_state_size] memory.

        GRU memory is h; SRU memory packs [h, c]. Only h feeds the output head.
        """
        output, hidden, _ = self._step_recurrent(depth_features, state, hidden)
        return output, hidden

    def forward_with_aux(self, depth_features, state, hidden=None):
        """Return (action_output, hidden, aux_input) for decoupled aux supervision."""
        output, hidden, perception_embed = self._step_recurrent(depth_features, state, hidden)
        aux_input = perception_embed if (self.aux_from_perception and perception_embed is not None) else hidden
        return output, hidden, aux_input

    def forward_checkpointed(self, depth_features, state, hidden):
        """Recompute neural activations during backward; never replay the physical simulation."""
        return checkpoint(self, depth_features, state, hidden, use_reentrant=False)

    def forward_checkpointed_with_aux(self, depth_features, state, hidden):
        """Checkpointed forward returning (output, hidden, aux_input)."""
        return checkpoint(self.forward_with_aux, depth_features, state, hidden, use_reentrant=False)


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


def policy_state(
    position,
    velocity,
    quaternion,
    target,
    max_speed,
    margin,
    *,
    body_frame=False,
    goal_speed_profile="linear",
    goal_approach_time_s=2.0,
    goal_clear_weight=None,
):
    rotation = quaternion_matrix(quaternion.detach())
    if body_frame:
        heading = rotation
    else:
        forward = rotation[:, :, 0].clone()
        forward[:, 2] = 0
        forward = torch.nn.functional.normalize(forward, dim=-1)
        up = torch.zeros_like(forward)
        up[:, 2] = 1
        heading = torch.stack((forward, torch.cross(up, forward, dim=-1), up), dim=-1)
    from .goal import goal_velocity

    target_velocity = goal_velocity(
        position,
        target,
        max_speed,
        profile=goal_speed_profile,
        approach_time_s=goal_approach_time_s,
        clear_weight=goal_clear_weight,
    )
    local_velocity = torch.einsum("bi,bij->bj", velocity, heading)
    local_target = torch.einsum("bi,bij->bj", target_velocity, heading)
    state = torch.cat((local_velocity, local_target, rotation[:, 2], margin[:, None]), dim=-1)
    return state, heading, target_velocity, rotation


def decode_action(output, heading, thrust_error):
    gravity = output.new_tensor((0.0, 0.0, -9.80665))
    if output.shape[-1] == 3:
        acceleration = torch.einsum("bij,bj->bi", heading, output)
        return (acceleration - gravity) * thrust_error + gravity, torch.zeros_like(acceleration)
    acceleration, velocity_prediction = (heading @ output.reshape(-1, 3, 2)).unbind(-1)
    command = (acceleration - velocity_prediction - gravity) * thrust_error + gravity
    return command, velocity_prediction
