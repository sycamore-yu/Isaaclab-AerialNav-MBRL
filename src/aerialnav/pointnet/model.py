"""PointNet without T-Net/BatchNorm, followed by a 192-wide GRU.

The optional fourth point feature is an M-detector label, an extension to the
paper's XYZ input. The last packed channel is validity, never a learned input.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .points import compact_points, segment_amax


class PointNetPolicy(nn.Module):
    body_frame = True
    checkpoint_batch_size = 64
    recurrent_state_size = 192

    def __init__(self, recurrent_model="gru", motion=False):
        super().__init__()
        if recurrent_model != "gru":
            raise ValueError("PointNet requires recurrent_model=gru")
        self.channels = 4 if motion else 3
        self.point_precision = "fp32"
        self.point_backend = "torch"
        self._compiled_pool = None
        self._rays = self._valid_points = self._encoder_calls = 0
        self.point_encoder = nn.Sequential(
            nn.Conv1d(self.channels, 64, 1),
            nn.LeakyReLU(0.05),
            nn.Conv1d(64, 128, 1),
            nn.LeakyReLU(0.05),
            nn.Conv1d(128, 1024, 1),
            nn.LeakyReLU(0.05),
        )
        self.perception_projection = nn.Linear(1024, 192)
        self.v_proj = nn.Linear(10, 192)
        self.gru = nn.GRUCell(192, 192)
        self.fc = nn.Linear(192, 3, bias=False)
        self.act = nn.LeakyReLU(0.05)
        with torch.no_grad():
            self.v_proj.weight.mul_(0.5)
            self.fc.weight.mul_(0.01)

    def configure_execution(self, cfg):
        self.checkpoint_batch_size = cfg.point_microbatch_envs
        self.point_precision = cfg.point_precision
        self.point_backend = cfg.point_encoder_backend
        self._compiled_pool = None
        if self.point_backend == "compile":
            self._compiled_pool = torch.compile(self._pool_compact, dynamic=True, fullgraph=True)
        if self.point_backend == "triton" and self.fc.weight.device.type != "cuda":
            raise ValueError("The Triton PointNet encoder requires CUDA")

    def _pool_compact(self, points, counts):
        with torch.autocast(points.device.type, dtype=torch.bfloat16, enabled=self.point_precision == "bf16"):
            x = F.leaky_relu(F.linear(points, self.point_encoder[0].weight[..., 0], self.point_encoder[0].bias), 0.05)
            x = F.leaky_relu(F.linear(x, self.point_encoder[2].weight[..., 0], self.point_encoder[2].bias), 0.05)
            if self.point_backend == "triton":
                from .triton_linear import linear_128_1024

                x = linear_128_1024(x, self.point_encoder[4].weight[..., 0], self.point_encoder[4].bias)
            else:
                x = F.linear(x, self.point_encoder[4].weight[..., 0], self.point_encoder[4].bias)
        # Unsafe skips a GPU->CPU lengths validation; compact_points constructed exact segment lengths.
        pooled = segment_amax(x.float(), counts)
        # Monotonic LeakyReLU commutes with max, saving an activation over every point/channel.
        return torch.where(counts[:, None] > 0, F.leaky_relu(pooled, 0.05), 0)

    def _pool(self, points, counts):
        if self._compiled_pool is not None:
            return self._compiled_pool(points, counts)
        return self._pool_compact(points, counts)

    def encode(self, points):
        if points.ndim != 3 or points.shape[1] != self.channels + 1 or points.shape[-1] == 0:
            raise ValueError("Expected [batch, XYZ(+motion)+validity, points]")
        packed, counts, _ = compact_points(points)
        return self.perception_projection(self._pool(packed, counts))

    def _recurrent(self, pooled, state, hidden):
        feature = self.act(self.perception_projection(pooled) + self.v_proj(state))
        hidden = self.gru(feature, hidden)
        return self.fc(self.act(hidden)), hidden

    def forward(self, points, state, hidden=None):
        feature = self.act(self.encode(points) + self.v_proj(state))
        hidden = self.gru(feature, hidden)
        return self.fc(self.act(hidden)), hidden

    def forward_checkpointed(self, points, state, hidden):
        if points.ndim != 3 or points.shape[1] != self.channels + 1 or points.shape[-1] == 0:
            raise ValueError("Expected [batch, XYZ(+motion)+validity, points]")
        packed, counts, offsets = compact_points(points)
        self._rays += points.shape[0] * points.shape[-1]
        self._valid_points += offsets[-1]
        outputs = []
        for start in range(0, points.shape[0], self.checkpoint_batch_size):
            end = min(start + self.checkpoint_batch_size, points.shape[0])
            # Only the large point encoder is recomputed; projection/GRU/action head stay in the BPTT graph.
            pooled = checkpoint(
                self._pool, packed[offsets[start] : offsets[end]], counts[start:end], use_reentrant=False
            )
            outputs.append(self._recurrent(pooled, state[start:end], hidden[start:end]))
            self._encoder_calls += 1
        return tuple(torch.cat(values, dim=0) for values in zip(*outputs, strict=True))

    def execution_metrics(self):
        metrics = {
            "lidar_valid_fraction": self._valid_points / max(self._rays, 1),
            "lidar_points_encoded": self._valid_points,
            "point_encoder_calls": self._encoder_calls,
        }
        self._rays = self._valid_points = self._encoder_calls = 0
        return metrics
