"""Detached running moments and fixed observation transforms for one rollout.

Collect each *actual policy input* once, outside activation checkpointing.  A
rollout uses the snapshot taken before collecting any of its inputs, so later
statistics cannot change its forward pass or its recomputed backward pass.
Depth statistics cover the single channel of pooled policy features; state
statistics are separate for each of the ten features.  This module never
normalizes physical state, targets used by losses, or decoded actions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MomentSnapshot:
    mean: torch.Tensor
    inverse_std: torch.Tensor
    count: torch.Tensor

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        # Only statistics are detached.  The input's BPTT path remains intact.
        return (value - self.mean.to(value)) * self.inverse_std.to(value)


@dataclass(frozen=True)
class ObservationSnapshot:
    depth: MomentSnapshot
    state: MomentSnapshot


class _RunningMoments(nn.Module):
    def __init__(self, shape: tuple[int, ...], device: str | torch.device | None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64, device=device))
        self.register_buffer("m2", torch.zeros_like(self.mean))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64, device=device))
        self.register_buffer("skipped_batches", torch.zeros((), dtype=torch.long, device=device))

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        samples = value.detach().reshape(-1, *self.mean.shape)
        batch_count = samples.shape[0]
        if batch_count == 0:
            return
        if samples.dtype in (torch.float16, torch.bfloat16):
            samples = samples.float()
        variance, batch_mean = torch.var_mean(samples, dim=0, correction=0)
        variance, batch_mean = variance.double(), batch_mean.double()
        finite = torch.isfinite(variance).all() & torch.isfinite(batch_mean).all()
        count = self.count + batch_count
        delta = batch_mean - self.mean
        mean = self.mean + delta * (batch_count / count)
        m2 = self.m2 + variance * batch_count + delta.square() * (self.count * batch_count / count)
        # A nonfinite input still reaches the policy unchanged.  It must not
        # permanently poison moments for later episodes; skip and count the
        # entire affected stream batch, without a device-to-host check.
        self.mean.copy_(torch.where(finite, mean, self.mean))
        self.m2.copy_(torch.where(finite, m2, self.m2))
        self.count.copy_(torch.where(finite, count, self.count))
        self.skipped_batches.add_((~finite).long())

    @torch.no_grad()
    def snapshot(self, variance_floor: float) -> MomentSnapshot:
        ready = self.count > 0
        mean = torch.where(ready, self.mean, torch.zeros_like(self.mean))
        variance = self.m2 / self.count.clamp_min(1)
        inverse_std = torch.where(ready, variance.clamp_min(variance_floor).rsqrt(), torch.ones_like(self.mean))
        # These tensors own their storage; subsequent updates cannot invalidate
        # an autograd saved tensor or alter a checkpoint recomputation.
        return MomentSnapshot(mean.float().clone(), inverse_std.float().clone(), self.count.clone())


class PolicyObservationNormalizer(nn.Module):
    """Checkpointable policy-input normalization, disabled by default.

    ``begin_rollout`` also creates the fixed inference transform after loading
    ``state_dict``.  Inference should call ``eval`` and never collect statistics.
    There is no output clipping: the derivative w.r.t. input is exactly the
    fixed inverse standard deviation.  ``variance_floor`` bounds that gain.
    """

    def __init__(self, enabled: bool = False, variance_floor: float = 1e-4, device: str | torch.device | None = None):
        super().__init__()
        self._validate(enabled, variance_floor)
        self.enabled = enabled
        self.variance_floor = variance_floor
        self.depth_moments = _RunningMoments((), device)
        self.state_moments = _RunningMoments((10,), device)

    @staticmethod
    def _validate(enabled: bool, variance_floor: float) -> None:
        if type(enabled) is not bool or not math.isfinite(variance_floor) or variance_floor <= 0:
            raise ValueError("enabled must be bool and variance_floor must be finite and positive")

    @staticmethod
    def _check_shapes(depth: torch.Tensor, state: torch.Tensor) -> None:
        if depth.ndim != 4 or depth.shape[1] != 1 or state.shape != (depth.shape[0], 10):
            raise ValueError("Expected pooled depth [N,1,H,W] and policy state [N,10]")

    @torch.no_grad()
    def begin_rollout(self) -> ObservationSnapshot | None:
        if not self.enabled:
            return None
        return ObservationSnapshot(
            self.depth_moments.snapshot(self.variance_floor), self.state_moments.snapshot(self.variance_floor)
        )

    def normalize(
        self, depth: torch.Tensor, state: torch.Tensor, snapshot: ObservationSnapshot | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return depth, state
        self._check_shapes(depth, state)
        if snapshot is None:
            raise ValueError("Enabled normalization requires a fixed begin_rollout() snapshot")
        return snapshot.depth.normalize(depth), snapshot.state.normalize(state)

    @torch.no_grad()
    def observe(self, depth: torch.Tensor, state: torch.Tensor) -> None:
        if not self.enabled or not self.training:
            return
        self._check_shapes(depth, state)
        self.depth_moments.update(depth)
        self.state_moments.update(state)

    def get_extra_state(self) -> dict:
        return {"version": 1, "enabled": self.enabled, "variance_floor": self.variance_floor}

    def set_extra_state(self, state: dict) -> None:
        if state.get("version") != 1:
            raise ValueError("Unsupported observation normalization state")
        self._validate(state["enabled"], state["variance_floor"])
        self.enabled = state["enabled"]
        self.variance_floor = state["variance_floor"]
