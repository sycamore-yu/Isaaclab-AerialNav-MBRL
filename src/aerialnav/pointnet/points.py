"""Lossless ragged point packing with one small host transfer per scan batch."""

from itertools import accumulate

import torch


def compact_points(points):
    if points.ndim != 3 or points.shape[1] < 2 or points.shape[2] == 0:
        raise ValueError("Expected [environments, features+validity, rays]")
    valid = points[:, -1] > 0
    counts = valid.sum(-1)
    # One count-vector synchronization, not nonzero()/item() for every environment or microbatch.
    offsets = [0, *accumulate(counts.cpu().tolist())]
    indices = torch.nonzero_static(valid.flatten(), size=offsets[-1]).flatten()
    measured = points[:, :-1].transpose(1, 2).reshape(-1, points.shape[1] - 1)
    return measured[indices], counts, offsets


class _SegmentAmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, counts):
        maxima = torch.segment_reduce(features, "max", lengths=counts, unsafe=True)
        ctx.save_for_backward(features, counts, maxima)
        return maxima

    @staticmethod
    def backward(ctx, gradient):
        features, counts, maxima = ctx.saved_tensors
        owners = torch.repeat_interleave(
            torch.arange(counts.shape[0], device=counts.device), counts, output_size=features.shape[0]
        )
        winners = features == maxima[owners]
        ties = torch.segment_reduce(winners.to(features.dtype), "sum", lengths=counts, unsafe=True)
        # Torch 2.11's native max-segment backward fails to split negative gradients at ties.
        return torch.where(winners, gradient[owners] / ties[owners].clamp_min(1), 0), None


def segment_amax(features, counts):
    return _SegmentAmax.apply(features, counts)
