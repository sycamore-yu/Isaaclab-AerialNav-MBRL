"""128→1024 pointwise affine candidate using Triton's tiled matmul pattern.

Forward follows triton-lang's matrix multiplication tutorial (MIT); backward
uses native GEMMs so parameter and upstream gradients remain ordinary PyTorch tensors.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _affine(
    A, W, B, Y, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + ks
        a = tl.load(A + rows[:, None] * K + kk[None, :], (rows[:, None] < M) & (kk[None, :] < K), 0)
        w = tl.load(W + cols[None, :] * K + kk[:, None], (cols[None, :] < N) & (kk[:, None] < K), 0)
        acc += tl.dot(a, w, input_precision="ieee")
    bias = tl.load(B + cols, cols < N, 0).to(tl.float32)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc + bias[None, :], (rows[:, None] < M) & (cols[None, :] < N))


class _Linear128(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        if not x.is_cuda or x.ndim != 2 or x.shape[1] != 128 or weight.shape != (1024, 128):
            raise ValueError("Triton point affine expects CUDA [points,128] and [1024,128] weights")
        x = x.contiguous()
        w, b = weight.to(x.dtype).contiguous(), bias.to(x.dtype).contiguous()
        out = torch.empty((x.shape[0], 1024), device=x.device, dtype=x.dtype)
        if x.shape[0]:
            _affine[(triton.cdiv(x.shape[0], 64), triton.cdiv(1024, 64))](
                x, w, b, out, x.shape[0], 128, 1024, 64, 64, 32, num_warps=4
            )
        ctx.save_for_backward(x, w)
        ctx.weight_dtype, ctx.bias_dtype = weight.dtype, bias.dtype
        return out

    @staticmethod
    def backward(ctx, gradient):
        x, weight = ctx.saved_tensors
        gradient = gradient.contiguous()
        return (gradient @ weight, (gradient.T @ x).to(ctx.weight_dtype), gradient.sum(0).to(ctx.bias_dtype))


def linear_128_1024(x, weight, bias):
    return _Linear128.apply(x, weight, bias)
