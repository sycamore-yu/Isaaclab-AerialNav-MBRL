# Equivalent PointNet execution options

Keep all 5,400 points, the 3→64→128→1024 encoder, LeakyReLU slope 0.05,
validity masking, empty-cloud behavior, GRU192 and the three acceleration outputs.
Equivalent execution changes preserve this function and its gradients within FP32
rounding tolerance. Sampling fewer points, changing the backbone or reducing sensor
frequency changes the training problem and is outside this optimization assessment.

## Prefer the native PyTorch compiler over a new point-cloud dependency

1. Test `torch.compile` on the encoder, rather than compiling the simulator loop.
   Inductor can fuse pointwise activation/masking and reductions and reduce launch
   overhead; `reduce-overhead` can use CUDA graphs, while `max-autotune` searches
   matrix kernels. Warm-up/compile time must be reported separately. FP32 outputs,
   parameter/input gradients and end-to-end BPTT need comparison before adoption.
   Source: [PyTorch 2.11 compile API](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html),
   [official fusion explanation](https://pytorch.org/blog/why-is-pytorch-compile-so-fast-kernel-fusion/).
2. Move the final LeakyReLU after global maximum reduction. Its positive slope makes
   it monotone: `max_p leaky(z_p) = leaky(max_p z_p)`. This runs the last activation
   on B×1024 values instead of B×1024×5400. Keep masks, empty-cloud behavior and
   `amax` tie-gradient semantics. This does not remove the 128→1024 matrix product.
3. A width-one Conv1d is a shared affine map over points; using contiguous B×P×C
   inputs and `F.linear(x, weight.squeeze(-1), bias)` is an algebraic alternative.
   Reuse existing parameters/checkpoint keys rather than reinitializing layers.
   Kernel choice and layout copies determine performance; it is not inherently faster.
   Sources: [Conv1d](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Conv1d.html),
   [Linear](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Linear.html).

## Existing GitHub implementations are not all drop-in replacements

- [fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py)
  uses the familiar Conv1d/global-pooling structure, but its T-Net, BatchNorm and
  pooling choices differ. Copying its whole backbone would change this model.
- [NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) is a mature fused-MLP
  framework. Its standard fully fused configuration has equal hidden widths selected
  from 16/32/64/128 and its documented LeakyReLU slope is 0.01, unlike this encoder's
  unequal widths and slope 0.05. The default precision also differs. It is not a
  parameter-compatible replacement for the current FP32 model.
  [Configuration details](https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md).
- [Pointnet2_PyTorch CUDA operators](https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/pointnet2_utils.py)
  accelerate sampling, grouping and interpolation. This global PointNet does not
  perform those operations; they do not solve its 128→1024 affine/pooling bottleneck.

## Earlier uncompressed probes on 2026-11-02

RTX 4090, FP32, shared GPU, H4. Warm-up excluded; these are small network-only probes,
not isolated or end-to-end training speedups. Probe sources were kept under /tmp;
these earlier measurements preceded valid-point compression. Current implementation and
controlled comparisons are in [encoder optimization](results/pointnet-s0-20261104.md).

| Candidate (batch16, GPU saved inputs) | Median forward/backward | Output maximum absolute difference | Parameter-gradient maximum absolute difference |
|---|---:|---:|---:|
| Current Conv1d | 119.15 ms | reference | reference |
| Pointwise F.linear | 121.44 ms | 1.91e-7 | 6.92e-8 |
| Final pooling before LeakyReLU | 108.33 ms | 0 | 0 |

Inputs included invalid returns containing NaNs and one empty cloud. The earlier probe
did not benchmark a compiler candidate. With lossless valid-only packing, Linear now
operates on a compact matrix; the old dense-layout comparison does not predict its speed.

For batch128/H4, 16-env chunks measured 0.8525 s versus 1.1307 s for a full 128-env
network call. Full-batch peak PyTorch allocation was 12,532.64 MiB. This justified 16-env chunks for the earlier dense encoder. The current compact encoder
uses 64-env chunks after a new comparison. The optimizer/simulator batch remains 128. CPU offload
is disabled. Full activation caching over H160 would require hundreds of GiB;
activation recomputation remains necessary. These historical measurements do not
establish current free GPU headroom; choose the environment count from a live capacity check.
