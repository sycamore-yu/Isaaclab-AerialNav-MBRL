# Isaaclab-AerialNav-MBRL

An **Isaac Lab 3 extension** for quadrotor navigation using differentiable model-based reinforcement learning (MBRL). Training gradients propagate through a differentiable point-mass dynamics surrogate, enabling efficient policy optimization.
We further integrate **CUDA Graph optimizations** into the controller and parallel task computations, reducing GPU kernel-launch overhead and improving training throughput.
**Measured single-stage training:** a fresh S0 policy reached **81.30% completed-episode training success** at update **201**, in **28 min 26 s process time** on an RTX 4090 (shared GPU). [Experimental evidence](docs/results/s0-fresh-20261102/README.md). This is **not** the time to finish all five curriculum stages or a held-out benchmark.

## Measured training example (S0 only)

![Fresh S0 training success](docs/media/s0-fresh-20261102.svg)

| Training result | Value |
|---|---:|
| First ≥80% completed-episode training report | **U201** (978/1203 = 81.30%) |
| Time on RTX 4090 | **27m 36.9s trainer / 28m 26.1s including startup** |
| Work completed | 19.76M transitions, 1024 env, H96, Newton/MJWarp + Warp |
| Scope | Fresh `pretrain_s0`, 240 static / 0 dynamic obstacles; **not a held-out test** |

The GPU was **shared with other training jobs**. This is an observed single-run wall-clock measurement, not a controlled isolated-GPU speedup or a claim for S1/S2/F1/F2. See the [full experimental receipt, timing data, and verification script](docs/results/s0-fresh-20261102/README.md).

## Flight demos

[![Watch the complete flight demonstration](docs/media/flight-preview.gif)](docs/media/flight.mp4)


## Installation

Linux, an NVIDIA GPU, and Python 3.12 are required. From the repository root:

```bash
git clone https://github.com/sycamore-yu/Isaaclab-AerialNav-MBRL.git
cd Isaaclab-AerialNav-MBRL
uv sync --frozen --extra curobo
```

The environment pins Isaac Lab 3.0.0, Newton 1.6.1, and the tested cuRobo revision. See [installation](docs/installation.md) for GPU setup.

## Test the Policy

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate.py \
  --checkpoint weights/baseline_direct.pt \
  --stage finetune_s1 --episodes 1024 \
  --seed 10000 --terrain_seed 10000 \
  --output runs/evaluation.json
```

Results include success, collision, bounds, invalid state and timeout for every initial episode. Change both seeds to test another map.

## Train the Policy

```bash
PYTHONPATH=src .venv/bin/python scripts/train_diffnav.py \
  --log_dir runs/baseline --num_envs 512 \
  --stage finetune_s1 --fresh_stage \
  --seed 0 --terrain_seed 0
```

The current `finetune_s1` task has a 120 × 40 m world, 320 static and 160 dynamic obstacles (480 total), randomized dynamics, and target speeds of 1–6 m/s. The packaged `baseline_direct.pt` policy was originally trained under the **historical** 480/32 distribution; testing it with the command above evaluates it on the new distribution, not its original training distribution.

GRU is the default. For optional [SRU-LSTM-Gated](https://github.com/leggedrobotics/sru-pytorch-spatial-learning) BPTT training, add `--recurrent_model sru_lstm_gated`; start from fresh weights rather than a GRU checkpoint.


## Acknowledgements

We would like to express our gratitude to the following projects and research works, which have provided significant support and inspiration for our work:

- [DiffAero](https://github.com/flyingbitac/diffaero) — A GPU-accelerated differentiable simulation framework for efficient quadrotor policy learning.
- [NavRL](https://github.com/Zhefan-Xu/NavRL) — Learning safe flight in dynamic environments using reinforcement learning.
- [Zhang et al. (2025)](https://doi.org/10.1038/s42256-025-01048-0) — Vision-based agile quadrotor flight through differentiable physics training.
- [Song et al.](https://proceedings.mlr.press/v270/song25a.html) — Differentiable policy learning with state alignment in non-differentiable physics simulation.