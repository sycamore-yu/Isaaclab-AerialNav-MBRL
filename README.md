# Isaaclab-AerialNav-MBRL

An **Isaac Lab 3 extension** that reimplements the vision-based quadrotor navigation method of [Zhang et al. (2025)](https://doi.org/10.1038/s42256-025-01048-0) using differentiable model-based reinforcement learning (MBRL). The Isaac Lab physics backend generates forward trajectories, while a differentiable point-mass surrogate propagates policy gradients through time. Inspired by [Song et al. (CoRL 2024)](https://proceedings.mlr.press/v270/song25a.html), the surrogate state is aligned with the non-differentiated simulator state at each step. CUDA Graph optimizations reduce kernel-launch overhead in the controller and Warp task computations to improve parallel training throughput.

**Train a functional navigation policy in as little as 30 minutes on a single NVIDIA RTX 4090 GPU (0.5 GPU-hours).**

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

The default task has a 120 × 40 m world, 480 static and 32 dynamic obstacles, randomized dynamics, and target speeds of 1–6 m/s. Routes span `18 × target_speed` metres. Training uses 150-step BPTT, a 2-second velocity window, linear goal-speed guidance, and the original avoidance gradient.

Success requires ending within 0.5 m of the goal without failure. Arrival-triggered reset waits until more than 5 seconds after first entry and the drone is currently inside the goal region. Earlier failures and the episode time limit still end the episode.

For a short training check, add `--max_iterations 2`. See [training](docs/training.md) and [evaluation](docs/evaluation.md) for configuration and checkpoint export.

GRU is the default. For optional [SRU-LSTM-Gated](https://github.com/leggedrobotics/sru-pytorch-spatial-learning) BPTT training, add `--recurrent_model sru_lstm_gated`; start from fresh weights rather than a GRU checkpoint.

## Acknowledgements

We would like to express our gratitude to the following projects, which have provided significant support and inspiration for our work:

- [DiffAero](https://github.com/flyingbitac/diffaero) — GPU-accelerated differentiable quadrotor simulation and policy learning.
- [NavRL](https://github.com/Zhefan-Xu/NavRL) — navigation task design and PPO baseline adaptations ([MIT](licenses/NavRL.txt)).

The implementation uses [Isaac Lab](https://github.com/isaac-sim/IsaacLab) for simulation and [cuRobo](https://github.com/NVlabs/curobo) as an external collision-query dependency. Third-party code and assets retain their original licenses:

- [Isaac Lab / ORBIT](https://github.com/isaac-sim/IsaacLab) — simulation integration and terrain helpers ([BSD-3-Clause](licenses/IsaacLab.txt)).
- [OmniDrones](https://github.com/btx0424/OmniDrones) — rotor dynamics and the Hummingbird model ([MIT](licenses/OmniDrones.txt)).
- [SRU](https://github.com/leggedrobotics/sru-pytorch-spatial-learning) — optional SRU-LSTM-Gated cell ([MIT](licenses/SRU.txt)).

These components retain their original copyright notices and licenses, collected in [`licenses/`](licenses/).

## License

Original project code is licensed under [BSD-3-Clause](LICENSE). Third-party components retain their respective licenses, linked above.
