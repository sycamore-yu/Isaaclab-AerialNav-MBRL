# AerialNav

Depth-based quadrotor navigation with differentiable training in Isaac Lab 3.

The baseline uses a CNN–GRU policy, Newton + Warp simulation, and SDF collision losses backed by cuRoboV2. Training gradients pass through a point-mass surrogate, not through the Newton solver.

## Flight demos

| Static obstacles | Moving obstacles |
| --- | --- |
| [![Static flight](docs/media/static-flight.gif)](docs/media/static-flight.mp4) | [![Dynamic flight](docs/media/dynamic-flight.gif)](docs/media/dynamic-flight.mp4) |
| [Full video · 1×](docs/media/static-flight.mp4) | [Full video · 1×](docs/media/dynamic-flight.mp4) |

Successful flights with the packaged GRU policy. Cyan marks the flown path; green marks the goal.

## Installation

Linux, an NVIDIA GPU, and Python 3.12 are required. From the repository root:

```bash
git clone https://github.com/sycamore-yu/Isaaclab-diff-aerialnav.git
cd Isaaclab-diff-aerialnav
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

Results include success, collision, bounds, and timeout for every initial episode. Change both seeds to test another map. The checkpoint is a training baseline; held-out performance is reported separately.

## Train the Policy

```bash
PYTHONPATH=src .venv/bin/python scripts/train_diffnav.py \
  --log_dir runs/baseline --num_envs 512 \
  --stage finetune_s1 --fresh_stage \
  --seed 0 --terrain_seed 0
```

The default task has a 120 × 40 m world, 480 static and 32 dynamic obstacles, randomized dynamics, and target speeds of 1–6 m/s. Routes span `18 × target_speed` metres. Training uses 150-step BPTT, a 2-second velocity window, linear goal-speed guidance, and the original avoidance gradient.

Success requires ending within 0.5 m of the goal without failure. Arrival-triggered reset waits until more than 10 seconds after first entry and the drone is currently inside the goal region. Earlier failures and the episode time limit still end the episode.

For a short training check, add `--max_iterations 2`. See [training](docs/training.md) and [evaluation](docs/evaluation.md) for configuration and checkpoint export.

GRU is the default. For optional SRU-LSTM-Gated BPTT training, add `--recurrent_model sru_lstm_gated`; start from fresh weights rather than a GRU checkpoint.

## Acknowledgements

Built with Isaac Lab, cuRobo, and OmniDrones, with upstream navigation components credited in [third-party notices](THIRD_PARTY_NOTICES.md).

## License

BSD-3-Clause for original project code. Upstream components retain their own licenses.
