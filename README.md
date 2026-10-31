# AerialNav: Static Obstacle Avoidance with Isaac Lab

A reproducible quadrotor **static-obstacle avoidance** research implementation using depth observations, backpropagation through time (BPTT), and explicit collision geometry. The default public configuration is **300 static obstacles, zero dynamic obstacles**, and a 20 Hz control/policy loop.

This project uses the neutral Python package **`aerialnav`**, an independently registered Isaac Lab task, and the **DiffNav** learning method. A separate PPO task is included as an optional baseline. It is not a direct optimizer-only ablation: the methods use different actions, observations and control rates.

## Measured time-to-success (single training run)

| Measured milestone | Training success rate | End-to-end process wall clock | Updates | Environment steps |
|---|---:|---:|---:|---:|
| First report at or above 80% | 80.74% (851/1,054) | **48 min 27 s** | 224 | 8.60M |
| Highest training report | 89.48% (944/1,055) | **3 h 09 min 51 s** | 798 | 30.64M |

These are **training success rates over naturally completed episodes**, not held-out evaluation results. Both times include process startup and initialization; they exclude any orchestration queue delay. Measured cumulative throughput was **2,959 environment steps/s** at the first 80% report and **2,690 environment steps/s** at the best report, using 256 parallel environments and the PhysX/Torch configuration. The run was recorded on a workstation equipped with an NVIDIA RTX 4090; GPU identity was checked on the original workstation after training, not stored in the run JSON.

No speedup relative to another paper or backend is claimed. [Benchmark methodology, source revision, limitations, and sanitized evidence](docs/benchmark.md).

## Reproducible software setup

- Linux x86-64, NVIDIA GPU, Python 3.12.
- Isaac Lab 3.0.0 and its Isaac Sim dependencies as locked in `pyproject.toml` and `uv.lock`.
- Recommended baseline: **PhysX + Torch**. Newton + Warp is available as an experimental backend; its throughput or policy success rate is *not* implied by the PhysX result.

Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
```

First simulator startup may require accepting the NVIDIA Omniverse license agreement. The local release preparation did **not** accept it automatically.

Run these commands from the repository root. Isaac Lab/Isaac Sim require a working Linux GPU installation; `uv sync` alone does not verify that the simulator will start.

## Evaluate the released policy

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate.py \
  --method diffnav --checkpoint weights/best_static_s0.pt \
  --static_obstacles 300 --dynamic_obstacles 0 \
  --episodes 1024 --seed 10000 --terrain_seed 10000 \
  --output runs/eval_static_seed10000.json \
  physics=isaacsim_physx
```

Change both seeds to additional **unseen seeds**, and report success, collision, bounds, and timeout separately. The released checkpoint is adapted from the original best static S0 checkpoint with **bitwise-identical network outputs** on three sampled CPU batches. Its held-out success rate on this reorganized repository is **not yet established**.

For an environment-only check:

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke.py \
  --method diffnav --num_envs 2 --steps 8 --dynamic_obstacles 0 \
  --output runs/smoke_static.json physics=isaacsim_physx
```

## Train from scratch

```bash
PYTHONPATH=src .venv/bin/python scripts/train_diffnav.py \
  --log_dir runs/train_static_seed0 \
  --stage pretrain_s0 --num_envs 256 \
  --seed 0 --terrain_seed 0 \
  physics=isaacsim_physx
```

The default public profile uses 300 static obstacles, zero dynamic obstacles, no domain randomization, a 20 Hz policy loop, 150-step BPTT, GRU memory, and the goal-dependent height band. Use `--max_iterations 2` for a **smoke run only**; a two-update smoke run does not reproduce the published training metrics.

To create an inference-only checkpoint from a **new-format** DiffNav training checkpoint:

```bash
PYTHONPATH=src .venv/bin/python scripts/export_weights.py \
  --input runs/train_static_seed0/checkpoint_best_pretrain_s0.pt \
  --output runs/policy_export.pt
```

The checkpoint exporter only accepts the current public checkpoint schema. Legacy source checks are not retained in the public runtime.

## Layout

- `src/aerialnav/diffnav/`: depth policy, BPTT, surrogate dynamics, collision geometry, checkpoint inference. See [architecture](docs/architecture.md).
- `src/aerialnav/tasks/navigation/`: Isaac Lab environment, static terrain, sensors and backend interfaces.
- `src/aerialnav/agents/`: optional PPO baseline.
- `scripts/`: training, evaluation, checkpoint export, benchmarking, and smoke checks.
- `weights/best_static_s0.pt`: compact inference-only pretrained weights (2.1 MB).
- `docs/data/static_training_summary.json`: sanitized training time evidence.
- `tests/`: CPU contract tests; optional upstream parity tests use explicitly configured external checkouts.

## Provenance, limitations, and license

The flight dynamics and terrain implementation build upon openly licensed work by the Isaac Lab, OmniDrones, and original NavRL teams. The optional SRU cell retains its original MIT attribution. The `hummingbird.usd` asset is an exact SHA-256 match to the OmniDrones asset included in the NavRL third-party checkout. See [third-party notices](THIRD_PARTY_NOTICES.md) and the preserved license files.

This is a **static-navigation v0.1.0 candidate**, not a verified multi-seed, cross-backend comparison and not a proof of real-world safety. The public identifiers intentionally differ from the names used by prior internal experiments; existing private training artifacts are neither renamed nor modified.

Project contributions are offered under BSD-3-Clause, subject to the retained upstream notices. No code, pretrained weights, or history has been pushed to GitHub by this local release preparation.
