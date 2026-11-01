# Train the direct baseline

Start from random weights on the fixed randomized navigation task:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_diffnav.py \
  --log_dir runs/baseline --stage finetune_s1 --fresh_stage \
  --num_envs 512 --seed 0 --terrain_seed 0
```

`configs/baseline_direct.json` records the selected baseline. Defaults are Newton/Warp, cuRobo SDF, trapezoidal surrogate integration, pre-reset losses, CNN/GRU192, H150 at 20 Hz, a full 2-second velocity window, and original avoidance gradients. The goal-speed reference decreases linearly near the goal.

Fresh direct training uses AdamW, learning rate 0.001, weight decay 0.01, cosine period 400 updates, and gradient clipping at 5. Loss weights are velocity 1, velocity prediction 2, avoidance 3, collision 7.5, acceleration 0.01, jerk 0.001, and goal-height band 1.

## Choose GRU or SRU

The default `--recurrent_model gru` is the published baseline. To train the optional SRU-LSTM-Gated backend from scratch:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_diffnav.py \
  --log_dir runs/sru --stage finetune_s1 --fresh_stage \
  --num_envs 512 --seed 0 --terrain_seed 0 \
  --recurrent_model sru_lstm_gated
```

SRU supports BPTT only; SHAC/AHAC are rejected before launch. Its recurrent state packs both hidden and cell states (384 values versus GRU's 192). The checkpoint records the backend for automatic inference loading. Start SRU from fresh weights; the packaged baseline is GRU.

## Reports and continuation

Each report contains at least 1024 naturally completed episodes, including failures. `training_statistics.jsonl` holds the report denominators; `updates.jsonl` records motion, losses and optimizer diagnostics. TensorBoard events are written beneath the run directory.

Interruptions save `checkpoint_stopped.pt` at a complete update boundary. Continue into a **new** output directory:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_diffnav.py \
  --log_dir runs/continued --stage finetune_s1 --num_envs 512 \
  --seed 0 --terrain_seed 0 \
  --resume_checkpoint runs/baseline/checkpoint_stopped.pt
```

Continuation checks the stage, environment count, seeds and training configuration; it restores weights, optimizer and learning-rate phase. Environment state, recurrent history and report windows are rebuilt. Never overwrite a previous run.

Export public inference weights without optimizer state, local paths or training history:

```bash
PYTHONPATH=src .venv/bin/python scripts/export_weights.py \
  --input runs/baseline/checkpoint_best_finetune_s1.pt \
  --output runs/policy.pt
```
