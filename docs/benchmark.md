# Historical static navigation benchmark

This record belongs to the earlier static-only draft, not the current rectangular SDF/trapezoid baseline. Its first-arrival training success rate is not the new final-state success rate.

**Result:** 80.74% training success after **2,906.783 s (48 min 27 s)**; best observed report **89.48%** after **11,391.278 s (3 h 09 min 51 s)**. These numbers come from a single training run and are not independent evaluation.

## Reproduction provenance

The immutable original training source was recorded at revision `02ab73d854f367007f22a25a988c83c6f2031684` on 2026-10-30. Its configuration was:

| Setting | Value |
|---|---|
| Method | BPTT with GRU policy, DCOL geometry |
| Scene | 300 static obstacles; 0 dynamic obstacles |
| Parallel environments | 256 |
| Seeds | Training 0; terrain 0 |
| Randomization | Disabled |
| Physics / frontend | PhysX / Torch |
| Physical and policy time step | 0.05 s (20 Hz) |
| BPTT rollout length | 150 policy steps |
| Device | NVIDIA RTX 4090 workstation (verified on the source host after the run) |

The best report was **944 successes, 92 collisions, 19 timeouts, 0 bounds, 0 invalid**, out of 1,055 naturally completed training episodes.

## Measurement

The training process wrote `updates.jsonl` metrics for every update and `training_statistics.jsonl` metrics for completed episodes. Join both by the integer `update` field.

- Threshold time: first complete-episode report with `success_rate >= threshold`.
- Wall clock: `process_elapsed_s` at that update. This includes startup and initialization; it excludes time queued before the process.
- Environment steps: `environment_steps` from the same update.
- Average throughput: `environment_steps / process_elapsed_s`. This is **not** simulated time, instantaneous FPS, or independent GPU kernel throughput.

At 80.74%: 8,601,600 steps / 2,906.783 s = **2,959.15 environment steps/s**.

At 89.48%: 30,643,200 steps / 11,391.278 s = **2,690.06 environment steps/s**.

Machine load, other running processes, driver details and training-versus-validation generalization were not fully recorded in the frozen training result. The asset and code were modified after the baseline revision; no claims are made that rerunning current code reproduces the precise time without new measurements.

The structured, sanitized values are in [data/static_training_summary.json](data/static_training_summary.json). Given local `training_statistics.jsonl` and `updates.jsonl` files, the benchmark is recomputable with:

```bash
python scripts/benchmark_training.py \
  --statistics <training_statistics.jsonl> \
  --updates <updates.jsonl>
```

The trained weights in `weights/best_static_s0.pt` are a sanitized inference-only export. No historical logs, original paths or optimizer/recovery state are distributed.

## Next evaluation before an official performance claim

Evaluate the exported policy in at least three independent map seeds, each with at least 1,024 naturally completed episodes. Report success, collision, bounds and timeout rates separately. Treat PhysX as the reference backend and report Newton/Warp metrics separately. Do not call the training success rate a held-out success rate.
