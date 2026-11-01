# Evaluate a checkpoint

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate.py \
  --checkpoint weights/baseline_direct.pt --stage finetune_s1 \
  --episodes 1024 --seed 10000 --terrain_seed 10000 \
  --output runs/evaluation.json
```

A stage selects its obstacle counts, timeout and randomization. `finetune_s1` matches the baseline's 480 static/32 dynamic randomized task. `pretrain_s1` uses 360/16 without randomization; these are different difficulty levels. Every initial episode receives one final outcome, and reset episodes do not enter the denominator.

The arrival radius is 0.5 m. First entry latches a timestamp; leaving or re-entering does not restart it. After strictly more than 10 seconds, current goal occupancy triggers truncation. Collision, bounds, invalid state or the original time limit can end the episode sooner. Success requires final goal occupancy without failure. First arrival is recorded separately, not scored as final success.

Report all failure classes and the denominator, with the stage, seeds, speed range, geometry backend and checkpoint configuration. Do not compare these results with the previous draft's first-arrival/static training rate.

For an environment-only check, use `scripts/smoke.py`. A short run verifies loading/reset/stepping, not navigation performance. Historical `weights/best_static_s0.pt` remains readable but is not the new dynamic baseline.
