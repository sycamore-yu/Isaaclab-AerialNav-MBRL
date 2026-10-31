# Evaluation protocol

The public checkpoint is a static-scene policy. Set `--dynamic_obstacles 0`. Use independent training and evaluation seeds.

For each held-out map seed, evaluate at least 1,024 complete episodes. Count first safe arrival rather than intermediate distance-to-goal, and report successes, collisions, altitude/bounds exits, invalid-state terminations and timeouts. Use no domain randomization at the reference protocol; report any other setting separately.

The original 89.48% figure is a **training** statistic. Until held-out runs have been completed in this repackaged repository, no held-out success rate is claimed.

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate.py \
  --method diffnav --checkpoint weights/best_static_s0.pt \
  --static_obstacles 300 --dynamic_obstacles 0 --episodes 1024 \
  --seed 10000 --terrain_seed 10000 \
  --output runs/eval_seed10000.json physics=isaacsim_physx
```

Run again with `10001`, `10002`. Do not merge results from different backends without showing backend-specific metrics.
