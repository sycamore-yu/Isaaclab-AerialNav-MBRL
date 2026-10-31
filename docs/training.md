# Training

Use the default static profile `configs/static_s0.json`. The authoritative values reside in the public Python task configuration; the JSON is a human-readable record, not a separate launcher configuration.

```bash
PYTHONPATH=src .venv/bin/python scripts/train_diffnav.py \
  --log_dir runs/train_static_seed0 --stage pretrain_s0 \
  --num_envs 256 --seed 0 --terrain_seed 0 physics=isaacsim_physx
```

Every training update uses 256 parallel environments and 150 policy steps at 20 Hz for the documented baseline. The trainer records `training_statistics.jsonl`, `updates.jsonl` and TensorBoard data.

Do not equate a decreasing loss with successful avoidance. Inspect first-safe-arrival, collision, bounds and timeout rates on natural episode terminations. Use independent scene seeds for final evaluation. Changing the backend, policy time step, number of environments, obstacle geometry or reward settings invalidates a direct wall-clock comparison.
