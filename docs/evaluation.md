# Evaluate a checkpoint

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate.py \
  --checkpoint weights/baseline_direct.pt --stage finetune_s1 \
  --episodes 1024 --seed 10000 --terrain_seed 10000 \
  --output runs/evaluation.json
```

A stage selects obstacle counts, timeout and randomization. `finetune_s1` uses 320 static and 160 dynamic obstacles with randomized dynamics. `pretrain_s1` uses 240 static and 120 dynamic obstacles without randomization.

For pretraining difficulty comparisons, select `pretrain_s0`, `pretrain_s1`, or `pretrain_s2`: respectively 240/0, 240/120, and 180/180 static/dynamic obstacles, all without perturbations.

The two fine-tuning stages each contain 480 obstacles: `finetune_s1` uses 320/160 (static/dynamic, 2:1) and `finetune_s2` uses 240/240 (1:1). The world is 120 × 40 m (4800 m²); total stage densities are 5, 7.5, 7.5, 10 and 10 obstacles per 100 m², respectively.

## Episode outcomes

First entry within 0.5 m of the goal records the arrival time. Leaving or re-entering keeps that timestamp. After strictly more than 5 seconds, current goal occupancy triggers reset. Collision, bounds, invalid state and the stage time limit also end episodes. Success requires final goal occupancy without failure.

The output contains one result per initial episode and a summary of success, collision, bounds, invalid state and timeout. Record the checkpoint, stage, seeds and speed range alongside comparisons. Use different terrain seeds for independent maps.

Episode rows also retain `first_event_code` when provided by the environment. Code 1 identifies a safe arrival before any failure under the historical first-event rule; `outcome` and the summary continue to use the final result. Comparing both on the same trajectories separates first arrival from final-goal retention.

## Packaged policy

`weights/baseline_direct.pt` is the GRU direct-training checkpoint at update 2129. Its historical 480/32 training distribution is preserved in [baseline metadata](data/baseline_checkpoint.json) and `configs/baseline_direct.json`. The evaluation command above now tests this historical checkpoint on the **new** 320/160 F1 distribution; it does not reproduce the original training map configuration. The [README flight demos](../README.md#flight-demos) show selected successful episodes. Run the evaluation above to obtain aggregate results.

`weights/best_static_s0.pt` is the earlier static-policy checkpoint. It remains readable for existing workflows. Use `scripts/smoke.py` to check environment loading, reset and stepping.
