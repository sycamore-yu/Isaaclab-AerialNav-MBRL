# Rectangular 18v navigation and gradient-field comparison

## Task definition

The user approved the 2x2 design and the XY interpretation of D on 2026-10-31. These defaults define a new task distribution; historical square-map results are not acceptance evidence for it.

- World: X=120 m, Y=40 m, centred at zero. Drone centre bounds are X=+/-60, Y=+/-20; height bounds remain (0.2,4.0) m. Torch and captured Warp event predicates use the same bounds.
- The two 6 m end strips contain no static or dynamic obstacle footprints. Route endpoints lie within X=+/-54, Y=+/-20. The specified densities use the full 4800 m2 area, including these strips.
- At every reset, draw target speed v independently from Uniform(1,6) m/s. Choose +/-X with equal probability and an angle Uniform(-8,8) degrees. XY displacement is exactly 18v times that unit direction. Sample the start uniformly from the feasible interval for that displacement. Sample initial/goal heights independently in (0.5,2.5) m. Reject occupied endpoints using a 0.3 m sphere, resampling geometry while retaining v; never clip or shorten D. A bounded rejection failure is explicit.
- Dynamic cuboids/cylinders retain their size categories and nominal 0.5--1.5 m/s speeds. Their origins, goals and reflected XY motion occupy the rectangle, with footprints excluded from the end strips. All drones share the same world.

| Stage | Static draws | Nominal density /m2 | Dynamic | Timeout s | Randomization |
|---|---:|---:|---:|---:|---|
| S0 | 240 | 0.05 | 0 | 30 | off |
| S1 | 360 | 0.075 | 16 | 35 | off |
| S2 | 480 | 0.1 | 32 | 40 | off |
| fS1 | 480 | 0.1 | 32 | 40 | on |
| fS2 | 600 | 0.125 | 48 | 45 | on |

Static generation retains the existing overlap rule; counts denote sampled obstacle placements, not connected components of the resulting mesh. The stage maps use the same seed, with the requested increasing placement counts. Finite timesteps are used only for validation/capacity probes, never for the four training runs.

## Completed tuning and recommended defaults

The old persistent study exhausted its 16 training starts: baseline screening, 11 adaptive candidates, then two baseline and two selected-candidate confirmation seeds. C12 (avoidance=3.5,H256) won screening at 195/256 vs baseline 182/256. Its independent confirmation did not pass the registered success-improvement, safety and retention gates.

Final checkpoint results pool training seeds 20001/20002 and evaluation seeds 11000/11001, 256 naturally completed initial episodes each:

| Setting | Success | Collision | Bounds | Timeout | Hovering subset |
|---|---:|---:|---:|---:|---:|
| baseline H150, avoidance 3.0 | 732/1024 (71.48%) | 271/1024 | 3/1024 | 18/1024 | 1/1024 |
| C12 H256, avoidance 3.5 | 729/1024 (71.19%) | 228/1024 | 3/1024 | 64/1024 | 4/1024 |

C12 reduced final collisions but did not improve final success, and failed safety/retention checks at intermediate checkpoints. Neither seed passed confirmation. Keep the baseline recommendation: H150, avoidance=3.0, collision=7.5, velocity=1.0, velocity prediction=2.0, acceleration=0.01, jerk=0.001, height=1.0; AdamW lr=0.001, weight decay=0.01, cosine period=400 updates, max gradient norm=5. Inherited fine-tuning stages retain lr=5e-5. This is a defensible starting point, not a demonstrated optimum for the new map.

The full terminal results and configuration identities are preserved in [previous_tuning_closeout.json](results/rectangular_gradient_20261031/previous_tuning_closeout.json). Source evidence remains under `/home/tong/tongworkspace/free/.worktrees/isaaclab-navrl/optim-loop-fs1-20261031/.optim-agent-runs/newton-fs1-u616-20261031/`. That study used frozen DCOL/original-integration geometry, the old square world, and speeds 2.25--9.75 m/s.

## Paper adaptation

Source: [arXiv:2604.10548v2, III-E, Eq.27](https://arxiv.org/html/2604.10548v2#S3.SS5). The [official repository](https://github.com/Fanxing-LI/avoidance) is pinned at `aca16362afe8306dc3023ada9bff8f49bd9567d5`; its `debug/angle_trans.py` supplies d_min=0.3 m, d_max=2 m, alpha=0.9. The released nearest-point remapping call is commented out in `envs/NavigationEnv.py:272`, so this implements the paper's stated mechanism rather than claiming to reproduce the release's default training.

The rotation angle is `(pi/2 - clamp(a,0,pi/2))*0.9*(1-exp(-3*(1-x)^2))`, with `x=(clamp(d,0.3,2)-0.3)/1.7`. Apply it to the horizontal avoidance derivative only. Preserve signed distance values, vertical derivatives and gradient norm. Use episode heading at zero speed and a deterministic rightward tie for head-on geometry. Receding/far obstacles are unchanged. This is a surrogate derivative: finite differences of the unchanged forward loss need not equal the reshaped gradient.

The implementation intentionally leaves collision-loss derivatives, geometric collision events, physical contacts, recurrent model, sensors and control unchanged. Both methods use the existing BPTT/surrogate/GRU stack, not the paper's CTBR architecture or training algorithm. Unit tests verify the formula and that loss values/collision gradients remain unchanged while the avoidance gradient becomes lateral.

cuRobo PBA packs each coordinate in 10 bits. Grid spacing is doubled only when a map would exceed 1024 cells on an axis. The new terrain uses 0.125 m, shape 1024x384x64 and 48 MiB of FP16 shared ESDF storage. Metadata records the actual grid; voxel distances retain discretization error.

## Four fresh training arms

| Name | Gradient | Initialization / curriculum |
|---|---|---|
| baseline_direct | original | random weights directly at fS1, stays there |
| baseline_curriculum | original | random weights S0 -> S1 -> S2 -> fS1 -> fS2 |
| reshaped_direct | reshaped avoidance | random weights directly at fS1, stays there |
| reshaped_curriculum | reshaped avoidance | random weights S0 -> S1 -> S2 -> fS1 -> fS2 |

All arms use seed 0, the same recommended hyperparameters, cuRobo/SDF/trapezoid/pre-reset and Newton+Warp. Start from committed main snapshots; do not reuse probe or historical checkpoints. A course stage advances only after a report of at least 1024 completed training episodes reaches >=80%. The last stage continues after reaching 80%; direct fS1 also continues. No training update/step limit and no separate evaluation stage are introduced. This first matrix is a single-seed mechanism comparison.

Run `PYTHONPATH=src .venv/bin/python scripts/launch_rectangular_comparison.py --output logs/curriculum/<new-run-name>` after committing runtime changes. The launcher measures two full fS2/reshaped/H150 updates, starting at N1024. It halves N if OOM occurs or measured per-process-group peak GPU memory plus headroom cannot fit four workers. All four use the same selected N and at least 1024 completed episodes per report. Probe weights are discarded as training inputs, while all probe artifacts remain. The estimate includes four times (peak+256 MiB) plus 2048 MiB reserve; other sessions continue to run, so this is not an isolated throughput measurement.

`capacity.json`, `selected_plan.json`, `launch.json`, `runs/queue.json`, each pipeline's `source/`, stage configs/checkpoints and `initial_rollout.json` identify the live work. Initial receipts record exact route timing error, heading range and endpoint clearance. TensorBoard uses the existing port 6012 and native metric tags under `logs/curriculum`.

## Verification

N32/H256 reshaped smoke on the new fS2 geometry completed two updates with finite losses/gradients and zero nonfinite events. Its initial routes had maximum D/v error 1.91e-6 seconds and maximum heading deviation 7.69 degrees. Evidence: `logs/2026-10-31_rectangular_reshaped_smoke/`. This is startup/backpropagation evidence, not success-rate acceptance. The subsequent committed-source capacity probe uses the selected H150 baseline.

Validation before promotion: full regression 527 passed, 1 skipped, 531 subtests passed; the additional final task suite passed all 13 cases, including clear static end strips. Real CUDA Warp navigation/event parity, graph replay and partial reset passed via `tests/warp_task_parity.py --device cuda:0`. All changed Python files pass Ruff lint and format checks; `git diff --check` passes. Main test logs are `/tmp/navrl-rectangular-complete-suite.log`, `/tmp/navrl-rectangular-static-final.log`, and `/tmp/navrl-rectangular-warp-parity.log`.
