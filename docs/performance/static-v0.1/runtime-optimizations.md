# Static v0.1 training-runtime optimizations

This change contains three runtime-only improvements: fixed-shape first-event bookkeeping (`torch.where` + `copy_`), partial-reset scalar writes (`index_fill_`), and reuse of sorted completed-episode indices during curriculum reductions. The release patch changes five production files (+52/-44 lines); the equivalent patch was ported to `main` under its `isaaclab_navrl` package layout. Neither task definitions nor the neural network, sensors, physics solver, optimizer, or BPTT algorithm were altered by these patches.

## Measured results on the frozen DiffNav benchmark

**GPU synchronization calls decreased by 69.3%**, from **10,253 to 3,147** in the instrumented complete-update comparison. Synchronization API host time dropped from **12.963 s to 3.941 s**. This is a profiler observation; it is not itself the wall-clock speedup.

**Complete training active-update throughput improved by 1.54–1.64×** in two repeated interleaved wall-clock comparisons:

| Repeat | Baseline seconds/update | Optimized seconds/update | Speed ratio |
|---|---:|---:|---:|
| r1 | 22.796823 | 13.919093 | 1.63781× |
| r2 | 22.610331 | 14.683661 | 1.53983× |

The benchmark used an RTX 4090, **512 environments, horizon 150, 20 Hz**, Newton/MuJoCo Warp, Warp frontend, cuRobo SDF, CNN-GRU192, differentiable point-mass dynamics and BPTT. The `pretrain_s0` environment had **240 static obstacles, no dynamic obstacles**, and **10 s arrival wait**. Each side had 2 warmup and 30 measured updates, with 76,800 environment steps per update. Two native training processes remained GPU-resident and were admitted to run their complete updates in alternating order. The numbers exclude coordination waits, startup, and the final checkpoint.

These numbers **do not establish a 1.54–1.64× speedup for a standalone process, whole training job, or today's five-second-arrival release**. Standalone ABBA comparisons were confounded by changes in external GPU processes; no matched standalone pair qualified. The `main` port was not independently throughput-benchmarked. The release's five-second-arrival protocol has separate numerical and fixed-evaluation checks, but no new complete-update throughput result. Training-quality tests were finite, but did not establish long-run convergence equivalence.

## Evidence and verification

The underlying study is preserved in `perf/static-v0.1-training-validated`, at `docs/performance/static-v0.1/REPORT.md`, with paired measurements in `paired_training_comparison.md` and annotated synchronization measurements in `module_comparison/module_profile_comparison.md`. Study sources were frozen as B0 `6c115fa` and B5 `8b661c3`, derived from original release baseline `a63e24d`. The validated integration demonstrated 26,269/26,269 numerical tensor comparisons and 22/22 full BPTT gradient comparisons at zero tolerance under the controlled Newton protocol.

The changes entered `release/static-v0.1` as `f37d28e`, `e1dd0bd`, and `e75cba2`, and were ported to the separately structured `main` as `df7732a`, `e69b5e2`, and `81a26f1`. Both keep their original task configurations. The measured claims above describe the frozen DiffNav study, not an unmeasured new benchmark on the branch heads.
