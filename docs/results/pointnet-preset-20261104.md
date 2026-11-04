# Native PointNet preset integration — 2026-11-04

The existing navigation environment now accepts `presets=pointnet` in the shared
BPTT entry point and evaluation entry point. Point-cloud task registrations and
independent environment adapters are removed. The release package and research-main
package retain their own base configuration, dynamics, control and curriculum.

## Runtime gates

Both targets were tested on the shared RTX 4090 with `physics=newton_mjwarp`.
The tests use one environment, not a throughput or convergence comparison.

| Gate | Release | Research main |
| --- | --- | --- |
| Native preset discovery; same environment class | passed | passed |
| No selector: original depth actor and PPO reset/step | passed | passed |
| PointNet Torch and Warp observation `[1,4,5400]` | passed | passed |
| Original U8251 inference on both frontends | passed | passed |
| Shared training entry: GPU forward/backward/AdamW update | passed | passed |
| Checkpoint mismatch rejection and optimizer/counter round-trip | passed | passed |

The one-update GPU runs used H2, eager FP32 to bound integration cost. Each produced
finite loss 3.063951015472412, gradient norm 0.007346402853727341 and nonzero parameter
update norm 0.04527235253868941, with zero nonfinite events. The PointNet recipe's
normal defaults remain H160 and BF16/compiled encoding. Main explicitly preserves
the historical PointNet linear goal reference; its existing depth recipe keeps its
own smooth reference. The release height-band label maps to the identical research
main height-band formula on checkpoint import.

Research-main focused suite: 71 passed / 4 CUDA-only skipped; follow-up controls,
sensor alignment and revised recording: 58 passed (83 subtests); new PointNet
checkpoint round-trip: 2 passed. Release focused suite: 74 passed / 5 CUDA-only
skipped; revised contracts, recording and public naming: 9 passed. Skips in these
CPU runs are separate from the actual GPU optimizer and frontend gates above.
The four PointNet CUDA kernel, pooling and BF16-compiler tests were subsequently
run on each target: 4 passed on release and 4 passed on research main. The release
CPU suite also skips a separate legacy CUDA-only base diagnostic.
The original U8251 FP32 eager outputs and GRU hidden state match the frozen
corrected-sensor implementation exactly (maximum errors 0).

## Static obstacle Rerun recording

U8251 was run without learning on seed 0 / terrain seed 0, S0 (240 static, 0 dynamic,
no randomization). The episode succeeded after 503 steps / 25.15 simulation seconds.
The recorded trajectory length was 49.101 m, with 0.491494 m minimum recorded
clearance after the drone margin. This single demonstration is not a success-rate
or retention estimate.

The Rerun recording contains the actual queried obstacle mesh, Hummingbird pose,
flown trajectory and finite measured LiDAR hits in world and body coordinates.
Miss-ray endpoints are never filled or displayed. The trace contains 923,707 hits
across 503 frames; maximum world/body reconstruction error is 5.72e-6 m.
The body-fixed chase and body-cloud views were visually inspected.
The first-view camera was corrected by replacing only the blueprint in the saved
RRD; the original measured episode and the initial-view RRD are retained.

## Evidence and use

Local acceptance records are under `logs/diagnostics/pointnet-preset-integration-20261104/`:
`runtime-queue.log`, per-target/per-mode JSON and console logs, GPU one-update
records, pytest logs, and `main-u8251-parity.json`. The delivered recording is
`logs/recordings/pointnet-static-u8251-20261104/navigation.rrd`, with `trace.npz`,
`trace-validation.json`, `result.json` and `preview.png` beside it.
Use the same `presets=pointnet` selector for training and playback; see
[commands and observation meaning](../pointnet.md).

No old training queues were resumed, and no unrelated GPU workers were paused.
Formal H160 retraining and PhysX runtime recording were not part of this integration.
