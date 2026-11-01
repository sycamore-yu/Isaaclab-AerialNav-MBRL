# Release baseline status

The selected profile is [baseline_direct](../configs/baseline_direct.json): dynamic rectangular navigation, original avoidance gradients, linear goal-speed reference, cuRobo SDF/trapezoidal surrogate BPTT, Newton/Warp, and final-state success after the arrival wait.

| Area | Evidence |
|---|---|
| Public API and default configuration | `aerialnav` / `diffnav`; `variant=terminal`; baseline-profile contract test |
| Baseline checkpoint | `weights/baseline_direct.pt`, update 2129; exact CPU outputs and recurrent state at batches 1, 4 and 17 |
| Legacy checkpoint | `weights/best_static_s0.pt` remains readable; its benchmark is historical |
| Optional SRU | BPTT backend, checkpoint/state lifecycle and pinned upstream numerical parity: 10 checks pass; GRU remains the default |
| Source and metadata privacy | Full source/docs scan, decoded USD metadata, and both checkpoint metadata scans pass |
| Regression | 410 passed, 10 skipped, 85 subtests; skipped tests require optional upstream sources |
| Public export boundary | 10 targeted checks pass, including rejection of unrecognised configuration metadata |
| Geometry and packaging | cuRobo geometry checks pass; locked dependency resolution, wheel build/import and packaged asset/notices verified |
| Simulator smoke | User-authorized EULA acceptance; N4 Newton/Warp training completed 2 updates, no nonfinite states/gradients; packaged baseline inference completed 4 episodes (2 success, 2 timeout) |
| Held-out navigation result | Not established for the exported baseline checkpoint; no performance rate advertised |
| Flight demos | Static and moving-obstacle successful rollouts with the packaged checkpoint, real-time MP4 and GIF previews; [recording details](data/flight_demos.json) |
| Remote publication | Not performed |

The old Static Gym IDs belonged to the static-only draft. Current task IDs are `Isaac-AerialNav-Depth-v0` and `Isaac-AerialNav-PPO-v0`. File-based training/evaluation/export entrypoints remain available. The optional PPO training entrypoint keeps its static profile; it is not a matched dynamic-task comparison.

The simulator checks are startup/gradient/inference checks, not a reliable success-rate estimate. See [validation evidence](data/release_validation.json); raw simulator logs stay in the local `runs/release_checks/` directory. A fresh installation into an empty environment has not been tested.

Review the [publishing workflow](publishing.md) before copying changes to the standalone public repository. Do not merge private development ancestry into this release branch.
