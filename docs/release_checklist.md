# Release baseline status

The selected profile is [baseline_direct](../configs/baseline_direct.json): dynamic rectangular navigation, original avoidance gradients, linear goal-speed reference, cuRobo SDF/trapezoidal surrogate BPTT, Newton/Warp, and final-state success after the arrival wait.

| Area | Evidence |
|---|---|
| Public API and default configuration | `aerialnav` / `diffnav`; `variant=terminal`; baseline-profile contract test |
| Baseline checkpoint | `weights/baseline_direct.pt`, update 2129; exact CPU outputs and recurrent state at batches 1, 4 and 17 |
| Legacy checkpoint | `weights/best_static_s0.pt` remains readable; its benchmark is historical |
| Source and metadata privacy | Full source/docs scan, decoded USD metadata, and both checkpoint metadata scans pass |
| Regression | 409 passed, 10 skipped, 85 subtests; skipped tests require optional upstream sources |
| Public export boundary | 10 targeted checks pass, including rejection of unrecognised configuration metadata |
| Geometry and packaging | cuRobo geometry checks pass; locked dependency resolution, wheel build/import and packaged asset/notices verified |
| Simulator startup | Blocked by NVIDIA Omniverse EULA acceptance; no simulator result has been inferred from unit tests |
| Held-out navigation result | Not established for the exported baseline checkpoint; no performance rate advertised |
| Remote publication | Not performed |

The old Static Gym IDs belonged to the static-only draft. Current task IDs are `Isaac-AerialNav-Depth-v0` and `Isaac-AerialNav-PPO-v0`. File-based training/evaluation/export entrypoints remain available. The optional PPO training entrypoint keeps its static profile; it is not a matched dynamic-task comparison.

Review the [publishing workflow](publishing.md) before copying changes to the standalone public repository. Do not merge private development ancestry into this release branch.
