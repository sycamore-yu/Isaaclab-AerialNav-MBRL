# Zhang default: pre-reset loss and height-bound reset

The user selected pre-reset loss construction with height-bound reset after inspecting the previous four-cell study. Its operator pause receipt is retained at the study's `operator_pause.json`; the four stopped checkpoints remain at updates 316, 316, 280 and 285. This change does not restart them or declare that they completed their planned 512 updates.

The original depth CNN/GRU BPTT path with full-window velocity now defaults to `loss_timing=pre_reset`; the environment still resets on its original height bounds. Loss construction runs after full terminal snapshots and before reset. The optimizer still updates once per rollout. Velocity-window normalization, detached cross-update prefix, loss weights and collision-substep sampling remain unchanged. Explicit `--loss_timing deferred` is available for comparison. SHAC/AHAC and other velocity supervision modes retain their previous deferred default; legacy stage checkpoints preserve their saved timing identity.

Validation: 497 tests and 531 subtests passed (one skipped) in the full CPU suite, with Ruff and diff checks clean. A real Newton/Warp fresh fS1 N4/H150 run without a loss-timing override completed two updates with `loss_timing=pre_reset`, `height_bounds_reset=true`, and zero nonfinite-gradient events. Raw logs are `logs/default_pre_reset_cpu_suite_v2.log` and `logs/2026-10-31_default_pre_reset_validation`. The earlier regression failure log is retained separately. These are implementation checks, not a new policy acceptance claim.

The SDF and position-integration factorial comparison remains a separate change, so this default commit does not bundle the in-progress collision/integrator work or point-policy features.
