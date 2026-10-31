# DCOL / SDF and position-integration controls

The latest user instruction fixes **pre-reset loss construction and height-bound reset**, now committed to `main@addcbff`. The operator-paused timing/height study remains stopped with all four checkpoints and statistics retained. The following four new runs start from scratch:

| Control | Collision metric | Differentiable position update |
|---|---|---|
| dcol_original | historical DCOL | original |
| sdf_original | existing geometry SDF | original |
| dcol_trapezoid | historical DCOL | trapezoid |
| sdf_trapezoid | existing geometry SDF | trapezoid |

All cells use the original depth CNN + GRU, six-output head, BPTT, the fixed fS1 scene (300 static and 40 dynamic obstacles with randomization), 256 environments, H150, 512 optimizer updates, seed/terrain seed 0, AdamW, cosine base LR 0.001/period 400, unchanged loss weights, pre-reset construction and height-bound reset. No checkpoint or point policy is loaded. First/last update component gradients are recorded. Final evaluation uses a common original-reset fS1 task and 256 episodes at seed/terrain seed 11000. Raw natural training windows, failure counts and unreported completed episodes remain separate.

Production `ZhangTrainer` has no DCOL alpha branch. It calls the existing `nearest_obstacle_signed_distance` and subtracts the same safety radius at the same look-ahead and first-collision-substep samples. The differentiable soft collision penalty is retained. Dynamic boxes and finite cylinders use their analytic signed distance and deterministic one-sided escape branches; static terrain retains its existing verified-exit signed-distance approximation, not a new exact mesh SDF. The old `variant=dcol` name denotes the existing terminal-sampling protocol only. Shared AHAC clearance helpers moved to `zhang/collision.py`. Historical alpha code is isolated under `reference/` and selected only by the reference training entrypoint. Network-only inference can load historical checkpoints while retaining their original metric metadata.

For the trapezoid factor, only the surrogate position equation changes to `p_next = p + (v + v_next) * dt / 2`. Existing gradient decay on the state identity edge is preserved. The velocity equation, motor response, physical Newton solver, measured-state alignment, event predicates, samples and losses remain unchanged. Thus a fixed action sequence still uses the same real simulator; the experiment changes the surrogate control-to-position derivative. Local fixed-input checks verify `dp_next/du = dt/2 * dv_next/du`, including after measurement alignment. The original update has zero direct same-substep position/control derivative.

Validation before GPU admission: 617 CPU tests on the persistent experiment branch and 505 tests on the depth-only main implementation, each with 531 subtests, passed (one skipped), including actual collision-loss escape directions at box centres, deep penetrations, corners, cylinder axes and caps, batched snapshot poses, float32/float64 control Jacobians, measurement alignment and matched factorial inputs. Historical DCOL function ASTs are unchanged. This establishes implementation properties, not training superiority. GPU qualification and 256-environment capacity evidence are recorded with the frozen-source study before the full run is admitted.

The explicit plan is `configs/experiments/zhang_collision_integration_512_20261031.json`; the shared capacity driver accepts it through `--plan`. It preserves all failures and requires identical initial model/world hashes and identical non-factor settings. Other sessions may continue; no isolated throughput gain is claimed.

Actual GPU preflight: all four N4/H150 controls completed two updates with identical initial model and world hashes. The N256/H150 SDF/trapezoid path completed two updates including the first component-gradient diagnostic, with a sampled process peak of 2178 MiB under shared load. A separate main-source default SDF/trapezoid N4 run also completed two updates with pre-reset loss and height reset. These checks do not establish policy quality or isolated throughput.


## Default selection after the common 512-update evaluation

The operator requested a default selected from all four controls. On the identical held-out fS1/40 task (evaluation seed and terrain seed 11000, 256 first naturally completed episodes per checkpoint), results were:

| Control | Success | Collision | Timeout | Bounds | Invalid |
|---|---:|---:|---:|---:|---:|
| DCOL/original | 173/256 | 67/256 | 13/256 | 3/256 | 0/256 |
| SDF/original | 126/256 | 123/256 | 7/256 | 0/256 | 0/256 |
| DCOL/trapezoid | 163/256 | 89/256 | 4/256 | 0/256 | 0/256 |
| SDF/trapezoid | 186/256 | 63/256 | 7/256 | 0/256 | 0/256 |

New BPTT/full-window Zhang configurations therefore default to **SDF + trapezoid surrogate position integration**, retaining pre-reset loss and height-bound reset. Original integration remains explicit and old checkpoint configurations retain their stored integration mode. Uncompared SHAC/AHAC and other velocity-loss modes retain original integration; older checkpoints without an integration field are interpreted as original. Physical Newton integration does not change. Existing frozen experiments and their 1024-update extensions are untouched.

Training evidence was compared at the same updates 301-400. SDF/trapezoid had raw mean return 2512.65 but reward/step 4.3798, versus SDF/original 2507.83 and 4.4349: the longer episode duration means raw return alone would mislead. SDF/trapezoid had lower acceleration/jerk costs (3.2662/19.9348 versus 3.5231/21.2976), but higher velocity loss (1.2710 versus 1.1598) and more training timeouts (6.84% versus 3.33%). DCOL/trapezoid had the best training success and lowest total loss yet did not win the held-out comparison. Collision loss definitions differ between DCOL and SDF and are not directly calibrated safety scores.

This is a provisional operational default based on one training seed and one fixed evaluation scene seed, not multiseed convergence or final 1024-update retention acceptance. Evidence is preserved under the persistent tuning worktree's `logs/2026-10-31_collision_integration_256/default_selection/ANALYSIS.md`, `analysis.json`, and the four `*_evaluation.json` files; training source commit is `464e56e832fa61af2abd0a3ac8b596f6041e3f24`.
