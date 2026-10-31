# cuRoboV2 geometry comparison

The operator requested the same budget concept as C12: **256 optimizer updates**, not 256 completed episodes. Two N256/H150 depth CNN+GRU BPTT arms start from the same frozen SDF/trapezoid update-512 checkpoint. Model, AdamW, normalization, RNG, cosine phase, LR caps/scales and recovery state are restored. The input identity is under `.optim-agent-runs/curobo-v2-geometry-256-20261031/inputs`. The plan is `configs/experiments/curobo_v2_geometry_256_20261031.json`.

Only `geometry_backend` differs. Both use pre-reset loss, height-bound reset, SDF collision loss, trapezoid surrogate integration, the original weights/ranges and Newton+Warp. The native evaluator remains the common judge for both final checkpoints (256 first natural episodes, seed/terrain 11000, fS1/40). Training naturally completed episodes and failures remain separate from evaluation. The original four-cell study and its 1024-update extension continue unchanged.

## Actual integration

`reference/curobo_geometry.py` imports optional cuRoboV2 only when selected. The official reference checkout is pinned to `78fd485fa82d9b9a063fb4985e371814587e666a`. Additional CUDA-core packages live in that checkout's `.navrl-deps`; the shared simulation environment is not modified.

- Static map: preserve the existing known height-field occupancy, including the ground plane, seed voxel surface sites on GPU, and use cuRoboV2's exact PBA distance transform. Signed voxel distances are stored as FP16 and queried using cuRobo's trilinear value/gradient sampler. Resolution is 0.0625 m. The original occupancy proxy is not called to compute query distances. It supplies only the map occupancy boundary at construction.
- Dynamic world: use cuRobo cuboid SDF helpers and watertight 128-sided cylinder meshes, traversing every active obstacle. Maximum radial cylinder approximation error is below 0.16 mm for the task's radii. Poses at each lookahead/collision sample are immutable query snapshots.
- Autograd: cuRobo's public `get_sphere_distance_raw` returns an activated collision cost, not the signed meter distance needed by Zhang losses. The adapter uses the underlying raw helpers, reverses their collision-cost gradient convention for cuboids/meshes, and retains the voxel sampler's actual unnormalized interpolation derivative. At interior medial-axis plateaus a documented one-sided escape branch prevents a zero recovery gradient. Every call owns its backward gradient snapshot; mutable scene buffers are never retained for later backward.
- Scope: this backend replaces static/dynamic distance queries for avoidance, collision loss and geometric collision events. Policy sensors, scene meshes, reward equations, physical Newton contacts, rotor dynamics and action delay are unchanged.

All drones in this task share one static map and one dynamic obstacle process. Thus one map plus batched queries serves 256 drones; there is no need to allocate 256 maps. The library also supports independent collision environments through `SceneCollisionCfg.num_envs` and `env_query_idx`; an independent ESDF per environment would need separate memory/cost accounting.

The cuRobo `Mapper` TSDF/depth/LiDAR fusion API is deliberately not enabled. Replacing known scene occupancy with partial sensor observations changes available information, unknown-space handling and map lifetime, introducing an additional research factor. Likewise cuRobo's swept-sphere checker and trajectory optimization/MPC can be separate future controls. Its manipulator inverse-dynamics/IK stack does not directly replace the drone's rotor/aerodynamic dynamics or recurrent policy.

## Validation and limits

Real GPU geometry tests cover signed distance/finite differences, box centre/corners/deep penetration, cylinder axis/caps, mutable pose snapshots, full traversal beyond the nearest five centres, and 256 simultaneous queries. The normal CPU suite runs without cuRobo installed. Existing static occupancy is a conservative height-field definition, not a general watertight mesh volume; ESDF voxel resolution changes its boundary approximation. This is an explicit geometry experiment and not an exact-geometry or policy acceptance claim.

Official sources: [cuRoboV2](https://github.com/NVlabs/curobo), [volumetric mapping](https://nvlabs.github.io/curobo/latest/api/curobo.examples.getting_started.volumetric_mapping.html), and the pinned source's `curobo/_src/geom/collision/collision_scene.py` / `curobo/_src/perception/mapper/esdf/edt_parallel_banding.py`.

## Promotion to main and new speed range

On 2026-10-31 the user selected cuRoboV2 as the Zhang default. The matched study finished both arms at cumulative update 768 (256 added updates), from checkpoint SHA256 `8920d7c7c5b9a8cec980ce05036da89f8c0842e97c089fbb7b51cbc54bebb50e`, source `1e299bbdab421928e99f628749d01ab874fe385b`. Both arms used the common native evaluator at seed/terrain 11000:

| Backend | Success | Collision | Timeout | Bounds / invalid |
|---|---:|---:|---:|---:|
| native | 187/256 (73.05%) | 66/256 | 3/256 | 0 / 0 |
| cuRoboV2 | 191/256 (74.61%) | 64/256 | 1/256 | 0 / 0 |

This is a single training seed and one evaluation scene: four additional successes (1.56 percentage points), not multi-seed confirmation or curriculum acceptance. Full evidence is in `/home/tong/tongworkspace/free/.worktrees/isaaclab-navrl/optim-loop-fs1-20261031/logs/2026-10-31_curobo_v2_geometry_256/{comparison.json,REPORT.md}`.

New Zhang tasks and training default to `geometry_backend=curobo_v2`. `--geometry_backend native` retains the native control. Historical DCOL reference and the named six-cell, scene-control, SRU and collision/integration comparison plans keep explicit native geometry. Checkpoints without a recorded backend are interpreted as native. Existing frozen runs and operator-paused runs are untouched.

Separately, the user requested the target speed distribution to change from Uniform(2.25,9.75) to **Uniform(1,6) m/s**, independently resampled for each resetting drone on every episode. This changes the task distribution; the comparison above was run under the old speed range and is not evidence for a gain under the new range. Explicit `--target_speed_min` / `--target_speed_max` overrides remain supported. Run configs, evaluations and new checkpoints record the range.

Training starts and goals are sampled independently on the square boundary: each side has probability 1/4, the free coordinate is Uniform(-24,24), and default heights are Uniform(0.5,2.5) m. Same-side pairs and short routes are possible; there is no interior-area uniform sampler or opposite-side constraint. Evaluation without `--stage` uses the existing fixed opposite-side layout instead.

The pinned cuRobo checkout and its `.navrl-deps` directory are available to the current shared Python environment through `.venv/lib/python3.12/site-packages/navrl_curobo_v2.pth`. This reuses the existing packages without replacing Torch, Warp or NumPy. Fresh installations need the pinned cuRobo checkout plus its CUDA-core dependencies on the Python import path. The default requires CUDA and does not silently fall back to native.

Validation artifacts: `logs/2026-10-31_main_curobo_speed_1_6_smoke/` contains the two-update N16/H150 Newton+Warp smoke, configuration, geometry metadata and loadable checkpoint. This is startup/backpropagation validation, not policy performance evidence. The first direct launch omitted the standard EULA environment variable and exited before environment creation; its log remains `/tmp/navrl-main-curobo-speed-smoke.log`. The successful launch log is `/tmp/navrl-main-curobo-speed-smoke-retry.log`.

Final main validation: full regression **515 passed, 1 skipped, 531 subtests passed**; the five real cuRobo GPU tests are included. Changed Python files pass Ruff lint/format checks, and `git diff --check` passes. N16/H150 smoke completed two updates with gradient norms 2.1823 and 3.7108; final checkpoint was load-verified.
