# PointNet-GRU point-cloud navigation

Train the static S0 recipe with the existing Isaac Lab Python environment:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_diffnav.py presets=pointnet \
  --num_envs 128 --log_dir logs/pointnet_s0 \
  physics=newton_mjwarp --frontend warp
```

Use a new log directory for every launch. `--resume_checkpoint PATH` restores
model, optimizer, normalizer, scheduler, RNG and cumulative counters; environment,
recurrent state and the partial report window restart. An explicitly requested
environment-count change is supported. Frozen historical sources and results are
not rewritten by naming cleanup.

The package is `aerialnav.pointnet`; inference is
`aerialnav.pointnet.inference.load_pointnet_policy`. `presets=pointnet` composes the
existing navigation environment and BPTT recipe through Isaac Lab's native resolver.
`train_pointnet.py` is a convenience alias for the same shared entry point.
No separate point-cloud Gym task is registered. See the [integration](pointnet-integration.md).

## Sensor, policy and execution contract

S0 uses 240 static obstacles, no dynamic obstacles or domain randomization,
a 120 x 40 m map, target speeds 1–6 m/s, and five-second arrival-wait/final-goal
outcomes. Native reporting requires at least 1024 naturally completed episodes
per report and at least 80% success to finish the stage. It does not perform an
additional evaluation rollout or advance to S1 automatically.

The body-mounted LiDAR uses `ray_alignment="base"`, 5400 rays, 360-degree horizontal
and 60-degree vertical coverage at 2-degree resolution, and a 20 m range.
World hits become body XYZ through `(hit_world - position) @ R_body_to_world`;
missing returns are zero XYZ with validity 0. Navigation's separate yaw-only
LiDAR is unchanged. Its yaw correction is applied only in yaw mode.

PointNet uses 3→64→128→1024 pointwise features, LeakyReLU slope 0.05, no T-Net or
BatchNorm, global maximum pooling, projection to 192 features, and GRU192.
The actor consumes XYZ and ten proprioceptive values and produces three body-frame
acceleration outputs. It does not consume camera images or motion-detector labels
in this recipe. The existing depth sensor remains available for base diagnostics.

Training uses step BPTT, H160, 20 Hz, and FP32 dynamics, losses, projection, GRU and
actions. Valid-point packing preserves every return, without a fixed validity quota.
One count vector is copied to the host per scan batch. Encoder-only checkpointing,
64-env microbatch ceiling, BF16 pointwise affine operations and `torch.compile`
are the selected execution settings. No CPU offload is used.
`--point_microbatch_envs`, `--point_precision` and `--point_encoder_backend` retain
their existing choices; Triton remains an experimental benchmark candidate.

## Objective and parameter provenance

Source: Liu et al., *Learning to Fly from Point Clouds via Differentiable Simulation*, equations 5–7.

- Velocity: `0.8 Huber(||v_ref|| - ||v_avg||) + 0.6 sum_i Huber(v_ref_i - v_avg_i)`.
- Collision: nonnegative detached closing speed multiplies the entire quadratic
  clearance and smooth softplus distance penalty. Receding clearance contributes zero.
- Acceleration: squared L2 norm, `sum_i a_i**2`.
- Jerk: per-environment/per-episode mean/variance regularization with reset masking.
- Height: the existing project's additional goal-height-band constraint.

Only velocity component weights 0.8 and 0.6 are attributed to the paper. Outer
weights (velocity 1, quadratic avoidance 3, smooth distance 7.5, acceleration 0.01,
jerk 0.001, height 1), Huber delta 1, softplus slope 32, jerk mean/variance weights
1/1 and causal velocity window 2 s are project parameters, not published original
hyperparameters. AdamW uses lr 1e-4 and weight decay 0.01.

New checkpoints use `method="pointnet"` and
`observation_contract="body_xyz_lidar_validity_v2"`. The loader accepts only the verified corrected-sensor historical
source receipt and observation contract, without modifying old files. Model state-dict keys and tensor shapes
are unchanged. Historical checkpoints before the full-attitude sensor correction
must still be interpreted with their original frozen sensing configuration.
Native TensorBoard tag names and completed-episode meanings are unchanged.

## Record the actual policy and LiDAR hits

Install the optional recording extra (`uv sync --extra recording`) and use an
original policy checkpoint. Recording does not train or update it:

```bash
PYTHONPATH=src .venv/bin/python scripts/record_pointcloud.py \
  --checkpoint PATH_TO_CHECKPOINT --output logs/recordings/pointnet-static \
  --seed 0 --terrain_seed 0 presets=pointnet \
  physics=newton_mjwarp --frontend warp
.venv/bin/rerun logs/recordings/pointnet-static/navigation.rrd
```

The output directory must be new. The RRD contains the actual queried static mesh,
vehicle pose, flown trajectory, world hits and body XYZ hits. Only finite measured
hits are shown; miss-ray endpoints are omitted. `trace.npz` retains those same hits,
poses and validity offsets; `result.json` records the checkpoint, scene and outcome.
The default main palette separates dark ground (58, 73, 86) from light obstacles
(215, 225, 235), using the original 0.05 m raised-triangle threshold without
changing geometry. The follow view tracks translation only; the overview tab
retains the full trajectory. The follow and body-frame views share simulation time. The trajectory
stops before the automatic terminal reset, rather than drawing a reset teleport.
The SDK is pinned to 0.30.0 to avoid newer SDKs' psutil conflict with Isaac Sim.

For ordinary playback/evaluation, use the same `presets=pointnet` selector with
`scripts/evaluate.py --method diffnav --checkpoint PATH --output RESULT.json`.

Latest [S1/F1 checkpoint tests and default-style recording](results/pointnet-s1-f1-transfer-20261104.md).
