# One navigation environment, a PointNet preset

`presets=pointnet` now selects the body-mounted LiDAR observation and the matching
PointNet–GRU192 BPTT recipe on the existing navigation task. It does not register a
second Gym task or copy physics, control, reset, curriculum, geometry or reporting.
Without the selector, depth training and PPO keep their existing defaults.

This uses Isaac Lab 3.0's native `PresetCfg`, `preset()` and `resolve_task_config()`:
[Backends and Presets](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/backends_and_presets.html),
[Hydra composition](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/features/hydra.html).
The environment's perception node and the sibling custom BPTT recipe both expose
`pointnet`. Each selected node replaces a complete configuration section. The
launcher composes both roots before creating the existing Torch/Warp environment;
explicit command-line training options override recipe defaults.

## Observation and recipe are separate

The perception node selects full-attitude `ray_alignment="base"`, 5400 rays, 20 m
range, packed body XYZ plus validity, ten proprioceptive features and three body
acceleration outputs. The mask excludes misses from pooling; it is not a learned
fourth channel. Navigation's original yaw-aligned diagnostic LiDAR is unchanged.

`PointNetTrainCfg` selects step BPTT, H160, the point-cloud objective, optimizer and
execution defaults. It does not modify PPO rewards or silently select another
algorithm. This custom BPTT trainer is not an RSL-RL PPO policy implementation.
The base Gym `policy` diagnostic vector remains unchanged; the BPTT accessor adds
`policy_points` with shape `[N, 4, 5400]`.

## Checkpoint meaning is checked on load

New checkpoints and run configuration record the selected preset, observation
contract, learned channels, state/output/action dimensions and actual sensor
pattern, alignment, range, period and mounting transform. A mismatch raises before
loading weights, including continuation, inference and stable recovery.

The historical corrected-sensor source `f8d42aa` is accepted only with its original
source receipt and observation version. Earlier yaw-sensor checkpoints are rejected.
Weights, GRU tensor shapes, historical paths and training event files are not rewritten.

## Scoped integration

The release and research-main histories and package layouts differ. Only the
PointNet modules, shared extension points, entry-point composition, sensor fix,
recording, tests and documentation are ported. Their own scene settings, baseline
recipes and unrelated changes remain on their respective branches. No whole-feature
branch merge is used. Commands and contracts are in [pointnet.md](pointnet.md);
actual acceptance results are in [the integration report](results/pointnet-preset-20261104.md).
