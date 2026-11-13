# Navigation architecture

Training, evaluation and diagnostics construct the same Newton MJWarp environment
through the runtime module in `physics.py`. The registered NavRL and Zhang classes
in `tasks/navigation/navrl_env.py` use the Warp loop in `warp_loop.py`.
`presets=pointnet` selects the point-cloud observation and matching BPTT recipe
through Isaac Lab's native task configuration resolver.

## Runtime and scene

The runtime module owns argument parsing, configuration composition, environment
construction and actual-runtime metadata. Its interface is shared by entry points
and tests; callers do not select another physics engine or environment frontend.
This gives runtime checks locality and each caller leverage from the same
implementation.

The scene contains one shared static terrain and shared dynamic obstacle geometry.
Dynamic transforms come from the obstacle-motion tensors. Cameras consume the
live shared transforms; LiDAR uses its configured mesh set. Clearance queries
combine static terrain with the obstacle-motion snapshot. Resetting one drone
leaves the obstacle world and other drones unchanged.

Warp buffers and CUDA graphs cover fixed-shape control and task calculations.
Simulator writes, sensor updates, random sampling, delay histories and partial
resets remain outside capture. These internal seams preserve sensor timing and
buffer addresses. The Hummingbird mass, inertia, rotor allocation, Lee control law
and RotorGroup motor response define physical execution.

## State has one owner

`NavigationTask` owns targets, target speeds and directions, height bounds, arrival
and failure bookkeeping, episode IDs and reward history. Partial resets mutate
only the selected rows of its tensors.

`ObservationContext` separates policy features from privileged simulation state
and training diagnostics. Both namespaces are read-only maps. Policy perturbations
are sampled once per policy tick. Zhang's known-map smooth-goal guidance keeps its
existing geometry input alongside the measured policy features.

```python
observation = env.get_zhang_observation()
depth = observation.policy["depth_features"]
measured_velocity = observation.policy["velocity"]
true_velocity = observation.privileged["velocity"]
```

NavRL's Gym observation is `{"policy": tensor}`. PointNet obtains body-frame XYZ
and validity through the same observation module.

`TransitionBatch` owns a detached snapshot of the physical transition before
autoreset. The snapshot retains the old episode ID, target, terminal state,
executed action, substep events and optional critic inputs. The next observation
may already belong to a new episode.

Register a one-shot pre-reset consumer with
`env.transitions.set_consumer(callback)`. The callback receives `TransitionBatch`
and is cleared before invocation. After `env.step()`, `extras["transition"]`
contains the completed transition and reward bookkeeping. Consumers keep snapshot
tensors immutable.

## Clearance and differentiable training

`env.query_clearance(points, body_radius=0.3, query_time=0.0)` returns signed
clearance in metres. It subtracts the supplied body radius once. Query times are
offsets from the supplied obstacle snapshot; moving obstacles are extrapolated
with their velocity. Already time-aligned substep snapshots use zero offsets.

Zhang uses the cuRoboV2 geometry adapter by default; NavRL uses the native geometry
adapter. The geometry interface is the test surface for distance, timing and
gradient semantics. Static native terrain queries retain their signed-heightfield
and verified-exit proxy definition.

Newton advances the physical state. Zhang and PointNet construct a separate
differentiable point-mass model for the training gradient. The trainer owns that
graph and its loss; the environment owns the executed transition.

NavRL uses RSL-RL PPO. Zhang supports BPTT, SHAC and AHAC. SRU-LSTM-Gated is limited
to Zhang BPTT: it has hidden and memory states, whereas the SHAC/AHAC critic reads
the GRU hidden state. Selecting SRU with SHAC or AHAC raises before simulation.

## Policy timing and perturbations

Zhang and PointNet use 20 Hz physics and policy updates by default. NavRL uses
62.5 Hz. `PolicyPerturbation` advances Isaac Lab delay buffers once per policy step.
Partial resets clear the selected drones' delay histories.

Zhang receives delayed depth with its capture pose and timestamp. Ego-motion
reprojection transforms that measurement to the current camera frame. Uncovered
pixels remain unknown. Current pose and velocity estimates include configured
noise. `Observation/depth_age_s` records the received measurement's age;
`Observation/depth_known_fraction` records its reprojected coverage.

The differentiable action history selects the same delay as physical execution.
`executed_action_max_error` measures their agreement. Extra randomization is enabled
by the fine-tuning stages described in the [curriculum](TRAINING_CURRICULUM.md).
