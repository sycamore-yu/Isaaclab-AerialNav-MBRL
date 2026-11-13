# Navigation architecture

This refactor keeps the existing NavRL, Zhang and PointNet tasks, default
configuration, controller timing, checkpoint formats and training objectives.
Isaac Lab still owns the physics loop and same-step autoreset. The differentiable
point-mass model remains outside the physical simulator.

## Boundaries

`NavigationTask` owns targets, target speed/direction, height bounds, arrival and
failure bookkeeping, episode IDs and reward history. Resetting selected rows
does not change shared obstacle motion or another drone's task state. The
existing first-arrival and final-success rules remain different: first entry is
latched; arrival truncation requires the drone to be inside the goal after the
original 5-second wait; a recorded failure excludes final success.

`ObservationContext` separates `policy` sensor features from `privileged`
simulation state and training diagnostics. Both namespaces are read-only maps.
Observation values and actor input dimensions are unchanged. Sensor noise and
delays are still sampled once per policy tick by `PolicyPerturbation`. This is
an internal Python API change, not a checkpoint observation-layout change.
The existing smooth-goal reference still uses known-map guidance; separating
observations does not turn that guidance into a sensor-only algorithm.

`TransitionBatch` is a detached, owned snapshot of a completed physical
transition, captured before autoreset. It retains the old episode's ID, target,
terminal state, executed action, substep event data and optional critic inputs.
The next observation may already belong to a new episode and is never used as
the terminal state. A one-shot transition consumer runs before reset. It may
construct the differentiable surrogate graph from the snapshot, but the
environment does not know the loss function or retain the graph.

`query_clearance` returns signed clearance in metres after subtracting a
specified body radius exactly once. Query times are offsets from the supplied
obstacle snapshot. Dynamic obstacles are extrapolated with their supplied
velocity; already time-aligned collision snapshots use zero offsets. Static
terrain retains its existing signed-heightfield/verified-exit proxy semantics;
the wrapper does not claim to make that proxy an exact mesh signed distance.

## Scope and integration

The chosen approach extracts small state/data boundaries and reuses the current
pure tensor kernels. It does not introduce a generic component registry, replace
DirectRLEnv, merge PPO and BPTT trainers, or import ProtoMotions. A whole-framework
rewrite would risk the existing reset and gradient contracts without improving
this experiment's acceptance criteria.

Running experiments, their frozen source/configuration, checkpoints and logs
are not modified. The unrelated untracked conversion script is not part of this
change. This work is committed to local development `main`, not the public
release remote.

## Verification

Verification covers partial resets, first-event/final-outcome precedence,
arrival waiting, namespace isolation, snapshot lifetime, one-shot consumption,
dynamic query timing, body-radius subtraction and gradient parity. The existing
trainer tests cover BPTT, SHAC/AHAC endpoints, recurrent policies and checkpoint
inference. Short real-simulator checks cover Torch/Warp environment integration.
Actual commands and results are recorded in
[`results/navrl-architecture-20261113.md`](results/navrl-architecture-20261113.md).

## Internal API

Zhang and PointNet callers obtain an `ObservationContext` from
`env.get_zhang_observation()`:

```python
observation = env.get_zhang_observation()
sensor_depth = observation.policy["depth_features"]
measured_velocity = observation.policy["velocity"]
true_velocity = observation.privileged["velocity"]
```

The PPO Gym observation remains the existing `{"policy": tensor}`. Policy
network dimensions, normalization and checkpoint layouts are not changed.
PointNet's body-frame conversion and the known-map smooth-goal guidance retain
their existing information sources; this change does not introduce a new state
estimator or claim sensor-only deployment for those modes.

Register a pre-reset consumer through `env.transitions.set_consumer(callback)`.
The callback receives `TransitionBatch` and is removed before invocation. After
`env.step()`, `extras["transition"]` holds that old-episode physical data with
the completed reward bookkeeping. The caller must not modify snapshot tensors.

`env.query_clearance(points, body_radius=0.3, query_time=0.0)` queries current
geometry. To predict moving obstacles, provide offsets from the current
snapshot. The standalone function also accepts explicit snapshots for terminal
physics-substep checks. Zero radius intentionally retains point-to-surface
diagnostics; safety callers specify the drone radius or training margin.

The historical DCOL comparison retains its dimensionless scaling calculation
and metre-scaled proxy. Its time-aware trainer adapter is updated, but its
metric is not silently replaced by signed distance.
