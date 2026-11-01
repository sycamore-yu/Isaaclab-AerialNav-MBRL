# Architecture

The depth policy consumes a 1 × 12 × 16 feature map and a 10-dimensional state vector. A CNN and GRU192 produce acceleration commands and an auxiliary velocity prediction. Optional SRU-LSTM-Gated uses the same observations and output head.

## Training path

1. Newton + Warp generates physical trajectories, sensor observations and episode outcomes.
2. A differentiable point-mass surrogate carries training derivatives through each rollout. Position uses trapezoidal integration.
3. cuRobo supplies signed obstacle distances and spatial gradients from the static ESDF and all active dynamic shapes.
4. BPTT optimizes the 2-second velocity window, velocity prediction, avoidance, collision, acceleration, jerk and height-band losses.

`aerialnav.diffnav` owns the policy, surrogate, losses and trainer. `aerialnav.tasks.navigation` owns the physical task, sensors, randomization and episode resets. `aerialnav.evaluation` records one outcome per initial episode.

## Runtime choices

The default is Newton/Warp with cuRobo SDF geometry. PhysX/Torch and native geometry are explicit alternatives. Task IDs are `Isaac-AerialNav-Depth-v0` for DiffNav and `Isaac-AerialNav-PPO-v0` for the optional static PPO baseline.

`variant=terminal` selects first-collision-substep sampling. Historical mathematical DCOL utilities live in `aerialnav.reference`. The inference loader accepts both packaged GRU checkpoints and selects the recurrent backend recorded in newer checkpoints.
