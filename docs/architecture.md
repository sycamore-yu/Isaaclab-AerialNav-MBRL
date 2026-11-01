# Baseline architecture

DiffNav consumes a 1 × 12 × 16 depth feature map and a 10-dimensional state vector. A CNN and a 192-dimensional GRU produce six outputs: acceleration and auxiliary velocity prediction. Policy inputs contain neither a global time field nor a planner command.

Newton + Warp supplies forward trajectories. A differentiable point-mass surrogate supplies training derivatives. Its trapezoidal position update creates a direct action-to-position derivative; the Newton solver itself is not differentiated.

SDF denotes the signed distance consumed by avoidance and collision losses. cuRoboV2 computes those distances from a shared known-occupancy ESDF and all active dynamic shapes. These are different layers, not duplicate avoidance objectives. The original soft avoidance and collision penalties remain distinct.

`variant=terminal` names the first-collision-substep terminal sampling protocol. Mathematical DCOL is not used by the public trainer. Historical functions remain under `aerialnav.reference`; the previous `aerialnav.diffnav.dcol` import is a compatibility re-export. Old inference weights with the former sampling label remain readable.

The public task IDs are `Isaac-AerialNav-Depth-v0` and `Isaac-AerialNav-PPO-v0`. The latter retains the optional legacy static PPO entrypoint, with a different observation/action contract; it is not a matched dynamic-task control. The selected baseline is depth/CNN/GRU/BPTT; alternate actor-critic, recurrent, and gradient options are not selected defaults.

The earlier static-only draft and `weights/best_static_s0.pt` are legacy artifacts. Its historical training metrics do not describe the new rectangular task or the final-state success protocol.
