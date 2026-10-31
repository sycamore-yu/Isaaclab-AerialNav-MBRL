# Architecture and method boundary

The primary learning method is `DiffNav`. It uses a single-channel depth feature map (1 × 12 × 16), a 10-dimensional state vector, a CNN stem, a 192-dimensional GRU state, and six network outputs. The six outputs are decoded to acceleration and auxiliary velocity prediction in a goal-aligned frame.

The publicly reproduced baseline uses a 150-step backpropagation-through-time window. Isaac Lab / PhysX produces the forward simulator trajectory. A separately implemented differentiable point-mass model supplies a surrogate state/action gradient for training. **PhysX and Newton solver internals are not themselves differentiated.**

A shared static height-field terrain provides geometry for camera sensing, collision handling and differentiable obstacle-distance losses. The released profile fixes the dynamic obstacle count to zero. The codebase retains some common geometry and sensor helpers to avoid introducing a second, inconsistent physics or perception implementation.

Two Gym task IDs are registered:

- `Isaac-AerialNav-Static-Depth-v0`: depth/BPTT research method, `DiffNavEnv`.
- `Isaac-AerialNav-Static-PPO-v0`: optional PPO velocity-control baseline, `NavigationEnv`.

The optional PPO baseline uses a different observation/action contract and should not be interpreted as an optimizer-only comparison. Train/evaluate each baseline independently.

The official public run uses PhysX/Torch. Newton/Torch and Newton/Warp frontends are research options and require new validation for cross-backend policy behavior or throughput.
