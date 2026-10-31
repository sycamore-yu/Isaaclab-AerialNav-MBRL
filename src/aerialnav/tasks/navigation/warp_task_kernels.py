"""Warp algebra for the existing Navigation observations, rewards, and event flags.

The caller owns sensor updates, geometry queries, nearest-neighbour selection,
randomization, and episode accounting. Inputs and outputs keep stable storage;
the launch methods contain only Warp kernel launches and are graph capturable.
"""

from __future__ import annotations

import torch
import warp as wp


@wp.func
def _at_least(value: float, lower: float) -> float:
    # Comparisons preserve NaN, matching Torch clamp instead of masking it.
    if value < lower:
        return lower
    return value


@wp.func
def _at_most(value: float, upper: float) -> float:
    if value > upper:
        return upper
    return value


@wp.func
def _goal_frame(vector: wp.vec3, direction: wp.vec3, inverse: bool) -> wp.vec3:
    horizontal_norm = _at_least(wp.sqrt(direction[0] * direction[0] + direction[1] * direction[1]), 1.0e-6)
    c = direction[0] / horizontal_norm
    s = direction[1] / horizontal_norm
    if inverse:
        return wp.vec3(c * vector[0] - s * vector[1], s * vector[0] + c * vector[1], vector[2])
    return wp.vec3(c * vector[0] + s * vector[1], -s * vector[0] + c * vector[1], vector[2])


@wp.kernel(enable_backward=False)
def goal_frame_kernel(
    vector: wp.array2d(dtype=wp.float32),
    direction: wp.array2d(dtype=wp.float32),
    inverse: bool,
    out: wp.array2d(dtype=wp.float32),
):
    """Transform one vector per environment, including the inverse action map."""
    env = wp.tid()
    value = _goal_frame(
        wp.vec3(vector[env, 0], vector[env, 1], vector[env, 2]),
        wp.vec3(direction[env, 0], direction[env, 1], direction[env, 2]),
        inverse,
    )
    for axis in range(3):
        out[env, axis] = value[axis]


@wp.kernel(enable_backward=False)
def navigation_terms_kernel(
    position: wp.array2d(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32),
    target: wp.array2d(dtype=wp.float32),
    direction: wp.array2d(dtype=wp.float32),
    lidar_distance: wp.array2d(dtype=wp.float32),
    obstacle_position: wp.array2d(dtype=wp.float32),
    obstacle_velocity: wp.array2d(dtype=wp.float32),
    obstacle_size: wp.array2d(dtype=wp.float32),
    selected_indices: wp.array2d(dtype=wp.int64),
    selected_distance_xy: wp.array2d(dtype=wp.float32),
    previous_velocity: wp.array2d(dtype=wp.float32),
    height_range: wp.array2d(dtype=wp.float32),
    selected_count: int,
    policy: wp.array2d(dtype=wp.float32),
    reward_terms: wp.array2d(dtype=wp.float32),
    reached: wp.array(dtype=wp.bool),
    bounds: wp.array(dtype=wp.bool),
):
    """Fuse task_logic.navigation_terms without changing its neighbour ordering."""
    env = wp.tid()
    p = wp.vec3(position[env, 0], position[env, 1], position[env, 2])
    v = wp.vec3(velocity[env, 0], velocity[env, 1], velocity[env, 2])
    g = wp.vec3(target[env, 0], target[env, 1], target[env, 2])
    d = wp.vec3(direction[env, 0], direction[env, 1], direction[env, 2])
    previous = wp.vec3(previous_velocity[env, 0], previous_velocity[env, 1], previous_velocity[env, 2])
    relative = g - p
    distance = wp.length(relative)
    normed_relative = relative / _at_least(distance, 1.0e-6)
    goal_direction = _goal_frame(normed_relative, d, False)
    goal_velocity = _goal_frame(v, d, False)
    for axis in range(3):
        policy[env, axis] = goal_direction[axis]
        policy[env, 5 + axis] = goal_velocity[axis]
    policy[env, 3] = wp.sqrt(relative[0] * relative[0] + relative[1] * relative[1])
    policy[env, 4] = relative[2]

    safety_static = float(0.0)  # noqa: UP018 - mutable Warp scalar in a dynamic loop
    for ray in range(144):
        ray_distance = _at_most(lidar_distance[env, ray], 4.0)
        policy[env, 8 + ray] = 4.0 - ray_distance
        safety_static += wp.log(_at_least(ray_distance, 1.0e-6))
    safety_static /= 144.0

    # Unused neighbours stay zero, exactly as the original padding operation.
    for feature in range(50):
        policy[env, 152 + feature] = 0.0
    safety_dynamic = float(0.0)  # noqa: UP018 - mutable Warp scalar in a dynamic loop
    for neighbour in range(selected_count):
        obstacle = int(selected_indices[env, neighbour])
        delta = wp.vec3(
            obstacle_position[obstacle, 0] - p[0],
            obstacle_position[obstacle, 1] - p[1],
            obstacle_position[obstacle, 2] - p[2],
        )
        width = obstacle_size[obstacle, 0]
        height = obstacle_size[obstacle, 2]
        if height > 1.0:
            delta[2] = 0.0
        norm = wp.length(delta)
        outside = selected_distance_xy[env, neighbour] > 4.0
        relative_goal = _goal_frame(delta, d, False)
        neighbour_velocity = wp.vec3(
            obstacle_velocity[obstacle, 0],
            obstacle_velocity[obstacle, 1],
            obstacle_velocity[obstacle, 2],
        )
        width_class = width / 0.25 - 1.0
        height_class = height
        if height > 1.0:
            height_class = 0.0
        surface_distance = norm - width / 2.0
        if outside:
            relative_goal = wp.vec3(0.0, 0.0, 0.0)
            neighbour_velocity = wp.vec3(0.0, 0.0, 0.0)
            width_class = 0.0
            height_class = 0.0
            surface_distance = 4.0
        velocity_goal = _goal_frame(neighbour_velocity, d, False)
        base = 152 + neighbour * 10
        for axis in range(3):
            policy[env, base + axis] = relative_goal[axis] / _at_least(norm, 1.0e-6)
            policy[env, base + 5 + axis] = velocity_goal[axis]
        policy[env, base + 3] = wp.sqrt(relative_goal[0] * relative_goal[0] + relative_goal[1] * relative_goal[1])
        policy[env, base + 4] = relative_goal[2]
        policy[env, base + 8] = width_class
        policy[env, base + 9] = height_class
        safety_dynamic += wp.log(_at_most(_at_least(surface_distance, 1.0e-6), 4.0))
    if selected_count > 0:
        safety_dynamic /= float(selected_count)

    # Keep the multiply-then-divide order of the original reward expression.
    velocity_reward = float(0.0)  # noqa: UP018 - mutable Warp scalar in a dynamic loop
    for axis in range(3):
        velocity_reward += v[axis] * relative[axis] / _at_least(distance, 1.0e-6)
    smooth = wp.length(v - previous)
    below = _at_least(height_range[env, 0] - 0.2 - p[2], 0.0)
    above = _at_least(p[2] - height_range[env, 1] - 0.2, 0.0)
    height_penalty = below * below + above * above
    reward_terms[env, 0] = velocity_reward + 1.0 + safety_static + safety_dynamic - 0.1 * smooth - 8.0 * height_penalty
    reward_terms[env, 1] = velocity_reward
    reward_terms[env, 2] = safety_static
    reward_terms[env, 3] = safety_dynamic
    reward_terms[env, 4] = smooth
    reward_terms[env, 5] = height_penalty
    reached[env] = distance < 0.5
    bounds[env] = p[2] < 0.2 or p[2] > 4.0


@wp.kernel(enable_backward=False)
def first_events_kernel(
    root_state: wp.array2d(dtype=wp.float32),
    target: wp.array2d(dtype=wp.float32),
    collision_gap: wp.array(dtype=wp.float32),
    collision_terminates: bool,
    bounds_terminates: bool,
    reached: wp.array(dtype=wp.bool),
    bounds: wp.array(dtype=wp.bool),
    collision: wp.array(dtype=wp.bool),
    invalid: wp.array(dtype=wp.bool),
    terminated: wp.array(dtype=wp.bool),
):
    """Evaluate the existing sphere predicates using caller-provided clearance.

    ``collision_gap`` already includes the task's collision radius. No contact
    or terrain geometry is recomputed here. Invalid checks all 13 root entries,
    matching the original predicate even for a non-finite quaternion or speed.
    """
    env = wp.tid()
    relative = wp.vec3(
        target[env, 0] - root_state[env, 0],
        target[env, 1] - root_state[env, 1],
        target[env, 2] - root_state[env, 2],
    )
    hit = collision_gap[env] <= 0.0
    outside = root_state[env, 2] < 0.2 or root_state[env, 2] > 4.0
    bad_state = bool(False)  # noqa: UP018 - mutable Warp scalar in a dynamic loop
    for component in range(13):
        if not wp.isfinite(root_state[env, component]):
            bad_state = True
    reached[env] = wp.length(relative) < 0.5
    bounds[env] = outside
    collision[env] = hit
    invalid[env] = bad_state
    terminated[env] = bad_state or (bounds_terminates and outside) or (collision_terminates and hit)


@wp.kernel(enable_backward=False)
def termination_mask_kernel(
    collision: wp.array(dtype=wp.bool),
    bounds: wp.array(dtype=wp.bool),
    invalid: wp.array(dtype=wp.bool),
    collision_terminates: bool,
    bounds_terminates: bool,
    terminated: wp.array(dtype=wp.bool),
):
    """Also support accumulated substep flags without recomputing predicates."""
    env = wp.tid()
    terminated[env] = invalid[env] or (collision_terminates and collision[env]) or (bounds_terminates and bounds[env])


@wp.kernel(enable_backward=False)
def latch_first_event_kernel(
    reached: wp.array(dtype=wp.bool),
    bounds: wp.array(dtype=wp.bool),
    collision: wp.array(dtype=wp.bool),
    invalid: wp.array(dtype=wp.bool),
    elapsed_s: wp.array(dtype=wp.float32),
    event_code: wp.array(dtype=wp.int8),
    event_time: wp.array(dtype=wp.float32),
):
    """Latch once with priority invalid > collision > bounds > reached."""
    env = wp.tid()
    code = int(0)  # noqa: UP018 - mutable Warp scalar across branches
    if reached[env]:
        code = 1
    if bounds[env]:
        code = 3
    if collision[env]:
        code = 2
    if invalid[env]:
        code = 4
    if event_code[env] == wp.int8(0) and code != 0:
        event_code[env] = wp.int8(code)
        event_time[env] = elapsed_s[env]


@wp.kernel(enable_backward=False)
def reset_first_event_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    event_code: wp.array(dtype=wp.int8),
    event_time: wp.array(dtype=wp.float32),
):
    """Clear only selected episodes using a fixed-size reset mask."""
    env = wp.tid()
    if reset_mask[env]:
        event_code[env] = wp.int8(0)
        event_time[env] = 0.0


def _check_tensor(name: str, value: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, device) -> None:
    if value.shape != shape or value.dtype != dtype or value.device != device:
        raise ValueError(
            f"{name} must have shape {shape}, dtype {dtype}, and device {device}; "
            f"got {value.shape}, {value.dtype}, {value.device}"
        )


class WarpNavigationTerms:
    """Bind fixed inputs and allocate the original Navigation term dictionary once.

    ``selected_indices`` and ``selected_distance_xy`` are the original Torch
    ``distances_xy.topk(min(5, obstacle_count), largest=False)`` outputs. The
    caller must copy new values into the same buffers before each launch. With
    no dynamic obstacles, both arrays have shape ``(num_envs, 0)``.

    ``launch()`` returns persistent Torch views; do not replace dictionary entries.
    Copy values at the existing transition snapshot boundary before the next
    launch, and use a shallow dictionary copy when combining additional terms.
    Noise and delayed policy delivery remain the caller's responsibility.
    """

    def __init__(
        self,
        *,
        position: torch.Tensor,
        velocity: torch.Tensor,
        target: torch.Tensor,
        direction: torch.Tensor,
        lidar_distance: torch.Tensor,
        obstacle_position: torch.Tensor,
        obstacle_velocity: torch.Tensor,
        obstacle_size: torch.Tensor,
        selected_indices: torch.Tensor,
        selected_distance_xy: torch.Tensor,
        previous_velocity: torch.Tensor,
        height_range: torch.Tensor,
    ):
        self.num_envs, self.device = len(position), position.device
        obstacle_count = len(obstacle_position)
        self.selected_count = min(5, obstacle_count)
        inputs = {
            "position": position,
            "velocity": velocity,
            "target": target,
            "direction": direction,
            "lidar_distance": lidar_distance,
            "obstacle_position": obstacle_position,
            "obstacle_velocity": obstacle_velocity,
            "obstacle_size": obstacle_size,
            "selected_indices": selected_indices,
            "selected_distance_xy": selected_distance_xy,
            "previous_velocity": previous_velocity,
            "height_range": height_range,
        }
        shapes = {
            "position": (self.num_envs, 3),
            "velocity": (self.num_envs, 3),
            "target": (self.num_envs, 3),
            "direction": (self.num_envs, 3),
            "lidar_distance": (self.num_envs, 144),
            "obstacle_position": (obstacle_count, 3),
            "obstacle_velocity": (obstacle_count, 3),
            "obstacle_size": (obstacle_count, 3),
            "selected_indices": (self.num_envs, self.selected_count),
            "selected_distance_xy": (self.num_envs, self.selected_count),
            "previous_velocity": (self.num_envs, 3),
            "height_range": (self.num_envs, 2),
        }
        for name, value in inputs.items():
            _check_tensor(
                name, value, shapes[name], torch.int64 if name == "selected_indices" else torch.float32, self.device
            )
        # Retain the owners; wp.from_torch creates views, never private copies.
        self._torch_inputs = inputs
        self.policy = torch.empty((self.num_envs, 202), dtype=torch.float32, device=self.device)
        self.policy_tensor = self.policy
        self.reward_terms = torch.empty((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self.reached = torch.empty(self.num_envs, dtype=torch.bool, device=self.device)
        self.bounds = torch.empty_like(self.reached)
        self.terms = {
            "state": self.policy[:, :8],
            "lidar": self.policy[:, 8:152].view(self.num_envs, 1, 36, 4),
            "dynamic_obstacle": self.policy[:, 152:].view(self.num_envs, 1, 5, 10),
            "reached": self.reached,
            "bounds": self.bounds,
        }
        for column, name in enumerate(
            ("reward", "reward_velocity", "reward_static", "reward_dynamic", "penalty_smooth", "penalty_height")
        ):
            self.terms[name] = self.reward_terms[:, column]
        self._inputs = [wp.from_torch(value) for value in inputs.values()] + [self.selected_count]
        self._outputs = [wp.from_torch(value) for value in (self.policy, self.reward_terms, self.reached, self.bounds)]
        self._device = wp.get_device(str(self.device))

    @torch.no_grad()
    def update(self, **inputs: torch.Tensor) -> None:
        """Copy changed inputs in place; call only outside the captured region.

        Omit inputs already updated through their bound simulation views. A new
        obstacle count requires a new binding and a new graph, never pointer
        replacement in an existing graph.
        """
        for name, value in inputs.items():
            if name not in self._torch_inputs:
                raise ValueError(f"Unknown navigation input: {name}")
            bound = self._torch_inputs[name]
            _check_tensor(name, value, tuple(bound.shape), bound.dtype, bound.device)
            bound.copy_(value)

    def launch(self) -> dict[str, torch.Tensor]:
        """Overwrite the fixed outputs using one capturable Warp launch."""
        wp.launch(
            navigation_terms_kernel,
            dim=self.num_envs,
            inputs=self._inputs,
            outputs=self._outputs,
            device=self._device,
        )
        return self.terms


class WarpFirstEvents:
    """Bind graphable event predicates and optional first-event history.

    ``launch()`` refreshes the four instantaneous flags plus ``terminated``.
    After copying accumulated substep flags into the existing output buffers,
    ``launch_termination()`` can recompute termination from those masks. History
    is only changed by ``launch_latch()``; call it at the original event boundary.
    To enable latching, pass all three existing buffers: ``event_code`` (int8),
    ``event_time`` (float32), and ``elapsed_s`` (float32). Timeout accounting
    stays with the environment. Do not replace entries in the output dictionary.

    As with ``WarpNavigationTerms``, ``update()`` is an uncaptured boundary and
    all launch methods keep the same input pointers and output references.
    """

    def __init__(
        self,
        *,
        root_state: torch.Tensor,
        target: torch.Tensor,
        collision_gap: torch.Tensor,
        collision_terminates: bool = True,
        bounds_terminates: bool = True,
        event_code: torch.Tensor | None = None,
        event_time: torch.Tensor | None = None,
        elapsed_s: torch.Tensor | None = None,
    ):
        self.num_envs, self.device = len(root_state), root_state.device
        inputs = {"root_state": root_state, "target": target, "collision_gap": collision_gap}
        for name, shape in (
            ("root_state", (self.num_envs, 13)),
            ("target", (self.num_envs, 3)),
            ("collision_gap", (self.num_envs,)),
        ):
            _check_tensor(name, inputs[name], shape, torch.float32, self.device)
        self._torch_inputs = inputs
        self.terms = {
            name: torch.empty(self.num_envs, dtype=torch.bool, device=self.device)
            for name in ("reached", "bounds", "collision", "invalid")
        }
        self.terminated = torch.empty(self.num_envs, dtype=torch.bool, device=self.device)
        self._inputs = [wp.from_torch(value) for value in inputs.values()] + [collision_terminates, bounds_terminates]
        self._outputs = [wp.from_torch(value) for value in self.terms.values()] + [wp.from_torch(self.terminated)]
        self._termination_inputs = [self._outputs[index] for index in (2, 1, 3)] + [
            collision_terminates,
            bounds_terminates,
        ]
        self._device = wp.get_device(str(self.device))
        self._latch_inputs = None
        self._latch_outputs = None
        supplied = (event_code is not None, event_time is not None, elapsed_s is not None)
        if any(supplied):
            if not all(supplied):
                raise ValueError("Latching requires event_code, event_time, and elapsed_s together")
            for name, value, dtype in (
                ("event_code", event_code, torch.int8),
                ("event_time", event_time, torch.float32),
                ("elapsed_s", elapsed_s, torch.float32),
            ):
                _check_tensor(name, value, (self.num_envs,), dtype, self.device)
                self._torch_inputs[name] = value
            self._latch_inputs = self._outputs[:4] + [wp.from_torch(elapsed_s)]
            self._latch_outputs = [wp.from_torch(event_code), wp.from_torch(event_time)]

    @torch.no_grad()
    def update(self, **inputs: torch.Tensor) -> None:
        """Copy new input values outside capture without replacing storage."""
        for name, value in inputs.items():
            if name not in self._torch_inputs:
                raise ValueError(f"Unknown event input: {name}")
            bound = self._torch_inputs[name]
            _check_tensor(name, value, tuple(bound.shape), bound.dtype, bound.device)
            bound.copy_(value)

    def launch(self) -> dict[str, torch.Tensor]:
        """Refresh event flags using one capturable launch."""
        wp.launch(
            first_events_kernel,
            dim=self.num_envs,
            inputs=self._inputs,
            outputs=self._outputs,
            device=self._device,
        )
        return self.terms

    def launch_termination(self) -> torch.Tensor:
        """Evaluate termination from the currently bound event masks."""
        wp.launch(
            termination_mask_kernel,
            dim=self.num_envs,
            inputs=self._termination_inputs,
            outputs=self._outputs[4:],
            device=self._device,
        )
        return self.terminated

    def launch_latch(self) -> None:
        """Latch current flags into the bound episode history buffers."""
        if self._latch_inputs is None:
            raise RuntimeError("First-event history buffers were not bound")
        wp.launch(
            latch_first_event_kernel,
            dim=self.num_envs,
            inputs=self._latch_inputs,
            outputs=self._latch_outputs,
            device=self._device,
        )
