"""Warp execution of the original Lee controller and RotorGroup motor response.

The fused kernel consumes XYZW root states and the same per-environment motor
parameters as ``LeeRotorController``. Input and output storage stays fixed across
steps and resets. Public simulator writes remain outside CUDA graph capture.
Equations adapted from OmniDrones; see licenses/OmniDrones.txt.
"""

from __future__ import annotations

import torch
import warp as wp

from .control import GRAVITY, MASS, ROTOR_CONFIG, LeeRotorController, RotorGroup, rotor_allocation
from .warp_utils import torch_warp_stream

_MASS = wp.constant(MASS)
_GRAVITY = wp.constant(GRAVITY)
_HOVER_FORCE = wp.constant(MASS * GRAVITY / 4.0)


@wp.kernel(enable_backward=False)
def _lee_rotor_step(
    state: wp.array2d(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32),
    acceleration: wp.array2d(dtype=wp.float32),
    yaw_target: wp.array2d(dtype=wp.float32),
    use_acceleration: bool,
    use_yaw: bool,
    update_rotors: bool,
    velocity_gain: wp.array2d(dtype=wp.float32),
    attitude_gain: wp.array(dtype=wp.float32),
    angular_rate_gain: wp.array(dtype=wp.float32),
    mixer: wp.array2d(dtype=wp.float32),
    max_thrusts: wp.array(dtype=wp.float32),
    force_constants: wp.array2d(dtype=wp.float32),
    moment_constants: wp.array2d(dtype=wp.float32),
    directions: wp.array2d(dtype=wp.float32),
    tau_up: wp.array2d(dtype=wp.float32),
    tau_down: wp.array2d(dtype=wp.float32),
    throttle: wp.array2d(dtype=wp.float32),
    commands: wp.array2d(dtype=wp.float32),
    thrusts: wp.array2d(dtype=wp.float32),
    moments: wp.array2d(dtype=wp.float32),
):
    env = wp.tid()
    x, y, z, w = state[env, 3], state[env, 4], state[env, 5], state[env, 6]
    xyz = wp.vec3f(x, y, z)
    omega_w = wp.vec3f(state[env, 10], state[env, 11], state[env, 12])
    # Keep the original inverse-rotation expression, including non-unit inputs.
    omega = omega_w * (2.0 * w * w - 1.0) - wp.cross(xyz, omega_w) * (2.0 * w) + xyz * (2.0 * wp.dot(xyz, omega_w))
    acc = wp.vec3f(
        (state[env, 7] - velocity[env, 0]) * velocity_gain[env, 0],
        (state[env, 8] - velocity[env, 1]) * velocity_gain[env, 1],
        (state[env, 9] - velocity[env, 2]) * velocity_gain[env, 2] - _GRAVITY,
    )
    if use_acceleration:
        acc -= wp.vec3f(acceleration[env, 0], acceleration[env, 1], acceleration[env, 2])
    yaw = wp.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    if use_yaw:
        yaw = yaw_target[env, 0]

    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    r0 = wp.vec3f(1.0 - (tyy + tzz), txy + twz, txz - twy)
    r1 = wp.vec3f(txy - twz, 1.0 - (txx + tzz), tyz + twx)
    r2 = wp.vec3f(txz + twy, tyz - twx, 1.0 - (txx + tyy))
    heading = wp.vec3f(wp.cos(yaw), wp.sin(yaw), 0.0)
    b3 = -acc / (wp.length(acc) + 1.0e-6)
    cross = wp.cross(b3, heading)
    b2 = cross / (wp.length(cross) + 1.0e-6)
    b1 = wp.cross(b2, b3)
    attitude_error = 0.5 * wp.vec3f(
        wp.dot(b3, r1) - wp.dot(r2, b2),
        wp.dot(b1, r2) - wp.dot(r0, b3),
        wp.dot(b2, r0) - wp.dot(r1, b1),
    )
    angular_acceleration = wp.vec3f(
        -attitude_error[0] * attitude_gain[0] - omega[0] * angular_rate_gain[0],
        -attitude_error[1] * attitude_gain[1] - omega[1] * angular_rate_gain[1],
        -attitude_error[2] * attitude_gain[2] - omega[2] * angular_rate_gain[2],
    ) + wp.cross(omega, omega)
    thrust = -_MASS * wp.dot(acc, r2)
    for rotor in range(4):
        mixed = (
            mixer[rotor, 0] * angular_acceleration[0]
            + mixer[rotor, 1] * angular_acceleration[1]
            + mixer[rotor, 2] * angular_acceleration[2]
            + mixer[rotor, 3] * thrust
        )
        command = mixed / max_thrusts[rotor] * 2.0 - 1.0
        commands[env, rotor] = command
        if update_rotors:
            target = wp.sqrt(wp.clamp((command + 1.0) / 2.0, 0.0, 1.0))
            previous = throttle[env, rotor]
            tau = tau_down[env, rotor]
            if target > previous:
                tau = tau_up[env, rotor]
            # These are per-step interpolation factors, not time constants in s.
            value = previous + wp.clamp(tau, 0.0, 1.0) * (target - previous)
            throttle[env, rotor] = value
            squared = wp.clamp(value * value, 0.0, 1.0)
            thrusts[env, rotor] = squared * force_constants[env, rotor]
            moments[env, rotor] = squared * moment_constants[env, rotor] * -directions[env, rotor]


@wp.kernel(enable_backward=False)
def _reset_rotors(
    ids: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.bool),
    use_mask: bool,
    max_thrusts: wp.array(dtype=wp.float32),
    nominal_moments: wp.array(dtype=wp.float32),
    nominal_directions: wp.array(dtype=wp.float32),
    throttle: wp.array2d(dtype=wp.float32),
    commands: wp.array2d(dtype=wp.float32),
    thrusts: wp.array2d(dtype=wp.float32),
    moments: wp.array2d(dtype=wp.float32),
):
    row = wp.tid()
    env = ids[row]
    if use_mask and not mask[env]:
        return
    for rotor in range(4):
        hover = _HOVER_FORCE
        value = wp.sqrt(hover / max_thrusts[rotor])
        throttle[env, rotor] = value
        commands[env, rotor] = hover / max_thrusts[rotor] * 2.0 - 1.0
        thrusts[env, rotor] = hover
        moments[env, rotor] = value * value * nominal_moments[rotor] * -nominal_directions[rotor]


class WarpLeeRotorController:
    """Fixed-buffer Lee/RotorGroup implementation for Warp environment graphs.

    ``compute`` accepts the original Torch arguments outside graph capture.
    A Warp task may instead fill ``state_wp``, ``velocity_wp``,
    ``acceleration_wp`` and ``yaw_wp``, then capture ``launch``. Call
    ``advance_rng`` outside that graph once per motor update to preserve the
    original zero-weight Torch noise draw and subsequent seeded event sequence.
    ``apply`` uses the existing simulator composer outside the captured kernel.
    """

    def __init__(self, num_envs, device, dt=0.01):
        device = str(device)
        self.device = device
        self.num_envs = num_envs
        self.rotor_group = RotorGroup(ROTOR_CONFIG, dt).to(device)
        self.rotor_parameters = {
            name: value.expand(num_envs, *value.shape).clone() for name, value in self.rotor_group.named_parameters()
        }
        self.throttle = self.rotor_parameters["throttle"]
        self.max_thrusts = self.rotor_group.KF
        inertia = torch.diag(torch.tensor([0.007, 0.007, 0.012, 1.0], device=device))
        allocation = rotor_allocation(device)[[3, 4, 5, 2]]
        self.mixer = allocation.T @ (allocation @ allocation.T).inverse() @ inertia
        self.attitude_gain = torch.tensor([0.7, 0.7, 0.035], device=device) @ inertia[:3, :3].inverse()
        self.angular_rate_gain = torch.tensor([0.1, 0.1, 0.025], device=device) @ inertia[:3, :3].inverse()
        self.nominal_velocity_gain = torch.tensor([2.2, 2.2, 2.2], device=device)
        self.velocity_gain = self.nominal_velocity_gain.expand(num_envs, -1).clone()
        self.gravity = torch.tensor([0.0, 0.0, GRAVITY], device=device)
        self.rotation_velocities = torch.tensor(ROTOR_CONFIG["max_rotation_velocities"], device=device)
        self.directions = self.rotor_group.directions
        self.allocation = rotor_allocation(device)
        self._base_body_ids = None
        self._joint_order = None
        self.state_wp = wp.zeros((num_envs, 13), dtype=wp.float32, device=device)
        self.velocity_wp = wp.zeros((num_envs, 3), dtype=wp.float32, device=device)
        self.acceleration_wp = wp.zeros((num_envs, 3), dtype=wp.float32, device=device)
        self.yaw_wp = wp.zeros((num_envs, 1), dtype=wp.float32, device=device)
        self.inputs = tuple(
            wp.to_torch(value) for value in (self.state_wp, self.velocity_wp, self.acceleration_wp, self.yaw_wp)
        )
        self.inputs[0][:, 6] = 1.0
        self.commands_wp = wp.empty((num_envs, 4), dtype=wp.float32, device=device)
        self.thrusts_wp = wp.empty_like(self.commands_wp)
        self.moments_wp = wp.empty_like(self.commands_wp)
        self.commands = wp.to_torch(self.commands_wp)
        self._motor_commands_wp = wp.empty_like(self.commands_wp)
        self._motor_commands = wp.to_torch(self._motor_commands_wp)
        self.thrusts = wp.to_torch(self.thrusts_wp)
        self.moments = wp.to_torch(self.moments_wp)
        self._parameter_wp = {name: wp.from_torch(value) for name, value in self.rotor_parameters.items()}
        self.throttle_wp = self._parameter_wp["throttle"]
        self.velocity_gain_wp = wp.from_torch(self.velocity_gain)
        self._attitude_gain_wp = wp.from_torch(self.attitude_gain)
        self._angular_rate_gain_wp = wp.from_torch(self.angular_rate_gain)
        self._mixer_wp = wp.from_torch(self.mixer)
        self._max_thrusts_wp = wp.from_torch(self.max_thrusts)
        self._nominal_moments_wp = wp.from_torch(self.rotor_group.KM)
        self._nominal_directions_wp = wp.from_torch(self.directions)
        self._all_ids = wp.array(list(range(num_envs)), dtype=wp.int32, device=device)
        self._all_mask = wp.ones(num_envs, dtype=wp.bool, device=device)
        self._noise_scratch = torch.empty_like(self.throttle)
        self.reset(None)

    def advance_rng(self):
        """Consume the original zero-weight ``randn_like`` draw, without adding noise."""
        self._noise_scratch.normal_()

    def launch(self, *, use_acceleration=False, use_yaw=False, update_rotors=True, command_output=None):
        """Launch only the fused Warp kernel; safe to capture with fixed inputs."""
        commands = self.commands_wp if command_output is None else command_output
        wp.launch(
            _lee_rotor_step,
            dim=self.num_envs,
            inputs=[
                self.state_wp,
                self.velocity_wp,
                self.acceleration_wp,
                self.yaw_wp,
                use_acceleration,
                use_yaw,
                update_rotors,
                self.velocity_gain_wp,
                self._attitude_gain_wp,
                self._angular_rate_gain_wp,
                self._mixer_wp,
                self._max_thrusts_wp,
                self._parameter_wp["KF"],
                self._parameter_wp["KM"],
                self._parameter_wp["directions"],
                self._parameter_wp["tau_up"],
                self._parameter_wp["tau_down"],
                self.throttle_wp,
            ],
            outputs=[commands, self.thrusts_wp, self.moments_wp],
            device=self.device,
        )
        return commands

    @torch.no_grad()
    def _copy_inputs(self, root_state_xyzw, velocity_world, acceleration_world, yaw_world):
        self.inputs[0].copy_(root_state_xyzw)
        self.inputs[1].copy_(velocity_world)
        if acceleration_world is not None:
            self.inputs[2].copy_(acceleration_world)
        if yaw_world is not None:
            self.inputs[3].copy_(yaw_world.reshape(-1, 1))

    @torch.no_grad()
    def compute(self, root_state_xyzw, velocity_world, acceleration_world=None, yaw_world=None):
        self._copy_inputs(root_state_xyzw, velocity_world, acceleration_world, yaw_world)
        self.advance_rng()
        # Tensor copies and the kernel must use the same CUDA stream. The pure
        # launch path follows the WarpGraphCache's current Warp stream instead.
        stream = torch_warp_stream(self.device)
        with wp.ScopedStream(stream) if stream is not None else wp.ScopedDevice(self.device):
            self.launch(use_acceleration=acceleration_world is not None, use_yaw=yaw_world is not None)
        return self.commands

    @torch.no_grad()
    def motor_commands(self, root_state_xyzw, velocity_world, acceleration_world=None, yaw_world=None):
        self._copy_inputs(root_state_xyzw, velocity_world, acceleration_world, yaw_world)
        stream = torch_warp_stream(self.device)
        with wp.ScopedStream(stream) if stream is not None else wp.ScopedDevice(self.device):
            self.launch(
                use_acceleration=acceleration_world is not None,
                use_yaw=yaw_world is not None,
                update_rotors=False,
                command_output=self._motor_commands_wp,
            )
        return self._motor_commands

    def _launch_reset(self, ids, mask, use_mask):
        wp.launch(
            _reset_rotors,
            dim=ids.shape[0],
            inputs=[ids, mask, use_mask, self._max_thrusts_wp, self._nominal_moments_wp, self._nominal_directions_wp],
            outputs=[self.throttle_wp, self.commands_wp, self.thrusts_wp, self.moments_wp],
            device=self.device,
        )

    def reset(self, env_ids):
        """Reset selected motor buffers; keep sampled gains and motor parameters."""
        if env_ids is None:
            ids = self._all_ids
        elif isinstance(env_ids, wp.array):
            if env_ids.dtype != wp.int32:
                raise TypeError("Warp reset indices must have dtype wp.int32.")
            ids = env_ids
        else:
            values = torch.as_tensor(env_ids, dtype=torch.int32, device=self.device)
            ids = wp.from_torch(values)
        stream = torch_warp_stream(self.device)
        with wp.ScopedStream(stream) if stream is not None else wp.ScopedDevice(self.device):
            self._launch_reset(ids, self._all_mask, False)

    def reset_mask(self, mask):
        """Reset a fixed-shape Warp boolean mask inside an environment graph."""
        self._launch_reset(self._all_ids, mask, True)

    def apply(self, drone, disturbance_acceleration_w=None):
        """Write the same named-body wrench and rotor speeds outside graph capture."""
        return LeeRotorController.apply(self, drone, disturbance_acceleration_w)
