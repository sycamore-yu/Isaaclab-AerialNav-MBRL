"""Original Navigation Lee/RotorGroup equations at the Isaac Lab XYZW boundary.

Adapted from OmniDrones, copyright (c) 2023 Botian Xu, Tsinghua University.
See LICENSE-OmniDrones (MIT).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

MASS = 0.716
GRAVITY = 9.81
ROTOR_CONFIG = {
    "arm_lengths": [0.17] * 4,
    "directions": [-1.0, 1.0, -1.0, 1.0],
    "force_constants": [8.54858e-6] * 4,
    "moment_constants": [1.3677728816219314e-7] * 4,
    "max_rotation_velocities": [838] * 4,
    "rotor_angles": [0.0, 1.57079632679, 3.14159265359, -1.57079632679],
}


def rotor_allocation(device="cpu"):
    angles = torch.as_tensor(ROTOR_CONFIG["rotor_angles"], device=device)
    arms = torch.as_tensor(ROTOR_CONFIG["arm_lengths"], device=device)
    directions = torch.as_tensor(ROTOR_CONFIG["directions"], device=device)
    ratio = torch.as_tensor(ROTOR_CONFIG["moment_constants"], device=device) / torch.as_tensor(
        ROTOR_CONFIG["force_constants"], device=device
    )
    allocation = torch.zeros(6, 4, device=device)
    allocation[2] = 1.0
    allocation[3] = arms * torch.sin(angles)
    allocation[4] = -arms * torch.cos(angles)
    allocation[5] = -directions * ratio
    return allocation


def make_drone_cfg(prim_path):
    """Load the same mechanical model through Isaac Lab's backend factory."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).parent / "assets" / "hummingbird.usd"),
            rigid_props=PhysxRigidBodyCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=[
                PhysxArticulationCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                ),
                NewtonArticulationCfg(self_collision_enabled=False),
            ],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        actuators={
            # RotorGroup supplies the motor dynamics and body wrench. The USD
            # rotor joints remain passive; their speeds are written by name.
            "rotors": ImplicitActuatorCfg(
                joint_names_expr=["rotor_.*_joint"],
                stiffness=0.0,
                damping=0.0,
            )
        },
    )


class RotorGroup(nn.Module):
    """Original RotorGroup; independent motor states are batched externally."""

    def __init__(self, rotor_config, dt: float):
        super().__init__()
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        moment_constants = torch.as_tensor(rotor_config["moment_constants"])
        max_rot_vels = torch.as_tensor(rotor_config["max_rotation_velocities"]).float()
        self.num_rotors = len(force_constants)
        self.dt = dt
        self.time_up = 0.15
        self.time_down = 0.15
        self.noise_scale = 0.002
        self.KF = nn.Parameter(max_rot_vels.square() * force_constants)
        self.KM = nn.Parameter(max_rot_vels.square() * moment_constants)
        self.throttle = nn.Parameter(torch.zeros(self.num_rotors))
        self.directions = nn.Parameter(torch.as_tensor(rotor_config["directions"]).float())
        self.tau_up = nn.Parameter(0.43 * torch.ones(self.num_rotors))
        self.tau_down = nn.Parameter(0.43 * torch.ones(self.num_rotors))
        self.f = torch.square
        self.f_inv = torch.sqrt
        self.requires_grad_(False)

    def forward(self, cmds: torch.Tensor):
        target_throttle = self.f_inv(torch.clamp((cmds + 1) / 2, 0, 1))
        tau = torch.where(target_throttle > self.throttle, self.tau_up, self.tau_down)
        tau = torch.clamp(tau, 0, 1)
        self.throttle.add_(tau * (target_throttle - self.throttle))
        noise = torch.randn_like(self.throttle) * self.noise_scale * 0.0
        t = torch.clamp(self.f(self.throttle) + noise, 0.0, 1.0)
        return t * self.KF, (t * self.KM) * -self.directions


def _rotation_matrix_wxyz(q):
    w, x, y, z = q.unbind(-1)
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    return torch.stack(
        (
            1 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1 - (txx + tyy),
        ),
        -1,
    ).unflatten(-1, (3, 3))


def _rotate_inverse_wxyz(q, vector):
    w, xyz = q[:, 0], q[:, 1:]
    a = vector * (2.0 * w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(xyz, vector, dim=-1) * w.unsqueeze(-1) * 2.0
    c = xyz * torch.bmm(xyz.view(-1, 1, 3), vector.view(-1, 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def _normalize(vector):
    return vector / (torch.norm(vector, dim=-1, keepdim=True) + 1e-6)


class LeeRotorController:
    """Original Lee velocity/acceleration control and motor integration per physics step."""

    def __init__(self, num_envs, device, dt=0.01):
        self.device = device
        self.num_envs = num_envs
        self.rotor_group = RotorGroup(ROTOR_CONFIG, dt).to(device)
        self.rotor_parameters = {
            name: value.expand(num_envs, *value.shape).clone() for name, value in self.rotor_group.named_parameters()
        }
        self._motor_step = torch.vmap(
            lambda parameters, command: torch.func.functional_call(self.rotor_group, parameters, (command,)),
            randomness="different",
        )
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
        self.thrusts = torch.empty(num_envs, 4, device=device)
        self.moments = torch.empty_like(self.thrusts)
        self.commands = torch.empty_like(self.thrusts)
        self.allocation = rotor_allocation(device)
        self._base_body_ids = None
        self._joint_order = None
        self.reset(torch.arange(num_envs, device=device))

    def reset(self, env_ids):
        self.throttle[env_ids] = torch.sqrt(MASS * GRAVITY / 4 / self.max_thrusts)
        self.thrusts[env_ids] = MASS * GRAVITY / 4
        self.moments[env_ids] = self.throttle[env_ids].square() * self.rotor_group.KM * -self.directions
        self.commands[env_ids] = MASS * GRAVITY / 4 / self.max_thrusts * 2 - 1

    def motor_commands(self, root_state_xyzw, velocity_world, acceleration_world=None, yaw_world=None):
        """Pure original Lee formula; state uses XYZW and world-frame velocities."""
        q_xyzw = root_state_xyzw[:, 3:7]
        q = torch.cat((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1)
        angular_velocity = _rotate_inverse_wxyz(q, root_state_xyzw[:, 10:13])
        velocity_error = root_state_xyzw[:, 7:10] - velocity_world
        acc = velocity_error * self.velocity_gain - self.gravity
        if acceleration_world is not None:
            acc = acc - acceleration_world
        if yaw_world is None:
            w, x, y, z = q.unbind(-1)
            yaw_world = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yaw = yaw_world.reshape(-1, 1)
        rotation = _rotation_matrix_wxyz(q)
        b1 = torch.cat((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), -1)
        b3 = -_normalize(acc)
        b2 = _normalize(torch.cross(b3, b1, dim=1))
        desired = torch.stack((torch.cross(b2, b3, dim=1), b2, b3), -1)
        error_matrix = 0.5 * (
            torch.bmm(desired.transpose(-2, -1), rotation) - torch.bmm(rotation.transpose(-2, -1), desired)
        )
        attitude_error = torch.stack((error_matrix[:, 2, 1], error_matrix[:, 0, 2], error_matrix[:, 1, 0]), -1)
        angular_acceleration = (
            -attitude_error * self.attitude_gain
            - angular_velocity * self.angular_rate_gain
            + torch.cross(angular_velocity, angular_velocity, dim=-1)
        )
        thrust = -MASS * (acc * rotation[:, :, 2]).sum(-1, True)
        command = (self.mixer @ torch.cat((angular_acceleration, thrust), -1).T).T
        return command / self.max_thrusts * 2 - 1

    def compute(self, root_state_xyzw, velocity_world, acceleration_world=None, yaw_world=None):
        self.commands = self.motor_commands(root_state_xyzw, velocity_world, acceleration_world, yaw_world)
        self.thrusts, self.moments = self._motor_step(self.rotor_parameters, self.commands)
        return self.commands

    def apply(self, drone, disturbance_acceleration_w=None):
        """Apply the RotorGroup wrench at the named base link's centre of mass."""
        if self._base_body_ids is None:
            self._base_body_ids = [drone.body_names.index("base_link")]
            self._joint_order = [drone.joint_names.index(f"rotor_{i}_joint") for i in range(4)]
        wrench = (self.allocation @ self.thrusts.T).T
        force = wrench[:, :3]
        if disturbance_acceleration_w is not None:
            from isaaclab.utils.math import quat_apply_inverse

            mass = drone.data.body_mass.torch.sum(-1, keepdim=True)
            force = force + quat_apply_inverse(drone.data.root_quat_w.torch, disturbance_acceleration_w * mass)
        drone.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force[:, None, :].contiguous(),
            torques=wrench[:, None, 3:].contiguous(),
            body_ids=self._base_body_ids,
            is_global=False,
        )
        joint_velocity = drone.data.joint_vel.torch.clone()
        joint_velocity[:, self._joint_order] = self.throttle * self.directions * self.rotation_velocities
        drone.write_joint_velocity_to_sim_index(velocity=joint_velocity)
