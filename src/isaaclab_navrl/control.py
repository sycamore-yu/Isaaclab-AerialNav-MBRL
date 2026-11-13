"""Original NavRL Lee/RotorGroup equations at the Isaac Lab XYZW boundary.

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
    from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).parent / "assets" / "hummingbird.usd"),
            # Newton reads these USD attributes; the schema does not select a physics backend.
            rigid_props=PhysxRigidBodyCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=NewtonArticulationCfg(self_collision_enabled=False),
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
