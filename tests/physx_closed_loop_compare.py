"""Closed-loop PhysX comparison of two ways to apply NavRL rotor forces.

Use the installed Isaac Lab 3.0 + Isaac Sim 6.2 runtime:

    OMNI_KIT_ACCEPT_EULA=YES /path/to/python tests/physx_closed_loop_compare.py \
        --visualizer none --steps 360

Both drones load the unchanged OmniDrones Hummingbird USD and run the unchanged
OmniDrones LeePositionController and RotorGroup. The only intended change is:

* combined: native Multirotor applies an equivalent wrench at the base COM.
* individual: applies each rotor thrust at its physical rotor body, and the
  rotor drag moments at the base, as in OmniDrones.

This is a force-application comparison, not the full NavRL RL environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, conflict_handler="resolve")
parser.add_argument("--steps", type=int, default=480)
parser.add_argument(
    "--spin-rotors", choices=["none", "both", "individual"], default="none",
    help="Match rotor joint speeds in neither robot, both robots, or only the original-force robot.",
)
parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "docs/results/physx_closed_loop.json")
parser.add_argument("--physics", choices=["isaacsim_physx"], default="isaacsim_physx")
add_launcher_args(parser)
parser.set_defaults(visualizer=["none"])
cli = parser.parse_args()

import numpy as np
import torch
import warp as wp
import yaml

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import instantiate
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab_contrib.actuators import ThrusterCfg
from isaaclab_contrib.assets import MultirotorCfg
from test_thruster_parity import load_legacy_rotor_group

NAVRL = Path(__file__).resolve().parents[2] / "NavRL"
LEGACY = NAVRL / "isaac-training/third_party/OmniDrones/omni_drones"
USD = LEGACY / "robots/assets/usd/hummingbird.usd"


def load_original_controller():
    """Load original source without importing its obsolete simulation package."""
    for package_name in ("omni_drones", "omni_drones.utils"):
        module = types.ModuleType(package_name)
        module.__path__ = []
        sys.modules[package_name] = module

    math_source = LEGACY / "utils/torch.py"
    spec = importlib.util.spec_from_file_location("omni_drones.utils.torch", math_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    controller_source = LEGACY / "controllers/lee_position_controller.py"
    spec = importlib.util.spec_from_file_location("navrl_original_lee", controller_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LeePositionController


def as_torch(x):
    return x if isinstance(x, torch.Tensor) else x.torch


def rotor_allocation(rotor_cfg, device):
    angles = torch.as_tensor(rotor_cfg["rotor_angles"], device=device)
    arms = torch.as_tensor(rotor_cfg["arm_lengths"], device=device)
    directions = torch.as_tensor(rotor_cfg["directions"], device=device)
    coefficients = torch.as_tensor(rotor_cfg["moment_constants"], device=device) / torch.as_tensor(
        rotor_cfg["force_constants"], device=device
    )
    A = torch.zeros(6, 4, dtype=torch.float32, device=device)
    A[2] = 1.0
    A[3] = arms * torch.sin(angles)
    A[4] = -arms * torch.cos(angles)
    A[5] = -directions * coefficients
    return A


def build_cfg(name, position, hover_rps, max_thrust, kf_rps, allocation, rotor_cfg, dt):
    return MultirotorCfg(
        prim_path=f"/World/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD),
            rigid_props=PhysxRigidBodyCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=MultirotorCfg.InitialStateCfg(
            pos=position,
            rot=(0.0, 0.0, 0.0, 1.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            rps={f"rotor_{i}": hover_rps for i in range(4)},
        ),
        actuators={
            "thrusters": ThrusterCfg(
                dt=dt,
                thrust_range=(0.0, max_thrust),
                max_thrust_rate=1e5,
                thrust_const_range=(kf_rps, kf_rps),
                tau_inc_range=(0.0, 0.0),
                tau_dec_range=(0.0, 0.0),
                torque_to_thrust_ratio=float(rotor_cfg["moment_constants"][0] / rotor_cfg["force_constants"][0]),
                use_discrete_approximation=True,
                integration_scheme="euler",
                # Config validation counts expressions, not their regex matches.
                thruster_names_expr=[f"rotor_{i}" for i in range(4)],
            )
        },
        rotor_directions=list(rotor_cfg["directions"]),
        allocation_matrix=allocation.tolist(),
    )


def state_for_lee(asset, device):
    data = asset.data
    pos = as_torch(data.root_pos_w)
    quat = as_torch(data.root_quat_w)
    vel = as_torch(data.root_lin_vel_w)
    ang = as_torch(data.root_ang_vel_w)
    # Original OmniDrones expects WXYZ and world-frame linear/angular velocities.
    return torch.cat((pos, quat[:, 3:4], quat[:, :3], vel, ang), dim=-1).to(device)


def apply_individual_thrust(asset, forces, moments, rotor_body_ids):
    """Apply thrust on each rotor body and drag moments at the base (body frame)."""
    count = asset.num_bodies
    body_names = asset.body_names
    base_id = body_names.index("base_link")
    f = torch.zeros((1, count, 3), device=forces.device)
    m = torch.zeros_like(f)
    f[0, rotor_body_ids, 2] = forces[0]
    # Rotor axes are along local +Z in the unchanged USD.
    m[0, base_id, 2] = moments.sum()
    asset.root_view.apply_forces_and_torques_at_position(
        force_data=wp.from_torch(f.reshape(-1, 3), dtype=wp.float32),
        torque_data=wp.from_torch(m.reshape(-1, 3), dtype=wp.float32),
        position_data=None,
        indices=asset._ALL_INDICES,
        is_global=False,
    )


def spin_rotor_joints(asset, rotor, rotor_cfg):
    """Reproduce the legacy joint-velocity update for all four rotor revolute joints."""
    joint_vel = as_torch(asset.data.joint_vel).clone()
    for rotor_id in range(4):
        joint_name = f"rotor_{rotor_id}_joint"
        assert joint_name in asset.joint_names
        joint_vel[0, asset.joint_names.index(joint_name)] = (
            rotor.throttle[rotor_id]
            * rotor_cfg["directions"][rotor_id]
            * rotor_cfg["max_rotation_velocities"][rotor_id]
        )
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel)


def yaw_from_quat_xyzw(quat):
    """Extract world yaw without changing the physical quaternion convention."""
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=cli) as physics_cfg:
        dt = 0.016  # Original NavRL physics step, also one controller/rotor update.
        sim_cfg = sim_utils.SimulationCfg(
            dt=dt,
            device=cli.device,
            physics=physics_cfg,
            use_newton_actuators=False,
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        with open(LEGACY / "robots/assets/usd/hummingbird.yaml", encoding="utf-8") as stream:
            params = yaml.safe_load(stream)
        rotor_cfg = params["rotor_configuration"]
        allocation = rotor_allocation(rotor_cfg, cli.device)
        max_force = rotor_cfg["force_constants"][0] * rotor_cfg["max_rotation_velocities"][0] ** 2
        kf_rps = rotor_cfg["force_constants"][0] * (2 * math.pi) ** 2
        hover_force = params["mass"] * 9.81 / 4
        hover_rps = math.sqrt(hover_force / kf_rps)
        combined = instantiate(
            build_cfg("Combined", (0.0, -3.0, 2.0), hover_rps, max_force, kf_rps, allocation, rotor_cfg, dt)
        )
        individual = instantiate(
            build_cfg("Individual", (0.0, 3.0, 2.0), hover_rps, max_force, kf_rps, allocation, rotor_cfg, dt)
        )
        sim.reset()

        print("COMPARE_BODY_NAMES", combined.body_names, individual.body_names, flush=True)
        print("COMPARE_MASS", as_torch(combined.data.body_mass), flush=True)
        print("COMPARE_JOINTS", individual.joint_names, flush=True)
        assert len(combined.thruster_names) == 4 and len(individual.thruster_names) == 4
        assert individual.body_names == combined.body_names
        assert set(combined.thruster_names) == {f"rotor_{i}" for i in range(4)}
        rotor_body_ids = [individual.body_names.index(f"rotor_{i}") for i in range(4)]
        # Multirotor orders forces by joint name, which need not match rotor indices.
        ordered_thrust_indices = [
            int(name.removeprefix("rotor_")) for name in combined.thruster_names
        ]

        Lee = load_original_controller()
        Rotor = load_legacy_rotor_group(NAVRL)
        controllers = [Lee(9.81, params).to(cli.device) for _ in range(2)]
        rotors = [Rotor(rotor_cfg, dt).to(cli.device) for _ in range(2)]
        for rotor in rotors:
            rotor.throttle.copy_(torch.sqrt(torch.tensor(hover_force, device=cli.device) / rotor.KF))
        base_positions = torch.tensor([[0, -3, 2], [0, 3, 2]], dtype=torch.float32, device=cli.device)
        trajectory = []
        phase_size = max(1, cli.steps // 4)

        with torch.no_grad():
            for step in range(cli.steps):
                if step < phase_size:
                    scenario, target_vel = "hover", (0.0, 0.0, 0.0)
                elif step < 2 * phase_size:
                    scenario, target_vel = "velocity_x", (1.0, 0.0, 0.0)
                elif step < 3 * phase_size:
                    scenario, target_vel = "velocity_z", (0.0, 0.0, 0.5)
                else:
                    # NavRL normally keeps the current yaw. This extra controller
                    # channel tests the torque allocation under a yaw command.
                    scenario, target_vel = "yaw_step", (0.0, 0.0, 0.0)

                inputs = []
                for idx, robot in enumerate((combined, individual)):
                    velocity = torch.tensor([target_vel], dtype=torch.float32, device=cli.device)
                    yaw_ref = torch.tensor([[1.0471975512]], device=cli.device) if scenario == "yaw_step" else None
                    command = controllers[idx](
                        state_for_lee(robot, cli.device), target_vel=velocity, target_yaw=yaw_ref
                    )
                    actual_force, drag_moment = rotors[idx](command.squeeze(0))
                    inputs.append((actual_force.unsqueeze(0), drag_moment.unsqueeze(0)))

                thrust_combined = inputs[0][0][:, ordered_thrust_indices]
                combined.set_thrust_target(thrust_combined)
                # Native model combines the 4 thrusts into a 6D wrench at the COM.
                combined.write_data_to_sim()

                # Original model instead applies each thrust at the rotor body.
                apply_individual_thrust(individual, inputs[1][0], inputs[1][1], rotor_body_ids)

                if cli.spin_rotors in ("both", "individual"):
                    spin_rotor_joints(individual, rotors[1], rotor_cfg)
                if cli.spin_rotors == "both":
                    spin_rotor_joints(combined, rotors[0], rotor_cfg)

                sim.step(render=False)
                combined.update(dt)
                individual.update(dt)
                p = torch.cat((as_torch(combined.data.root_pos_w), as_torch(individual.data.root_pos_w)), dim=0)
                v = torch.cat((as_torch(combined.data.root_lin_vel_w), as_torch(individual.data.root_lin_vel_w)), dim=0)
                w = torch.cat((as_torch(combined.data.root_ang_vel_w), as_torch(individual.data.root_ang_vel_w)), dim=0)
                q = torch.cat((as_torch(combined.data.root_quat_w), as_torch(individual.data.root_quat_w)), dim=0)
                yaw = yaw_from_quat_xyzw(q)
                trajectory.append({
                    "step": step,
                    "phase": scenario,
                    "combined_p": (p[0] - base_positions[0]).cpu().tolist(),
                    "individual_p": (p[1] - base_positions[1]).cpu().tolist(),
                    "combined_v": v[0].cpu().tolist(),
                    "individual_v": v[1].cpu().tolist(),
                    "combined_omega": w[0].cpu().tolist(),
                    "individual_omega": w[1].cpu().tolist(),
                    "combined_yaw_rad": float(yaw[0]),
                    "individual_yaw_rad": float(yaw[1]),
                    "position_difference_m": (p[0] - p[1] - base_positions[0] + base_positions[1]).norm().item(),
                    "velocity_difference_mps": (v[0] - v[1]).norm().item(),
                    "yaw_difference_rad": float(torch.atan2(torch.sin(yaw[0] - yaw[1]), torch.cos(yaw[0] - yaw[1])).abs()),
                })
                if step % 60 == 0:
                    print("COMPARE_STEP", step, scenario, trajectory[-1]["position_difference_m"], flush=True)

        p_error = np.array([x["position_difference_m"] for x in trajectory])
        v_error = np.array([x["velocity_difference_mps"] for x in trajectory])
        yaw_error = np.array([x["yaw_difference_rad"] for x in trajectory])
        by_phase = {}
        for label in ("hover", "velocity_x", "velocity_z", "yaw_step"):
            matching = [x for x in trajectory if x["phase"] == label]
            if not matching:
                continue
            by_phase[label] = {
                "rms_position_difference_m": float(np.sqrt(np.mean([x["position_difference_m"] ** 2 for x in matching]))),
                "max_position_difference_m": float(max(x["position_difference_m"] for x in matching)),
                "rms_velocity_difference_mps": float(np.sqrt(np.mean([x["velocity_difference_mps"] ** 2 for x in matching]))),
                "rms_yaw_difference_deg": float(np.sqrt(np.mean([x["yaw_difference_rad"] ** 2 for x in matching])) * 180 / math.pi),
                "combined_final_z_relative_m": matching[-1]["combined_p"][2],
                "individual_final_z_relative_m": matching[-1]["individual_p"][2],
            }
        result = {
            "task": "closed-loop Hummingbird rotor application comparison",
            "isaac_lab_version": "3.0.0",
            "isaac_sim_version": "6.2.0",
            "world": "two identical original Hummingbird USD assets in empty gravity field",
            "dt": dt,
            "steps": cli.steps,
            "spin_rotors": cli.spin_rotors,
            "variants": ["combined_body_wrench", "individual_rotor_body_forces"],
            "rms_position_difference_m": float(np.sqrt(np.mean(p_error ** 2))),
            "max_position_difference_m": float(p_error.max()),
            "rms_velocity_difference_mps": float(np.sqrt(np.mean(v_error ** 2))),
            "rms_yaw_difference_deg": float(np.sqrt(np.mean(yaw_error ** 2)) * 180 / math.pi),
            "max_yaw_difference_deg": float(yaw_error.max() * 180 / math.pi),
            "per_phase": by_phase,
            "trajectory": trajectory,
        }
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("COMPARE_RESULT", json.dumps({key: val for key, val in result.items() if key != "trajectory"}), flush=True)


if __name__ == "__main__":
    main()
