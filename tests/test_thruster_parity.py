"""Compare the real Navigation RotorGroup and Isaac Lab 3.0 Thruster on the CPU.

Run with the Python interpreter of an Isaac Lab 3.0 installation:
    python tests/test_thruster_parity.py --aerialnav-root ../Navigation

This tests motor thrust and the rigid-body allocation matrix. It does not
test the Lee controller or PhysX flight trajectories.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import torch
import yaml
from isaaclab_contrib.actuators import Thruster, ThrusterCfg
from isaaclab_contrib.utils.types import MultiRotorActions


def load_legacy_rotor_group(aerialnav_root: Path):
    path = aerialnav_root / "isaac-training/third_party/OmniDrones/omni_drones/actuators/rotor_group.py"
    spec = importlib.util.spec_from_file_location("aerialnav_legacy_rotor_group", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RotorGroup


@torch.no_grad()
def run(aerialnav_root: Path, num_envs: int, num_steps: int) -> None:
    params_path = aerialnav_root / "isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/hummingbird.yaml"
    with params_path.open(encoding="utf-8") as stream:
        params = yaml.safe_load(stream)

    rotor_cfg = params["rotor_configuration"]
    rotor_cls = load_legacy_rotor_group(aerialnav_root)
    dt = 0.016
    # In the original code, 0.43 is the throttle update factor PER STEP.
    alpha = 0.43
    tau_s = dt * (1.0 - alpha) / alpha
    rotor_count = int(rotor_cfg["num_rotors"])
    omega_max = torch.tensor(rotor_cfg["max_rotation_velocities"], dtype=torch.float32)
    kf = torch.tensor(rotor_cfg["force_constants"], dtype=torch.float32)
    km = torch.tensor(rotor_cfg["moment_constants"], dtype=torch.float32)
    directions = torch.tensor(rotor_cfg["directions"], dtype=torch.float32)
    angles = torch.tensor(rotor_cfg["rotor_angles"], dtype=torch.float32)
    arms = torch.tensor(rotor_cfg["arm_lengths"], dtype=torch.float32)
    max_force = omega_max.square() * kf
    assert torch.allclose(max_force, max_force[0].expand_as(max_force))
    assert torch.allclose(kf, kf[0].expand_as(kf))
    hover_per_rotor = float(params["mass"]) * 9.81 / rotor_count

    # The original rotor speed is in rad/s; the new Thruster uses revolutions/s.
    thrust_const_rps = kf[0].item() * (2.0 * math.pi) ** 2
    hover_rps = math.sqrt(hover_per_rotor / thrust_const_rps)
    new = Thruster(
        cfg=ThrusterCfg(
            dt=dt,
            thrust_range=(0.0, max_force[0].item()),
            max_thrust_rate=100000.0,
            thrust_const_range=(thrust_const_rps, thrust_const_rps),
            tau_inc_range=(tau_s, tau_s),
            tau_dec_range=(tau_s, tau_s),
            torque_to_thrust_ratio=(km / kf)[0].item(),
            use_discrete_approximation=True,
            integration_scheme="euler",
            thruster_names_expr=["rotor_[0-3]"],
        ),
        thruster_names=[f"rotor_{i}" for i in range(rotor_count)],
        thruster_ids=slice(None),
        num_envs=num_envs,
        device="cpu",
        init_thruster_rps=torch.full((num_envs, rotor_count), hover_rps),
    )

    old = [rotor_cls(rotor_cfg, dt).cpu() for _ in range(num_envs)]
    for rotor in old:
        rotor.throttle.copy_(torch.sqrt(torch.tensor(hover_per_rotor) / rotor.KF))

    # Wrench order in Multirotor: [Fx, Fy, Fz, Tx, Ty, Tz].
    allocation = torch.zeros(6, rotor_count, dtype=torch.float32)
    allocation[2] = 1.0
    allocation[3] = arms * torch.sin(angles)
    allocation[4] = -arms * torch.cos(angles)
    allocation[5] = -directions * (km / kf)
    assert torch.allclose(allocation[5], -directions * (km / kf))

    torch.manual_seed(20261028)
    max_abs_motor_error = 0.0
    max_abs_wrench_error = 0.0
    for _ in range(num_steps):
        # Include values outside [-1, 1] to test clipping in both implementations.
        cmds = torch.rand(num_envs, rotor_count) * 2.4 - 1.2
        old_results = [rotor(cmds[i]) for i, rotor in enumerate(old)]
        old_forces = torch.stack([result[0] for result in old_results])
        old_yaw_moments = torch.stack([result[1] for result in old_results])
        requested_forces = torch.clamp((cmds + 1.0) / 2.0, 0.0, 1.0) * max_force
        new_forces = new.compute(MultiRotorActions(thrusts=requested_forces, thruster_indices=slice(None))).thrusts
        max_abs_motor_error = max(max_abs_motor_error, (old_forces - new_forces).abs().max().item())

        old_wrench = old_forces @ allocation.T
        old_wrench[:, 5] = old_yaw_moments.sum(dim=1)
        new_wrench = new_forces @ allocation.T
        max_abs_wrench_error = max(max_abs_wrench_error, (old_wrench - new_wrench).abs().max().item())

    print(f"envs={num_envs} rotors={rotor_count} steps={num_steps} dt={dt}")
    print(f"old_alpha={alpha} equivalent_new_tau_s={tau_s:.10f}")
    print(f"max_thrust_N={max_force[0].item():.8f} hover_thrust_N={hover_per_rotor:.8f}")
    print(f"max_abs_motor_error_N={max_abs_motor_error:.9g}")
    print(f"max_abs_wrench_error={max_abs_wrench_error:.9g}")
    assert max_abs_motor_error < 1e-5
    assert max_abs_wrench_error < 1e-5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aerialnav-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "Navigation",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=250)
    args = parser.parse_args()
    run(args.aerialnav_root.resolve(), args.num_envs, args.num_steps)
