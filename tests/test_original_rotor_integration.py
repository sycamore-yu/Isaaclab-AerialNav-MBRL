"""Verify that Navigation's unchanged RotorGroup works with the Isaac Lab 3.0 motor interface.

The native Thruster is configured as a zero-lag thrust passthrough. The legacy
RotorGroup alone performs the motor response calculation. This is a CPU-only
contract test; it does not launch PhysX or instantiate Multirotor.
"""

import math

import pytest
import torch
import yaml
from isaaclab_contrib.actuators import Thruster, ThrusterCfg
from isaaclab_contrib.utils.types import MultiRotorActions
from reference_repo import UPSTREAM_NAVIGATION
from test_thruster_parity import load_legacy_rotor_group
from torch.func import functional_call, vmap


def test_original_rotor_group(num_envs=8, num_steps=250):
    root = UPSTREAM_NAVIGATION
    if not (root / ".git").exists():
        pytest.skip("Set UPSTREAM_NAVIGATION_SOURCE to a pinned upstream checkout")
    param_file = root / "isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/hummingbird.yaml"
    with param_file.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    rotor_cfg = config["rotor_configuration"]
    rotor_cls = load_legacy_rotor_group(root)
    dt = 0.016
    count = rotor_cfg["num_rotors"]
    force_constants = torch.tensor(rotor_cfg["force_constants"])
    max_rot_vel = torch.tensor(rotor_cfg["max_rotation_velocities"])
    max_forces = force_constants * max_rot_vel.square()
    hover_force = config["mass"] * 9.81 / count
    hover_throttle = torch.sqrt(torch.tensor(hover_force) / max_forces)

    # Keep the real legacy class. Modern PyTorch replaces the removed
    # tensordict.nn.make_functional call with functional_call + vmap.
    rotor = rotor_cls(rotor_cfg, dt)
    params = {
        name: value.detach().unsqueeze(0).expand(num_envs, *value.shape).clone()
        for name, value in rotor.named_parameters()
    }
    params["throttle"][:] = hover_throttle
    serial = [rotor_cls(rotor_cfg, dt) for _ in range(num_envs)]
    for instance in serial:
        instance.throttle.copy_(hover_throttle)

    thrust_const = float(force_constants[0]) * (2 * math.pi) ** 2
    max_force = float(max_forces[0])
    passthrough = Thruster(
        cfg=ThrusterCfg(
            dt=dt,
            thrust_range=(0.0, max_force),
            max_thrust_rate=100000.0,
            thrust_const_range=(thrust_const, thrust_const),
            tau_inc_range=(0.0, 0.0),
            tau_dec_range=(0.0, 0.0),
            torque_to_thrust_ratio=0.016,
            use_discrete_approximation=True,
            integration_scheme="euler",
            thruster_names_expr=["rotor_[0-3]"],
        ),
        thruster_names=[f"rotor_{i}" for i in range(count)],
        thruster_ids=slice(None),
        num_envs=num_envs,
        device="cpu",
        init_thruster_rps=torch.full((num_envs, count), math.sqrt(hover_force / thrust_const)),
    )

    torch.manual_seed(20261028)
    max_rotor_error = 0.0
    max_passthrough_error = 0.0
    max_moment_error = 0.0
    directions = torch.tensor(rotor_cfg["directions"])
    ratio = torch.tensor(rotor_cfg["moment_constants"]) / torch.tensor(rotor_cfg["force_constants"])

    with torch.no_grad():
        for step in range(num_steps):
            cmds = torch.rand(num_envs, count) * 2.4 - 1.2
            thrust_old, moment_old = zip(*(r(cmds[i]) for i, r in enumerate(serial)), strict=True)
            thrust_old = torch.stack(thrust_old)
            moment_old = torch.stack(moment_old)
            thrust_batch, moment_batch = vmap(
                lambda state, command: functional_call(rotor, state, (command,), strict=True),
                randomness="different",
            )(params, cmds)
            applied = passthrough.compute(MultiRotorActions(thrusts=thrust_batch, thruster_indices=slice(None))).thrusts
            max_rotor_error = max(
                max_rotor_error,
                (thrust_batch - thrust_old).abs().max().item(),
                (moment_batch - moment_old).abs().max().item(),
            )
            max_passthrough_error = max(max_passthrough_error, (applied - thrust_batch).abs().max().item())
            # The native allocation matrix must reproduce this yaw row.
            native_yaw_moment = (applied * (-directions * ratio)).sum(dim=-1)
            max_moment_error = max(
                max_moment_error,
                (native_yaw_moment - moment_old.sum(dim=-1)).abs().max().item(),
            )
            if step == num_steps // 2:
                # Reset a single robot's motor state; other robots keep theirs.
                serial[0].throttle.copy_(hover_throttle)
                params["throttle"][0] = hover_throttle
    print(
        "original RotorGroup vmap vs serial max error:",
        max_rotor_error,
        "N/Nm",
    )
    print("zero-lag Thruster passthrough max error:", max_passthrough_error, "N")
    print("yaw allocation max error:", max_moment_error, "Nm")
    assert max_rotor_error < 1e-5
    assert max_passthrough_error < 1e-5
    assert max_moment_error < 1e-5


if __name__ == "__main__":
    test_original_rotor_group()
