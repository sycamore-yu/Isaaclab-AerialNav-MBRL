"""Fixed-buffer Lee control, motor integration and simulator wrench contracts."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab_navrl.control import GRAVITY, MASS
from isaaclab_navrl.warp_control import GraphedWarpController, WarpLeeRotorController


def _inputs(count):
    state = torch.zeros(count, 13)
    state[:, 6] = 1
    return state, torch.zeros(count, 3)


def test_lee_command_values_and_pure_query_does_not_advance_motors():
    controller = WarpLeeRotorController(2, "cpu", dt=0.05)
    state, velocity = _inputs(2)
    state[1, 7:10] = torch.tensor([1.0, -2.0, 0.5])
    state[1, 10:] = torch.tensor([0.1, -0.2, 0.3])
    velocity[1] = torch.tensor([0.2, 0.1, -0.1])
    acceleration = torch.tensor([[0.0, 0.0, 0.0], [0.7, -0.4, 0.3]])
    yaw = torch.tensor([[0.2], [-0.3]])
    before = controller.throttle.clone()
    commands = controller.motor_commands(state, velocity, acceleration, yaw)
    wp.synchronize_device("cpu")
    torch.testing.assert_close(
        commands,
        torch.tensor(
            [[-0.37878442, -0.45117742, -0.37878442, -0.45117742], [-0.46778834, -0.68455833, -0.64827037, -0.10261726]]
        ),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(controller.throttle, before, rtol=0, atol=0)


@pytest.mark.parametrize("controller_type", [WarpLeeRotorController, GraphedWarpController])
def test_motor_response_partial_reset_and_stable_storage(controller_type):
    controller = controller_type(4, "cpu", dt=0.05)
    state, velocity = _inputs(4)
    state[:, 7] = torch.tensor([0.0, 1.0, -2.0, 3.0])
    pointers = {name: getattr(controller, name).data_ptr() for name in ("commands", "throttle", "thrusts", "moments")}
    for _ in range(12):
        before = controller.throttle.clone()
        controller.compute(state, velocity)
        wp.synchronize_device("cpu")
        target = ((controller.commands + 1) / 2).clamp(0, 1).sqrt()
        tau = torch.where(
            target > before, controller.rotor_parameters["tau_up"], controller.rotor_parameters["tau_down"]
        )
        expected = before + tau.clamp(0, 1) * (target - before)
        torch.testing.assert_close(controller.throttle, expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(controller.thrusts, expected.square() * controller.rotor_parameters["KF"])
        assert pointers == {name: getattr(controller, name).data_ptr() for name in pointers}
    retained = {name: getattr(controller, name)[1:].clone() for name in pointers}
    controller.reset(torch.tensor([0], dtype=torch.int32))
    wp.synchronize_device("cpu")
    torch.testing.assert_close(controller.thrusts[0], torch.full((4,), MASS * GRAVITY / 4))
    for name, expected in retained.items():
        torch.testing.assert_close(getattr(controller, name)[1:], expected, rtol=0, atol=0)


def test_compute_consumes_exactly_one_motor_noise_draw():
    controller = WarpLeeRotorController(4, "cpu")
    saved = torch.get_rng_state().clone()
    try:
        torch.randn_like(controller.throttle)
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(saved)
        controller.compute(*_inputs(4))
        assert torch.equal(torch.get_rng_state(), expected_rng)
    finally:
        torch.set_rng_state(saved)


def test_apply_targets_base_link_and_named_rotor_joints_once():
    controller = WarpLeeRotorController(2, "cpu")
    captured = {}
    drone = SimpleNamespace(
        body_names=["rotor_0", "base_link"],
        joint_names=["rotor_2_joint", "rotor_0_joint", "rotor_3_joint", "rotor_1_joint"],
        data=SimpleNamespace(joint_vel=SimpleNamespace(torch=torch.zeros(2, 4))),
        permanent_wrench_composer=SimpleNamespace(set_forces_and_torques_index=lambda **kw: captured.update(kw)),
        write_joint_velocity_to_sim_index=lambda **kw: captured.update(kw),
    )
    controller.apply(drone)
    assert captured["body_ids"] == [1] and not captured["is_global"]
    torch.testing.assert_close(captured["forces"][:, 0, 2], torch.full((2,), MASS * GRAVITY))
    torch.testing.assert_close(captured["torques"], torch.zeros(2, 1, 3), atol=1e-6, rtol=0)
    expected = controller.throttle * controller.directions * controller.rotation_velocities
    torch.testing.assert_close(captured["velocity"][:, [1, 3, 0, 2]], expected)
