"""Numerical and execution parity of Warp control against the original Torch equations."""

from __future__ import annotations

import pytest
import torch
import warp as wp

from aerialnav.control import GRAVITY, LeeRotorController
from aerialnav.warp_control import WarpLeeRotorController

_DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
]


def _inputs(num_envs, device, generator):
    state = torch.randn((num_envs, 13), device=device, generator=generator)
    quaternion = state[:, 3:7]
    state[:, 3:7] = quaternion / quaternion.norm(dim=-1, keepdim=True)
    velocity = 2.0 * torch.randn((num_envs, 3), device=device, generator=generator)
    acceleration = 4.0 * torch.randn((num_envs, 3), device=device, generator=generator)
    yaw = torch.randn((num_envs, 1), device=device, generator=generator)
    return state, velocity, acceleration, yaw


def _assert_outputs(actual, expected):
    for name in ("commands", "throttle", "thrusts", "moments"):
        value = getattr(actual, name)
        assert torch.isfinite(value).all(), name
        torch.testing.assert_close(value, getattr(expected, name), rtol=2e-5, atol=2e-5, msg=name)


@pytest.mark.parametrize("device", _DEVICES)
def test_fused_control_matches_original_with_randomization_and_partial_reset(device):
    count = 19
    generator = torch.Generator(device=device).manual_seed(20261030)
    expected = LeeRotorController(count, device, dt=0.05)
    actual = WarpLeeRotorController(count, device, dt=0.05)
    pointers = {name: getattr(actual, name).data_ptr() for name in ("commands", "throttle", "thrusts", "moments")}
    for name in ("KF", "KM"):
        scale = 0.9 + 0.2 * torch.rand((count, 4), device=device, generator=generator)
        value = expected.rotor_parameters[name] * scale
        expected.rotor_parameters[name].copy_(value)
        actual.rotor_parameters[name].copy_(value)
    # Include both asymmetric response and values clipped to the legal interval.
    for name in ("tau_up", "tau_down"):
        value = -0.15 + 1.3 * torch.rand((count, 4), device=device, generator=generator)
        expected.rotor_parameters[name].copy_(value)
        actual.rotor_parameters[name].copy_(value)
    gain = expected.velocity_gain * (0.6 + 0.8 * torch.rand((count, 1), device=device, generator=generator))
    expected.velocity_gain.copy_(gain)
    actual.velocity_gain.copy_(gain)
    probe = _inputs(count, device, generator)
    before_probe = {name: getattr(actual, name).clone() for name in pointers}
    pure_commands = actual.motor_commands(*probe)
    wp.synchronize_device(device)
    torch.testing.assert_close(pure_commands, expected.motor_commands(*probe), rtol=2e-5, atol=2e-5)
    for name, unchanged in before_probe.items():
        torch.testing.assert_close(getattr(actual, name), unchanged, rtol=0, atol=0)

    for step in range(24):
        state, velocity, acceleration, yaw = _inputs(count, device, generator)
        acceleration_arg = acceleration if step % 4 in (1, 3) else None
        yaw_arg = yaw if step % 4 in (2, 3) else None
        if step in (5, 11):
            # Zero desired acceleration and a nearly parallel desired heading
            # exercise both original +1e-6 normalizers without NaNs.
            state.zero_()
            state[:, 6] = 1.0
            velocity.zero_()
            acceleration.zero_()
            acceleration[:, 2] = -GRAVITY
            if step == 11:
                acceleration[:, 0] = 1e-8
            yaw.zero_()
            acceleration_arg, yaw_arg = acceleration, yaw
        expected.compute(state, velocity, acceleration_arg, yaw_arg)
        actual.compute(state, velocity, acceleration_arg, yaw_arg)
        wp.synchronize_device(device)
        _assert_outputs(actual, expected)
        assert pointers == {name: getattr(actual, name).data_ptr() for name in pointers}
        if step in (8, 16):
            ids = torch.tensor([0, 3, count - 1], dtype=torch.int32, device=device)
            keep = torch.ones(count, dtype=torch.bool, device=device)
            keep[ids] = False
            before = {name: getattr(actual, name)[keep].clone() for name in pointers}
            expected.reset(ids)
            if step == 8:
                actual.reset(ids)
            else:
                mask = ~keep
                stream = wp.stream_from_torch(torch.cuda.current_stream(device)) if device.startswith("cuda") else None
                with wp.ScopedStream(stream) if stream is not None else wp.ScopedDevice(device):
                    actual.reset_mask(wp.from_torch(mask))
            wp.synchronize_device(device)
            _assert_outputs(actual, expected)
            for name, unchanged in before.items():
                torch.testing.assert_close(getattr(actual, name)[keep], unchanged, rtol=0, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_compute_preserves_the_original_zero_noise_rng_consumption(device):
    count = 19
    generator = torch.Generator(device=device).manual_seed(1807)
    args = _inputs(count, device, generator)
    expected = LeeRotorController(count, device)
    actual = WarpLeeRotorController(count, device)
    if device.startswith("cuda"):

        def get_state():
            return torch.cuda.get_rng_state(device)

        def set_state(value):
            torch.cuda.set_rng_state(value, device)
    else:
        get_state, set_state = torch.get_rng_state, torch.set_rng_state
    initial = get_state().clone()
    try:
        expected.compute(*args)
        reference_state = get_state().clone()
        set_state(initial)
        actual.compute(*args)
        assert torch.equal(get_state(), reference_state)
    finally:
        set_state(initial)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires CUDA")
def test_official_graph_cache_advances_the_motors_once_per_call():
    from isaaclab_experimental.utils.warp_graph_cache import WarpGraphCache

    device, count = "cuda:0", 7
    stream = wp.Stream(device)
    torch_stream = wp.stream_to_torch(stream)
    caller_stream = torch.cuda.current_stream(device)
    torch_stream.wait_stream(caller_stream)
    try:
        # WarpGraphCache requires a Warp-registered stream. Run Torch copies and
        # Warp captures on the same stream; only cross at this test's boundary.
        with torch.cuda.stream(torch_stream), wp.ScopedStream(stream):
            generator = torch.Generator(device=device).manual_seed(714)
            expected = LeeRotorController(count, device)
            actual = WarpLeeRotorController(count, device)
            cache = WarpGraphCache()
            for step in range(4):
                args = _inputs(count, device, generator)
                expected.compute(*args)
                actual._copy_inputs(*args)
                actual.advance_rng()
                cache.capture_or_replay("control", lambda: actual.launch(use_acceleration=True, use_yaw=True))
                wp.synchronize_device(device)
                _assert_outputs(actual, expected)
                if step == 1:
                    ids = torch.tensor([1, 5], dtype=torch.int32, device=device)
                    expected.reset(ids)
                    actual.reset(ids)
                    wp.synchronize_device(device)
                    _assert_outputs(actual, expected)
    finally:
        caller_stream.wait_stream(torch_stream)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Torch CUDA graph requires CUDA")
def test_diffnav_torch_graph_preserves_rng_state_and_motor_outputs():
    from aerialnav.diffnav.controller import GraphedLeeRotorController

    device, count = "cuda:0", 7
    generator = torch.Generator(device=device).manual_seed(719)
    args = _inputs(count, device, generator)
    expected = GraphedLeeRotorController(LeeRotorController(count, device), args)
    actual = WarpLeeRotorController(count, device)
    saved_rng = torch.cuda.get_rng_state(device).clone()
    try:
        for _ in range(3):
            args = _inputs(count, device, generator)
            before = torch.cuda.get_rng_state(device).clone()
            expected.compute(*args)
            reference_rng = torch.cuda.get_rng_state(device).clone()
            torch.cuda.set_rng_state(before, device)
            actual.compute(*args)
            assert torch.equal(torch.cuda.get_rng_state(device), reference_rng)
            wp.synchronize_device(device)
            _assert_outputs(actual, expected)
    finally:
        torch.cuda.set_rng_state(saved_rng, device)
