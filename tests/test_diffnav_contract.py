"""Behavioral contracts for the differentiable training path (CPU only)."""

import importlib.util
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.diffnav.dynamics import (
    PointMassState,
    align_transition,
    disturbance_acceleration,
    point_mass_step,
    policy_transition,
)
from aerialnav.diffnav.events import randomize_diffnav_parameters, update_diffnav_disturbance
from aerialnav.diffnav.inference import load_diffnav_policy
from aerialnav.diffnav.losses import (
    diffnav_transition_costs,
    height_boundary_loss,
    velocity_step_loss,
    velocity_window_loss,
)
from aerialnav.diffnav.model import DiffNavPolicy, decode_action, policy_state
from aerialnav.diffnav.runner import DiffNavTrainer, cosine_learning_rate
from aerialnav.tasks.navigation.randomization import PolicyPerturbation, SimToRealCfg


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def nominal_parameters(batch):
    return {
        "max_speed": torch.full((batch, 1), 6.0),
        "margin": torch.full((batch,), 0.3),
        "thr_est_error": torch.ones(batch, 1),
        "pitch_ctl_delay": torch.full((batch, 1), 12.0),
        "yaw_ctl_delay": torch.full((batch, 1), 6.0),
        "drag_2": torch.zeros(batch, 2),
        "z_drag_coef": torch.ones(batch, 1),
        "wind": torch.zeros(batch, 3),
        "dg": torch.zeros(batch, 3),
    }


def loss_history(velocity, target=None, episode=None, done=None, collision=None, position=None, step_offset=0):
    """Construct measured transitions for loss contracts without a simulator."""
    count, batch, _ = velocity.shape
    target = torch.zeros_like(velocity) if target is None else target
    episode = torch.zeros(count, batch, dtype=torch.long) if episode is None else episode
    done = torch.zeros(count, batch, dtype=torch.bool) if done is None else done
    collision = torch.zeros_like(done) if collision is None else collision
    if position is None:
        position = torch.zeros_like(velocity)
        position[..., 2] = 2.0
    time = torch.arange(count)[:, None].expand(-1, batch)
    changed = torch.cat((torch.ones_like(episode[:1], dtype=torch.bool), episode[1:] != episode[:-1]), dim=0)
    starts = torch.where(changed, time, 0).cummax(0).values
    episode_step = time - starts + 1 + step_offset
    vectors = torch.zeros(batch, 10, 3)
    vectors[..., 0] = 10.0
    zeros = torch.zeros(batch, 3)
    return {
        "velocity": list(velocity.unbind(0)),
        "target_velocity": list(target.unbind(0)),
        "episode_id": list(episode.unbind(0)),
        "done": list(done.unbind(0)),
        "collision": list(collision.unbind(0)),
        "position": list(position.unbind(0)),
        "episode_step": list(episode_step.unbind(0)),
        **{key: list(torch.zeros_like(done).unbind(0)) for key in ("bounds", "timeout", "invalid")},
        "obstacle_signed_distance": [vectors.norm(dim=-1)] * count,
        "margin": [torch.full((batch,), 0.3)] * count,
        "height_target": [velocity.new_full((batch,), 2.0)] * count,
        "height_range": [velocity.new_full((batch, 2), 2.0)] * count,
        "command": [zeros] * count,
        "previous_command": [zeros] * count,
        "velocity_prediction": [zeros] * count,
        "observed_velocity": [zeros] * count,
    }


def test_step_velocity_uses_first_last_and_terminal_transitions():
    velocity = torch.tensor([[[2.0, 0, 0]], [[0, 3.0, 0]], [[0, 0, 4.0]]], requires_grad=True)
    target = torch.tensor([[[1.0, 0, 0]], [[0, 1.0, 0]], [[0, 0, 1.0]]])
    done = torch.tensor([[False], [False], [True]])
    history = loss_history(velocity, target, done=done, collision=done)
    losses, _, _ = diffnav_transition_costs(
        history, DiffNavTrainCfg(physics_dt=0.01, policy_dt=0.1, velocity_loss_mode="variant_default", variant="step")
    )
    torch.testing.assert_close(losses["velocity"], torch.tensor(1.5))
    torch.testing.assert_close(losses["velocity"], velocity_step_loss(velocity, target))
    assert losses["velocity_valid_samples"] == 3
    assert losses["velocity_sample_fraction"] == 1
    assert losses["velocity_short_terminal_samples"] == 0
    losses["velocity"].backward()
    torch.testing.assert_close(velocity.grad, torch.eye(3).reshape(3, 1, 3) / 3)


def test_height_boundary_is_zero_inside_c1_at_edges_and_points_inward():
    height = torch.tensor([0.4, 0.7, 2.0, 3.5, 3.8], dtype=torch.float64, requires_grad=True)
    position = torch.stack((torch.zeros_like(height), torch.zeros_like(height), height), dim=-1)
    loss = height_boundary_loss(position)
    expected = 2 * (0.5 * 0.6**2) / 5
    torch.testing.assert_close(loss, torch.tensor(expected, dtype=torch.float64))
    gradient = torch.autograd.grad(loss, height)[0]
    assert gradient[0] < 0 and gradient[-1] > 0
    torch.testing.assert_close(gradient[1:4], torch.zeros(3, dtype=torch.float64), atol=1e-14, rtol=0)
    interior = torch.tensor([[[0.0, 0.0, 0.7], [0.0, 0.0, 3.5]]], dtype=torch.float64, requires_grad=True)
    inside_loss = height_boundary_loss(interior)
    assert inside_loss == 0
    torch.testing.assert_close(torch.autograd.grad(inside_loss, interior)[0], torch.zeros_like(interior))
    nearby = torch.tensor([[[0.0, 0.0, 0.7 - 1e-6], [0.0, 0.0, 3.5 + 1e-6]]], dtype=torch.float64, requires_grad=True)
    nearby_gradient = torch.autograd.grad(height_boundary_loss(nearby), nearby)[0]
    assert nearby_gradient.abs().max() < 1e-5


def test_height_boundary_keeps_terminal_gradient_and_can_be_disabled():
    predicted_position = torch.tensor([[0.0, 0.0, 2.0]], requires_grad=True)
    zeros = torch.zeros(1, 3)
    prediction = PointMassState(predicted_position, zeros, zeros, zeros)
    terminal, reset = align_transition(
        prediction,
        torch.tensor([[0.0, 0.0, 3.8]]),
        zeros,
        zeros,
        torch.tensor([True]),
        torch.tensor([[0.0, 0.0, 2.0]]),
        zeros,
        0.1,
    )
    history = loss_history(terminal.velocity[None], position=terminal.position[None], done=torch.tensor([[True]]))
    active, _, _ = diffnav_transition_costs(
        history,
        DiffNavTrainCfg(
            physics_dt=0.01,
            policy_dt=0.1,
            variant="step",
            velocity_loss_mode="variant_default",
            height_loss_mode="boundary_huber",
        ),
    )
    disabled, _, _ = diffnav_transition_costs(
        history,
        DiffNavTrainCfg(
            physics_dt=0.01,
            policy_dt=0.1,
            variant="step",
            velocity_loss_mode="variant_default",
            coef_height_boundary=0,
            height_loss_mode="boundary_huber",
        ),
    )
    torch.testing.assert_close(active["total"] - disabled["total"], active["height_boundary"])
    gradient = torch.autograd.grad(active["height_boundary"], predicted_position, retain_graph=True)[0]
    assert gradient[0, 2] > 0
    torch.testing.assert_close(
        torch.autograd.grad(reset.position.sum(), predicted_position)[0], torch.zeros_like(predicted_position)
    )


def test_height_boundary_keeps_restoring_gradient_far_outside_and_is_c1_at_shoulders():
    height = torch.tensor([-4.3, -0.3, 4.5, 8.5], dtype=torch.float64, requires_grad=True)
    position = torch.stack((height * 0, height * 0, height), dim=-1)
    value = height_boundary_loss(position, reduction="none")
    torch.testing.assert_close(value, torch.tensor([9.5, 1.5, 1.5, 9.5], dtype=torch.float64))
    gradient = torch.autograd.grad(value.mean(), height)[0]
    torch.testing.assert_close(gradient, torch.tensor([-0.5, -0.5, 0.5, 0.5], dtype=torch.float64))

    # The Huber shoulder lies one scale outside each activation edge.
    epsilon = 1e-7
    shoulder = torch.tensor(
        [0.2 - epsilon, 0.2 + epsilon, 4.0 - epsilon, 4.0 + epsilon], dtype=torch.float64, requires_grad=True
    )
    shoulder_position = torch.stack((shoulder * 0, shoulder * 0, shoulder), dim=-1)
    derivative = torch.autograd.grad(height_boundary_loss(shoulder_position, reduction="sum"), shoulder)[0]
    torch.testing.assert_close(derivative, torch.tensor([-2.0, -2.0, 2.0, 2.0], dtype=torch.float64), rtol=0, atol=5e-7)


def test_transition_costs_keep_all_seven_axis_reductions_and_transition_times():
    dtype = torch.float64
    velocity = torch.zeros(3, 2, 3, dtype=dtype)
    velocity[..., 0] = torch.tensor([[2.0, 1.0], [3.0, 0.0], [4.0, 5.0]], dtype=dtype)
    velocity.requires_grad_()
    history = loss_history(velocity, done=torch.tensor([[False, False], [True, False], [False, True]]))
    clearance = torch.tensor([[0.5, 1.2], [0.0, 0.8], [-0.1, 2.0]], dtype=dtype)
    vectors = torch.zeros(3, 2, 10, 3, dtype=dtype)
    vectors[..., 0] = (clearance + 0.3)[..., None]
    commands = torch.zeros_like(velocity)
    commands[..., 0] = torch.tensor([[0.1, 0.2], [0.4, 0.5], [0.4, 0.9]], dtype=dtype)
    previous = torch.zeros_like(velocity)
    previous[..., 0] = torch.tensor([[0.0, 0.0], [0.1, 0.2], [0.4, 0.5]], dtype=dtype)
    prediction = torch.zeros_like(velocity)
    prediction[..., 0] = torch.tensor([[0.3, 0.6], [0.9, 1.2], [1.5, 1.8]], dtype=dtype)
    position = torch.zeros_like(velocity)
    position[..., 2] = torch.tensor([[2.0, 3.8], [0.4, 2.0], [5.0, -1.0]], dtype=dtype)
    history.update(
        obstacle_signed_distance=list(vectors.norm(dim=-1).unbind(0)),
        margin=[torch.full((2,), 0.3, dtype=dtype)] * 3,
        command=list(commands.unbind(0)),
        previous_command=list(previous.unbind(0)),
        velocity_prediction=list(prediction.unbind(0)),
        observed_velocity=list(torch.zeros_like(velocity).unbind(0)),
        position=list(position.unbind(0)),
    )
    cfg = DiffNavTrainCfg(
        physics_dt=0.01,
        policy_dt=0.1,
        velocity_loss_mode="variant_default",
        variant="step",
        height_loss_mode="boundary_huber",
        coef_v=0.7,
        coef_v_pred=0.4,
        coef_obj_avoidance=0.3,
        coef_collide=0.2,
        coef_d_acc=0.1,
        coef_d_jerk=0.05,
        coef_height_boundary=0.6,
    )
    expected_terms = {
        "velocity": torch.tensor([[1.5, 0.5], [2.5, 0.0], [3.5, 4.5]], dtype=dtype),
        "velocity_prediction": prediction[..., 0].square() / 3,
        "obstacle_avoidance": torch.tensor([[0.25, 0.0], [1.0, 0.04], [1.21, 0.0]], dtype=dtype),
        "collision": torch.tensor(
            [[math.log1p(math.exp(-32 * float(value))) for value in row] for row in clearance], dtype=dtype
        ),
        "acceleration": commands[..., 0].square(),
        "jerk": torch.tensor([[1.0, 4.0], [9.0, 9.0], [0.0, 16.0]], dtype=dtype),
        "height_boundary": torch.tensor([[0.0, 0.18], [0.18, 0.0], [2.5, 2.9]], dtype=dtype),
    }
    coefficients = (0.7, 0.4, 0.3, 0.2, 0.1, 0.05, 0.6)
    expected = sum(coef * value for coef, value in zip(coefficients, expected_terms.values(), strict=True))
    losses, cache, cost = diffnav_transition_costs(history, cfg)
    assert cache is None and cost.shape == (3, 2)
    torch.testing.assert_close(cost, expected)
    for name, value in expected_terms.items():
        torch.testing.assert_close(losses[name], value.mean())
    torch.testing.assert_close(losses["total"], cost.mean(), rtol=0, atol=0)
    # A local cost depends on its own velocity, not a broadcast rollout scalar.
    gradient = torch.autograd.grad(cost[0, 0], velocity)[0]
    expected_gradient = torch.zeros_like(velocity)
    expected_gradient[0, 0, 0] = 0.7
    torch.testing.assert_close(gradient, expected_gradient)


def test_window_transition_costs_score_only_current_endpoints_and_detach_prefix():
    dtype = torch.float64
    previous = torch.zeros(2, 2, 3, dtype=dtype)
    previous[..., 0] = torch.tensor([[10.0, 20.0], [12.0, 22.0]], dtype=dtype)
    previous.requires_grad_()
    previous_target = torch.zeros_like(previous)
    previous_target[..., 0] = torch.tensor([[3.0, 5.0], [5.0, 7.0]], dtype=dtype)
    previous_target.requires_grad_()
    current = torch.zeros(4, 2, 3, dtype=dtype)
    current[..., 0] = torch.tensor([[2.0, 100.0], [4.0, 200.0], [6.0, 300.0], [8.0, 400.0]], dtype=dtype)
    current.requires_grad_()
    target = torch.zeros_like(current)
    target[..., 0] = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]], dtype=dtype)
    episode = torch.tensor([[0, 1], [0, 1], [1, 1], [1, 1]])
    prefix = {
        "velocity": previous,
        "target_velocity": previous_target,
        "episode_id": torch.zeros(2, 2, dtype=torch.long),
    }
    cfg = DiffNavTrainCfg(
        physics_dt=0.01,
        policy_dt=0.1,
        variant="continuous",
        velocity_loss_mode="full_window",
        height_loss_mode="boundary_huber",
        velocity_window_s=0.2,
        coef_v=1.0,
        coef_v_pred=0.0,
        coef_obj_avoidance=0.0,
        coef_collide=0.0,
        coef_d_acc=0.0,
        coef_d_jerk=0.0,
        coef_height_boundary=0.0,
    )
    losses, cache, cost = diffnav_transition_costs(loss_history(current, target, episode), cfg, prefix)
    expected = torch.tensor([[2.0, 0.0], [2.0, 186.0], [0.0, 306.0], [14 / 3, 426.0]], dtype=dtype)
    torch.testing.assert_close(cost, expected)
    assert losses["velocity_valid_samples"] == 6 and losses["velocity_sample_fraction"] == 0.75
    original = velocity_window_loss(
        torch.cat((previous.detach(), current)),
        torch.cat((previous_target.detach(), target)),
        torch.cat((prefix["episode_id"], episode)),
        2,
        current_start=2,
    )
    torch.testing.assert_close(losses["velocity"], original)
    torch.testing.assert_close(losses["total"], expected.mean())
    torch.testing.assert_close(cache["velocity"], current[-2:].detach())
    assert all(not value.requires_grad for value in cache.values())
    old_gradients = torch.autograd.grad(cost.sum(), (previous, previous_target), retain_graph=True, allow_unused=True)
    assert old_gradients == (None, None)
    invalid_gradient = torch.autograd.grad(cost[2, 0], current, retain_graph=True)[0]
    torch.testing.assert_close(invalid_gradient, torch.zeros_like(current))
    gradient = torch.autograd.grad(cost[3, 0], current)[0]
    expected_gradient = torch.zeros_like(current)
    expected_gradient[2:, 0, 0] = 2 / 3
    torch.testing.assert_close(gradient, expected_gradient)


def test_transition_costs_without_a_full_window_still_keep_nonvelocity_cost():
    velocity = torch.zeros(2, 1, 3, requires_grad=True)
    position = torch.tensor([[[0.0, 0.0, 2.0]], [[0.0, 0.0, 4.0]]], requires_grad=True)
    cfg = DiffNavTrainCfg(
        physics_dt=0.01,
        policy_dt=0.1,
        variant="continuous",
        velocity_loss_mode="full_window",
        height_loss_mode="boundary_huber",
        velocity_window_s=0.3,
        coef_v_pred=0.0,
        coef_obj_avoidance=0.0,
        coef_collide=0.0,
        coef_d_acc=0.0,
        coef_d_jerk=0.0,
    )
    losses, cache, cost = diffnav_transition_costs(loss_history(velocity, position=position), cfg)
    torch.testing.assert_close(cost, torch.tensor([[0.0], [0.5]]))
    assert losses["velocity_valid_samples"] == 0 and losses["velocity"] == 0
    assert cache["velocity"].shape == (2, 1, 3)
    velocity_gradient, position_gradient = torch.autograd.grad(cost.sum(), (velocity, position))
    torch.testing.assert_close(velocity_gradient, torch.zeros_like(velocity))
    assert position_gradient[1, 0, 2] > 0


def test_dcol_changes_only_collision_gap_and_preserves_approach_weight():
    velocity = torch.zeros(3, 1, 3)
    history = loss_history(velocity)
    vectors = torch.zeros(1, 10, 3)
    vectors[0, :, 0] = torch.linspace(2.0, 1.1, 10)
    history["obstacle_signed_distance"] = [vectors.norm(dim=-1)] * 3
    gap = torch.full((3, 1, 9), 0.02, requires_grad=True)
    history["dcol_gap"] = list(gap.unbind(0))
    original, _, _ = diffnav_transition_costs(
        history,
        DiffNavTrainCfg(
            physics_dt=0.01,
            policy_dt=0.1,
            variant="continuous",
            velocity_loss_mode="full_window",
            height_loss_mode="boundary_huber",
            velocity_window_s=0.2,
        ),
    )
    dcol, _, _ = diffnav_transition_costs(
        history,
        DiffNavTrainCfg(
            physics_dt=0.01,
            policy_dt=0.1,
            variant="dcol",
            velocity_loss_mode="full_window",
            height_loss_mode="boundary_huber",
            velocity_window_s=0.2,
        ),
    )
    torch.testing.assert_close(dcol["collision"], (9 * F.softplus(-32 * gap)).mean())
    for name in ("velocity", "velocity_prediction", "obstacle_avoidance", "acceleration", "jerk", "height_boundary"):
        torch.testing.assert_close(dcol[name], original[name])
    dcol["collision"].backward()
    assert bool((gap.grad < 0).all())


def test_variant_and_height_boundary_configuration():
    cfg = DiffNavTrainCfg(physics_dt=0.01, policy_dt=0.1, variant="step", velocity_loss_mode="variant_default")
    assert cfg.variant == "step" and cfg.velocity_window_steps == 20
    assert cfg.height_bounds == (0.2, 4.0)
    DiffNavTrainCfg(
        physics_dt=0.01, policy_dt=0.1, variant="step", velocity_loss_mode="variant_default", coef_height_boundary=0
    ).validate()
    for options in (
        {"variant": "unknown"},
        {"boundary_scale": 0},
        {"boundary_margin": 2},
        {"coef_height_boundary": -1},
        {"height_bounds": (4.0, 0.2)},
    ):
        with pytest.raises(ValueError):
            DiffNavTrainCfg(**options).validate()


def test_original_network_matches_published_model():
    reference_root = os.environ.get("DIFFNAV_REFERENCE_ROOT")
    if not reference_root:
        pytest.skip("Set DIFFNAV_REFERENCE_ROOT to a pinned upstream checkout")
    source = Path(reference_root) / "model.py"
    if not source.exists():
        pytest.skip("Original read-only source checkout is not mounted")
    spec = importlib.util.spec_from_file_location("upstream_policy_reference_model", source)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    reference = reference_module.Model(10, 6)
    migrated = DiffNavPolicy()
    migrated.load_state_dict(reference.state_dict(), strict=True)
    depth, state, hidden = torch.randn(3, 1, 12, 16), torch.randn(3, 10), torch.randn(3, 192)
    expected, _, expected_hidden = reference(depth, state, hidden)
    actual, actual_hidden = migrated(depth, state, hidden)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)


@pytest.mark.parametrize("recurrent_model", ("gru", "sru_lstm_gated"))
def test_activation_checkpointing_preserves_recurrent_outputs_and_gradients(recurrent_model):
    import copy

    torch.manual_seed(29)
    original = DiffNavPolicy(recurrent_model)
    checkpointed = copy.deepcopy(original)
    depth = torch.randn(4, 1, 12, 16)
    state = torch.randn(4, 10)
    outputs, gradients, saved_counts = [], [], []
    for model, forward in ((original, original.forward), (checkpointed, checkpointed.forward_checkpointed)):
        state_input = state.clone().requires_grad_()
        hidden = torch.zeros(4, model.recurrent_state_size)
        saved = []

        def pack(tensor, saved=saved):
            saved.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            sequence = []
            for _ in range(3):
                action, hidden = forward(depth, state_input, hidden)
                sequence.append(action)
            loss = torch.stack(sequence).square().sum() + hidden.square().mean()
        saved_counts.append(sum(saved))
        loss.backward()
        outputs.append(torch.stack(sequence).detach())
        gradients.append([state_input.grad, *[p.grad for p in model.parameters()]])
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for reference, actual in zip(gradients[0], gradients[1], strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert saved_counts[1] < saved_counts[0] / 2


def test_interleaved_decode_and_ten_state_features_have_no_axis_speed_clip():
    quaternion = torch.tensor([[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]])
    state, heading, target, _ = policy_state(
        torch.zeros(1, 3),
        torch.tensor([[1.0, 2.0, 3.0]]),
        quaternion,
        torch.tensor([[0.0, 20.0, 0.0]]),
        torch.tensor([[9.0]]),
        torch.tensor([0.3]),
    )
    torch.testing.assert_close(
        state, torch.tensor([[2.0, -1.0, 3.0, 9.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3]]), atol=2e-6, rtol=1e-6
    )
    torch.testing.assert_close(target, torch.tensor([[0.0, 9.0, 0.0]]))
    command, predicted_velocity = decode_action(
        torch.tensor([[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]]), heading, torch.ones(1, 1)
    )
    torch.testing.assert_close(command, torch.tensor([[18.0, -9.0, -27.0]]), atol=4e-6, rtol=1e-6)
    torch.testing.assert_close(predicted_velocity, torch.tensor([[-20.0, 10.0, 30.0]]), atol=4e-6, rtol=1e-6)
    zero, _ = decode_action(torch.zeros(1, 6), heading, torch.ones(1, 1))
    torch.testing.assert_close(zero, torch.zeros(1, 3))


def test_point_mass_adjoint_decays_per_second_and_preserves_command_gradient():
    position = torch.zeros(1, 3, requires_grad=True)
    velocity = torch.zeros(1, 3, requires_grad=True)
    command = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)
    state = PointMassState(position, velocity, torch.zeros(1, 3), torch.zeros(1, 3))
    predicted = point_mass_step(state, command, torch.eye(3)[None], nominal_parameters(1), 0.01, 0.4)
    expected_actuator = 1 - math.exp(-0.12)
    torch.testing.assert_close(predicted.velocity, torch.tensor([[0.0, 0.0, 0.005 * expected_actuator]]))
    p_gradient, v_gradient = torch.autograd.grad(predicted.position.sum(), (position, velocity), retain_graph=True)
    torch.testing.assert_close(p_gradient, torch.full((1, 3), 0.4**0.01))
    torch.testing.assert_close(v_gradient, torch.full((1, 3), 0.01))
    action_gradient = torch.autograd.grad(predicted.velocity.sum(), command)[0]
    torch.testing.assert_close(action_gradient, torch.full((1, 3), 0.005 * expected_actuator))


def test_state_alignment_keeps_terminal_gradients_and_cuts_reset_history():
    predicted_position = torch.randn(2, 3, requires_grad=True)
    predicted_velocity = torch.randn(2, 3, requires_grad=True)
    predicted = PointMassState(predicted_position, predicted_velocity, torch.zeros(2, 3), torch.zeros(2, 3))
    measured_position, measured_velocity = torch.full((2, 3), 5.0), torch.full((2, 3), 2.0)
    terminal, next_state = align_transition(
        predicted,
        measured_position,
        measured_velocity,
        torch.zeros(2, 3),
        torch.tensor([True, False]),
        torch.full((2, 3), -5.0),
        torch.zeros(2, 3),
        0.1,
    )
    torch.testing.assert_close(terminal.position, measured_position)
    torch.testing.assert_close(next_state.position, torch.tensor([[-5.0] * 3, [5.0] * 3]))
    terminal_gradient = torch.autograd.grad(terminal.position.sum(), predicted_position, retain_graph=True)[0]
    next_gradient = torch.autograd.grad(next_state.position.sum(), predicted_position)[0]
    torch.testing.assert_close(terminal_gradient, torch.ones(2, 3))
    torch.testing.assert_close(next_gradient, torch.tensor([[0.0] * 3, [1.0] * 3]))


def test_collision_substep_trace_matches_the_model_and_keeps_only_its_causal_gradient():
    command = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    zero = torch.zeros(1, 3)
    start = PointMassState(zero, zero, zero, zero)
    parameters, rotation, cfg = (
        nominal_parameters(1),
        torch.eye(3)[None],
        DiffNavTrainCfg(physics_dt=0.01, policy_dt=0.1, variant="step", velocity_loss_mode="variant_default"),
    )
    final, positions = policy_transition(start, command, rotation, parameters, cfg, record_positions=True)
    expected = policy_transition(start, command, rotation, parameters, cfg)
    assert positions.shape == (1, 10, 3)
    torch.testing.assert_close(final.position, expected.position, rtol=0, atol=0)
    first_gradient = torch.autograd.grad(positions[:, 0].sum(), command, retain_graph=True, allow_unused=True)[0]
    assert first_gradient is None or bool((first_gradient == 0).all())
    early_gradient = torch.autograd.grad(positions[:, 2].sum(), command, retain_graph=True)[0]
    last_gradient = torch.autograd.grad(positions[:, -1].sum(), command)[0]
    assert bool((early_gradient > 0).all()) and bool((last_gradient > early_gradient).all())


def test_drag_matches_source_body_axis_formula():
    parameters = nominal_parameters(1)
    parameters["drag_2"][:] = torch.tensor([0.2, 0.3])
    parameters["wind"][:] = torch.tensor([0.1, 0.2, 0.3])
    parameters["dg"][:] = torch.tensor([0.01, 0.02, 0.03])
    parameters["z_drag_coef"][:] = 0.8
    relative = torch.tensor([[0.9, -2.2, 2.7]])
    expected = parameters["dg"] - (0.2 * relative * relative.abs() + 0.3 * relative) * torch.tensor([1, 1, 0.8])
    actual = disturbance_acceleration(
        torch.tensor([[1.0, -2.0, 3.0]]), torch.tensor([[0.0, 0.0, 0.0, 1.0]]), parameters
    )
    torch.testing.assert_close(actual, expected)


def test_velocity_loss_matches_source_forward_window_and_rejects_cross_episode():
    torch.manual_seed(7)
    velocity, target = torch.randn(90, 2, 3), torch.randn(90, 2, 3)
    cumulative = velocity.cumsum(0)
    average = (cumulative[60:] - cumulative[:-60]) / 60
    distance = (average - target[1 : 1 - 60]).norm(dim=-1)
    expected = F.smooth_l1_loss(distance, torch.zeros_like(distance))
    actual = velocity_window_loss(velocity, target, torch.zeros(90, 2, dtype=torch.long), 60)
    torch.testing.assert_close(actual, expected)
    short_velocity = torch.randn(5, 1, 3, requires_grad=True)
    mixed = velocity_window_loss(
        short_velocity, torch.zeros_like(short_velocity), torch.tensor([[0], [0], [0], [1], [1]]), 3
    )
    assert mixed == 0
    mixed.backward()
    torch.testing.assert_close(short_velocity.grad, torch.zeros_like(short_velocity))


def test_avoidance_speed_and_jerk_use_policy_time_units():
    cfg = DiffNavTrainCfg(
        physics_dt=0.01, policy_dt=0.1, variant="step", velocity_loss_mode="variant_default", velocity_window_s=0.2
    )
    vectors = torch.zeros(1, 10, 3)
    vectors[0, :, 0] = torch.linspace(1.2, 0.3, 10)
    commands = [torch.tensor([[0.1 * i, 0.0, 0.0]]) for i in range(4)]
    history = {
        "velocity": [torch.zeros(1, 3)] * 4,
        "position": [torch.tensor([[0.0, 0.0, 2.0]])] * 4,
        "done": [torch.zeros(1, dtype=torch.bool)] * 4,
        "collision": [torch.zeros(1, dtype=torch.bool)] * 4,
        "target_velocity": [torch.zeros(1, 3)] * 4,
        "episode_id": [torch.zeros(1, dtype=torch.long)] * 4,
        "obstacle_signed_distance": [vectors.norm(dim=-1)] * 4,
        "height_target": [torch.full((1,), 2.0)] * 4,
        "height_range": [torch.full((1, 2), 2.0)] * 4,
        "margin": [torch.full((1,), 0.3)] * 4,
        "command": commands,
        "previous_command": [commands[0], *commands[:-1]],
        "velocity_prediction": [torch.zeros(1, 3)] * 4,
        "observed_velocity": [torch.zeros(1, 3)] * 4,
    }
    losses, _, _ = diffnav_transition_costs(history, cfg)
    clearance = vectors.norm(dim=-1) - 0.3
    expected_speed = torch.full((1, 9), 9.0)
    torch.testing.assert_close(
        losses["obstacle_avoidance"], (expected_speed * (1 - clearance[:, 1:]).relu().square()).mean()
    )
    torch.testing.assert_close(losses["collision"], (expected_speed * F.softplus(-32 * clearance[:, 1:])).mean())
    torch.testing.assert_close(losses["jerk"], torch.tensor(0.75))
    assert set(losses) == {
        "velocity",
        "velocity_prediction",
        "obstacle_avoidance",
        "collision",
        "acceleration",
        "jerk",
        "height_boundary",
        "total",
        "velocity_valid_samples",
        "velocity_sample_fraction",
        "velocity_short_terminal_samples",
    }


def test_pre_step_lookahead_preserves_gradient_for_a_crossed_obstacle():
    velocity = torch.tensor([[40.0, 0.0, 0.0]], requires_grad=True)
    start = torch.tensor([[0.0, 0.1, 2.0]])
    times = torch.linspace(0, 0.1, 10)[None, :, None]
    obstacle = torch.tensor([[[2.2, 0.0, 2.0]]])
    pre_vectors = obstacle - (start[:, None] + velocity[:, None] * times)
    post_vectors = obstacle - (start[:, None] + velocity[:, None] * (times + 0.1))
    base = {
        "velocity": [torch.zeros(1, 3)],
        "target_velocity": [torch.zeros(1, 3)],
        "position": [torch.tensor([[0.0, 0.0, 2.0]])],
        "height_target": [torch.full((1,), 2.0)],
        "height_range": [torch.full((1, 2), 2.0)],
        "done": [torch.zeros(1, dtype=torch.bool)],
        "collision": [torch.zeros(1, dtype=torch.bool)],
        "episode_id": [torch.zeros(1, dtype=torch.long)],
        "margin": [torch.tensor([0.3])],
        "command": [torch.zeros(1, 3)],
        "previous_command": [torch.zeros(1, 3)],
        "velocity_prediction": [torch.zeros(1, 3)],
        "observed_velocity": [torch.zeros(1, 3)],
    }
    pre_loss, _, _ = diffnav_transition_costs(
        dict(base, obstacle_signed_distance=[pre_vectors.norm(dim=-1)]),
        DiffNavTrainCfg(physics_dt=0.01, policy_dt=0.1, variant="step", velocity_loss_mode="variant_default"),
    )
    post_loss, _, _ = diffnav_transition_costs(
        dict(base, obstacle_signed_distance=[post_vectors.norm(dim=-1)]),
        DiffNavTrainCfg(physics_dt=0.01, policy_dt=0.1, variant="step", velocity_loss_mode="variant_default"),
    )
    assert pre_loss["obstacle_avoidance"] > 0
    assert post_loss["obstacle_avoidance"] == 0
    pre_loss["total"].backward()
    assert velocity.grad[0, 1] < 0  # Gradient descent steers farther from the obstacle in +Y.


def test_unlimited_training_and_budget_independent_cosine_period():
    cfg = DiffNavTrainCfg(physics_dt=0.01, policy_dt=0.1, variant="step", velocity_loss_mode="variant_default")
    cfg.validate()
    assert cfg.max_iterations is None
    assert cfg.cosine_period_updates == 400
    assert cfg.velocity_window_steps == 20
    assert cfg.timesteps * cfg.policy_dt == 15
    for update in (0, 400, 800, 1200, 1600, 2000, 40000):
        assert cosine_learning_rate(update, cfg) == pytest.approx(0.001)
        assert cosine_learning_rate(
            update,
            DiffNavTrainCfg(
                physics_dt=0.01, policy_dt=0.1, variant="step", velocity_loss_mode="variant_default", max_iterations=2
            ),
        ) == pytest.approx(0.001)
    assert cosine_learning_rate(399, cfg) < 0.000011
    with pytest.raises(ValueError, match="policy steps"):
        DiffNavTrainCfg(
            physics_dt=0.01, policy_dt=0.1, variant="step", velocity_loss_mode="variant_default", velocity_window_s=0.15
        ).validate()


class AnalyticEnv:
    """Independent forward environment exercising the real trainer's reset contract."""

    num_envs = 2
    device = "cpu"
    physics_dt = 0.01
    step_dt = 0.1

    def __init__(self):
        self.cfg = SimpleNamespace(
            seed=7,
            terrain_seed=0,
            num_static_obstacles=350,
            num_dynamic_obstacles=100,
            evaluation=False,
            training_stage=None,
            sim_to_real=SimToRealCfg(),
            variant="step",
            height_target_range=(0.5, 2.5),
        )
        self.perturbation = PolicyPerturbation(
            self.cfg.sim_to_real, 2, "cpu", 7, policy_dt=self.step_dt, nominal_action_delay=1
        )
        self.diffnav_parameters = nominal_parameters(2)
        self.position = torch.zeros(2, 3)
        self.velocity = torch.zeros(2, 3)
        self.episode_id = torch.zeros(2, dtype=torch.long)
        self.depth = torch.ones(2, 1, 12, 16)
        self.total_steps = 0
        self.perturbation.reset(torch.arange(2))
        self.queried_at_steps = []
        self.motion = SimpleNamespace(position=torch.empty(0, 3), velocity=torch.empty(0, 3), size=torch.empty(0, 3))
        self.terrain_distance = SimpleNamespace(signed_distance=self.nearest_obstacle_signed_distance)
        self.nonfinite_events = torch.zeros((), dtype=torch.long)
        self.curriculum = None

    def reset(self):
        self.perturbation.reset(torch.arange(2))
        self.position.zero_()
        self.position[:, 2] = 2
        self.velocity.zero_()
        self.episode_id += 1
        self.total_steps = 0

    def get_diffnav_observation(self):
        observation = {
            "position": self.position.clone(),
            "velocity": self.velocity.clone(),
            "quaternion": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, -1),
            "target": torch.tensor([[10.0, 0.0, 2.0]]).expand(2, -1),
            "episode_id": self.episode_id,
            "depth_features": self.depth,
            "height_target": torch.full((2,), 2.0),
            "height_range": torch.full((2, 2), 2.0),
        }
        observation.update(
            {
                f"policy_{name}": observation[name]
                for name in ("position", "velocity", "quaternion", "target", "depth_features")
            }
        )
        observation["input_delay_steps"] = torch.zeros(2, dtype=torch.long)
        observation["depth_age_s"] = torch.zeros(2)
        observation["depth_known_fraction"] = torch.ones(2)
        return observation

    def nearest_obstacle_signed_distance(self, query):
        self.queried_at_steps.append(self.total_steps)
        torch.testing.assert_close(query[:, 0], self.position)
        surface = torch.zeros_like(query)
        surface[:, :, 0] = 12.0
        surface[:, :, 2] = 2.0
        return (surface - query).norm(dim=-1)

    def step(self, command):
        command = self.perturbation.execute(command)
        self.total_steps += 1
        self.velocity += command * self.step_dt
        self.position += self.velocity * self.step_dt
        transition = {
            "position": self.position.clone(),
            "velocity": self.velocity.clone(),
            "executed_action": command.clone(),
            "actuator": command.clone(),
            "bounds": torch.zeros(2, dtype=torch.bool),
        }
        done = torch.tensor([self.total_steps % 2 == 0, False])
        transition["collision"] = done.clone()
        if done[0]:
            self.position[0] = torch.tensor([0.0, 0.0, 2.0])
            self.velocity[0] = 0
            self.episode_id[0] += 1
            # Explicitly exercise mutation of an EventManager-owned parameter buffer.
            self.diffnav_parameters["pitch_ctl_delay"][0] = 11.0 + self.total_steps / 100
            self.perturbation.reset(torch.tensor([0]))
        self.depth.fill_(1 + self.total_steps / 100)
        return {}, torch.zeros(2), done, torch.zeros(2, dtype=torch.bool), {"transition": transition}


@pytest.mark.parametrize("recurrent_model", ("gru", "sru_lstm_gated"))
def test_two_bptt_updates_survive_resets_and_save_independent_inference(tmp_path, recurrent_model):
    torch.manual_seed(3)
    env = AnalyticEnv()
    cfg = DiffNavTrainCfg(
        recurrent_model=recurrent_model,
        physics_dt=0.01,
        policy_dt=0.1,
        variant="step",
        velocity_loss_mode="variant_default",
        timesteps=4,
        max_iterations=2,
        velocity_window_s=0.2,
    )
    trainer = DiffNavTrainer(env, cfg, tmp_path)
    before = trainer.model.fc.weight.detach().clone()
    result = trainer.run()
    assert result["completed_updates"] == 2 and result["nonfinite_grad_events"] == 0
    assert env.queried_at_steps == list(range(8))
    assert not torch.equal(before, trainer.model.fc.weight)
    checkpoint = tmp_path / "checkpoint_final.pt"
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["train_cfg"] == asdict(cfg)
    assert json.loads((tmp_path / "training_result.json").read_text())["fresh"] is True
    env.reset()
    policy = load_diffnav_policy(env, checkpoint)
    torch.testing.assert_close(env.step(policy())[-1]["transition"]["executed_action"], torch.zeros(2, 3))
    assert torch.isfinite(policy()).all()
    env.episode_id[0] += 1
    after_reset = policy()
    assert torch.isfinite(after_reset).all()
    # Old weights still support clean evaluation, but must not silently change
    # delayed-input semantics when training or evaluating an old experiment.
    del saved["environment"]["observation_contract"]
    legacy = tmp_path / "legacy.pt"
    torch.save(saved, legacy)
    assert torch.isfinite(load_diffnav_policy(env, legacy)()).all()
    with pytest.raises(ValueError, match="Observation timing changed"):
        trainer.load_stage_checkpoint(legacy)
    env.cfg.sim_to_real.enabled = True
    with pytest.raises(ValueError, match="Observation timing changed"):
        load_diffnav_policy(env, legacy)


@pytest.mark.parametrize("recurrent_model", ("gru", "sru_lstm_gated"))
def test_stage_handoff_preserves_weights_optimizer_and_counts_actual_transitions(tmp_path, recurrent_model):
    checkpoint = None
    previous_weights = None
    expected_steps = 0
    for index, stage in enumerate(("pretrain_s0", "pretrain_s1", "pretrain_s2", "finetune_s1")):
        env = AnalyticEnv()
        env.cfg.training_stage = stage
        env.cfg.sim_to_real.enabled = stage.startswith("finetune")
        cfg = DiffNavTrainCfg(
            recurrent_model=recurrent_model,
            physics_dt=0.01,
            policy_dt=0.1,
            variant="step",
            velocity_loss_mode="variant_default",
            timesteps=4,
            max_iterations=1,
            velocity_window_s=0.2,
        )
        trainer = DiffNavTrainer(env, cfg, tmp_path / stage)
        if checkpoint is not None:
            trainer.load_stage_checkpoint(checkpoint)
            for name, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, previous_weights[name], rtol=0, atol=0)
            assert all(int(value["step"]) == index for value in trainer.optimizer.state.values())
            assert trainer.schedule_updates == (0 if stage == "finetune_s1" else index)
        result = trainer.run()
        expected_steps += cfg.timesteps * env.num_envs
        assert result["environment_steps"] == expected_steps
        assert result["completed_updates"] == index + 1
        assert result["stage_updates"] == 1
        assert result["fresh"] is (index == 0)
        assert all(int(value["step"]) == index + 1 for value in trainer.optimizer.state.values())
        checkpoint = tmp_path / stage / "checkpoint_final.pt"
        previous_weights = {name: value.clone() for name, value in trainer.model.state_dict().items()}


def test_stage_handoff_rejects_changed_rollout_length(tmp_path):
    first_env = AnalyticEnv()
    first_env.cfg.training_stage = "pretrain_s0"
    first = DiffNavTrainer(
        first_env,
        DiffNavTrainCfg(
            physics_dt=0.01,
            policy_dt=0.1,
            variant="step",
            velocity_loss_mode="variant_default",
            timesteps=4,
            max_iterations=1,
            velocity_window_s=0.2,
        ),
        tmp_path / "first",
    )
    first.run()
    second_env = AnalyticEnv()
    second_env.cfg.training_stage = "pretrain_s1"
    second = DiffNavTrainer(
        second_env,
        DiffNavTrainCfg(
            physics_dt=0.01,
            policy_dt=0.1,
            variant="step",
            velocity_loss_mode="variant_default",
            timesteps=5,
            max_iterations=1,
            velocity_window_s=0.2,
        ),
        tmp_path / "second",
    )
    with pytest.raises(ValueError, match="Stage transitions cannot change timesteps"):
        second.load_stage_checkpoint(tmp_path / "first" / "checkpoint_final.pt")


def test_stage_handoff_rejects_another_comparison_variant(tmp_path):
    env = AnalyticEnv()
    env.cfg.training_stage = "pretrain_s1"
    first = DiffNavTrainer(
        env,
        DiffNavTrainCfg(
            physics_dt=0.01,
            policy_dt=0.1,
            variant="step",
            velocity_loss_mode="variant_default",
            timesteps=2,
            max_iterations=1,
        ),
        tmp_path / "first",
    )
    first.run()
    env.cfg.training_stage = "pretrain_s2"
    env.cfg.variant = "continuous"
    second = DiffNavTrainer(
        env,
        DiffNavTrainCfg(
            physics_dt=0.01,
            policy_dt=0.1,
            velocity_loss_mode="variant_default",
            variant="continuous",
            timesteps=2,
            max_iterations=1,
        ),
        tmp_path / "second",
    )
    with pytest.raises(ValueError, match="variant"):
        second.load_stage_checkpoint(tmp_path / "first" / "checkpoint_final.pt")


def test_bptt_curriculum_resets_temporal_history_without_recreating_optimizer(tmp_path):
    from aerialnav.tasks.navigation.curriculum import TrainingCurriculum

    class CurriculumEnv(AnalyticEnv):
        def __init__(self):
            super().__init__()
            self.curriculum = TrainingCurriculum(self.num_envs, self.device)
            self.cfg.num_dynamic_obstacles = 60

        def step(self, command):
            result = super().step(command)
            self.curriculum.observe(
                {
                    "first_event_code": result[2].long(),
                    "first_event_time_s": torch.full((2,), self.total_steps * self.step_dt),
                    "terminated": result[2],
                    "timeout": result[3],
                    "episode_return": torch.zeros(2),
                    "step": torch.full((2,), self.total_steps),
                }
            )
            return result

        def advance_curriculum(self):
            self.cfg.num_dynamic_obstacles = self.curriculum.advance()
            self.reset()

    env = CurriculumEnv()
    cfg = DiffNavTrainCfg(
        physics_dt=0.01,
        policy_dt=0.1,
        variant="step",
        velocity_loss_mode="variant_default",
        timesteps=4,
        max_iterations=3,
        velocity_window_s=0.2,
    )
    trainer = DiffNavTrainer(env, cfg, tmp_path)
    optimizer = trainer.optimizer
    result = trainer.run()
    assert result["completed_updates"] == 3
    assert env.cfg.num_dynamic_obstacles == 120
    assert trainer.velocity_history is None
    assert env.queried_at_steps == list(range(4)) * 3
    assert trainer.optimizer is optimizer
    assert all(int(value["step"]) == 3 for value in optimizer.state.values())
    assert (tmp_path / "checkpoint_best_100.pt").exists()
    assert result["acceptance_passed"] is False  # Two synthetic environments are not a 1024-env acceptance run.


def test_event_randomization_uses_geometry_margin_and_nominal_evaluation():
    class EventEnv:
        num_envs = 16
        device = "cpu"
        cfg = SimpleNamespace(seed=17, evaluation=False, training_stage=None)
        mass_scale = torch.ones(16, 1)
        inertia_scale = torch.ones(16, 1)
        perturbation = SimpleNamespace(thrust_scale=torch.ones(16, 1), control_gain_scale=torch.ones(16, 1))

        def apply_diffnav_camera_randomization(self, env_ids):
            self.camera_updated = env_ids.clone()

    env = EventEnv()
    randomize_diffnav_parameters(env, None, body_radius=0.3)
    parameters = env.diffnav_parameters
    assert bool(((parameters["max_speed"] >= 2.25) & (parameters["max_speed"] <= 9.75)).all())
    assert bool(((parameters["margin"] >= 0.3) & (parameters["margin"] <= 0.5)).all())
    untouched = {key: value[1:].clone() for key, value in parameters.items()}
    randomize_diffnav_parameters(env, torch.tensor([0]), body_radius=0.3)
    for key, value in untouched.items():
        torch.testing.assert_close(parameters[key][1:], value)
    env.cfg.evaluation = True
    randomize_diffnav_parameters(env, None, body_radius=0.3)
    update_diffnav_disturbance(env, None)
    for key in ("wind", "dg", "drag_2", "camera_pitch"):
        torch.testing.assert_close(parameters[key], torch.zeros_like(parameters[key]))
    torch.testing.assert_close(parameters["margin"], torch.full((16,), 0.3))


def test_short_collision_window_crosses_update_with_detached_history():
    cfg = DiffNavTrainCfg(
        physics_dt=0.01, policy_dt=0.1, variant="step", velocity_loss_mode="truncated_window", velocity_window_s=0.4
    )
    previous = torch.tensor([[[1.0, 0, 0]], [[3.0, 0, 0]]], requires_grad=True)
    first, cache, _ = diffnav_transition_costs(loss_history(previous), cfg)
    assert first["velocity_valid_samples"] == 0
    current = torch.tensor([[[5.0, 0, 0]]], requires_grad=True)
    terminal = torch.ones(1, 1, dtype=torch.bool)
    losses, _, _ = diffnav_transition_costs(
        loss_history(current, done=terminal, collision=terminal, step_offset=2), cfg, cache
    )
    torch.testing.assert_close(losses["velocity"], torch.tensor(2.5))
    assert losses["velocity_valid_samples"] == 1
    assert losses["velocity_short_terminal_samples"] == 1
    losses["velocity"].backward()
    assert previous.grad is None
    torch.testing.assert_close(current.grad, torch.tensor([[[1 / 3, 0, 0]]]))


def test_exact_first_window_collision_is_scored_once_without_duplicate_later_windows():
    cfg = DiffNavTrainCfg(
        physics_dt=0.01, policy_dt=0.1, variant="step", velocity_loss_mode="truncated_window", velocity_window_s=0.3
    )
    velocity = torch.zeros(3, 1, 3, requires_grad=True)
    with torch.no_grad():
        velocity[:, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    done = torch.tensor([[False], [False], [True]])
    original = velocity_window_loss(velocity, torch.zeros_like(velocity), torch.zeros(3, 1, dtype=torch.long), 3)
    assert original == 0
    first, _, _ = diffnav_transition_costs(loss_history(velocity, done=done, collision=done), cfg)
    torch.testing.assert_close(first["velocity"], torch.tensor(1.5))
    assert first["velocity_valid_samples"] == 1
    assert first["velocity_short_terminal_samples"] == 1
    first["velocity"].backward()
    expected_gradient = torch.zeros_like(velocity)
    expected_gradient[:, :, 0] = 1 / 3
    torch.testing.assert_close(velocity.grad, expected_gradient)
    later_velocity = torch.cat((torch.tensor([[[9.0, 0, 0]]]), velocity.detach()), dim=0)
    later_episode = torch.tensor([[0], [1], [1], [1]])
    later_done = torch.tensor([[True], [False], [False], [True]])
    later_collision = torch.tensor([[False], [False], [False], [True]])
    later, _, _ = diffnav_transition_costs(
        loss_history(later_velocity, episode=later_episode, done=later_done, collision=later_collision), cfg
    )
    torch.testing.assert_close(later["velocity"], torch.tensor(1.5))
    assert later["velocity_valid_samples"] == 1
    assert later["velocity_short_terminal_samples"] == 0
