"""SDF escape gradients, immediate control/position Jacobian, and paired inputs."""

import json
import runpy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_zhang_contract import nominal_parameters

from isaaclab_navrl.reference.config import DcolReferenceTrainCfg
from isaaclab_navrl.zhang.config import ZhangTrainCfg
from isaaclab_navrl.zhang.dynamics import PointMassState, align_transition, point_mass_step, policy_transition
from isaaclab_navrl.zhang.losses import zhang_transition_costs
from isaaclab_navrl.zhang.runner import ZhangTrainer


@pytest.mark.parametrize("snapshots", (False, True))
def test_actual_sdf_collision_loss_escapes_centres_axes_corners_caps_and_deep_penetration(snapshots):
    positions = torch.tensor([[0.0, 0.0, 3.0], [10.0, 0.0, 3.0]], dtype=torch.double)
    sizes = torch.tensor([[2.0, 1.2, 1.0], [2.0, 2.0, 4.0]], dtype=torch.double)
    points = torch.tensor(
        [
            [0.0, 0.0, 3.0],
            [0.1, 0.1, 3.1],
            [1.05, 0.65, 3.55],
            [1.2, 0.0, 3.0],
            [10.0, 0.0, 3.0],
            [10.0, 0.0, 3.1],
            [10.1, 0.0, 3.1],
            [11.2, 0.0, 3.0],
            [10.0, 0.0, 5.1],
            [11.05, 0.0, 5.05],
        ],
        dtype=torch.double,
    )
    if snapshots:
        shift = torch.arange(len(points), dtype=points.dtype)[:, None] * points.new_tensor([0.03, 0.01, 0.0])
        positions = positions[None] + shift[:, None]
        points = points + shift
    points.requires_grad_()
    env = SimpleNamespace(
        terrain_distance=SimpleNamespace(signed_distance=lambda p: p[..., 0] * 0 + 100),
        motion=SimpleNamespace(position=positions, size=sizes),
    )
    trainer = SimpleNamespace(env=env)
    radius = points.new_full((len(points),), 0.3)
    gap = ZhangTrainer._collision_gap(trainer, points, radius)
    assert (gap < 0).all()
    escape = torch.autograd.grad(gap.sum(), points, retain_graph=True)[0]
    torch.testing.assert_close(escape.norm(dim=-1), torch.ones_like(gap))
    cfg = ZhangTrainCfg()
    zero = torch.zeros_like(points)
    history = {
        "velocity": [zero],
        "target_velocity": [zero],
        "episode_id": [torch.zeros(len(points), dtype=torch.long)],
        "velocity_prediction": [zero],
        "observed_velocity": [zero],
        "command": [zero],
        "previous_command": [zero],
        "position": [points],
        "height_range": [points.new_tensor([0.5, 2.5]).expand(len(points), 2)],
        "margin": [radius],
        "obstacle_signed_distance": [points.new_full((len(points), 10), 10.0)],
        "collision_gap": [gap[:, None].expand(-1, 9)],
    }
    losses, _, _ = zhang_transition_costs(history, cfg)
    descent = -torch.autograd.grad(losses["collision"], points)[0]
    assert torch.isfinite(descent).all() and ((descent * escape).sum(-1) > 0).all()
    improved = ZhangTrainer._collision_gap(trainer, points.detach() + 0.01 * escape, radius)
    assert (improved > gap.detach()).all()


@pytest.mark.parametrize("dt", (0.01, 0.05))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_trapezoid_matches_requested_local_control_jacobian_and_survives_measurement_alignment(dt, dtype):
    state = PointMassState(
        torch.tensor([[0.3, -0.2, 2.0]], dtype=dtype),
        torch.tensor([[0.2, 0.3, -0.1]], dtype=dtype),
        torch.zeros(1, 3, dtype=dtype),
        torch.zeros(1, 3, dtype=dtype),
    )
    parameters = {k: v.to(dtype) for k, v in nominal_parameters(1).items()}
    rotation = torch.eye(3, dtype=dtype)[None]
    command = torch.tensor([0.2, -0.3, 0.4], dtype=dtype, requires_grad=True)

    def transition(action, integration):
        return point_mass_step(state, action[None], rotation, parameters, dt, 0.4, position_integration=integration)

    original = transition(command, "original")
    trapezoid = transition(command, "trapezoid")
    torch.testing.assert_close(trapezoid.position, state.position + 0.5 * (state.velocity + trapezoid.velocity) * dt)
    jp = torch.autograd.functional.jacobian(lambda u: transition(u, "trapezoid").position[0], command)
    jv = torch.autograd.functional.jacobian(lambda u: transition(u, "trapezoid").velocity[0], command)
    old_jp = torch.autograd.functional.jacobian(lambda u: transition(u, "original").position[0], command)
    torch.testing.assert_close(jp, dt / 2 * jv)
    assert jp.norm() > 0 and torch.count_nonzero(old_jp) == 0
    measured = state.position + 1

    def aligned(action):
        prediction = transition(action, "trapezoid")
        terminal, _ = align_transition(
            prediction,
            measured,
            state.velocity,
            state.velocity,
            torch.tensor([True]),
            state.position,
            state.velocity,
            dt,
        )
        torch.testing.assert_close(terminal.position, measured)
        return terminal.position[0]

    torch.testing.assert_close(torch.autograd.functional.jacobian(aligned, command), jp)
    # Velocity integration remains exactly the previous implementation.
    torch.testing.assert_close(original.velocity, trapezoid.velocity, rtol=0, atol=0)
    cfg = ZhangTrainCfg(position_integration="trapezoid")
    cfg.physics_dt = cfg.policy_dt = dt
    torch.testing.assert_close(
        policy_transition(state, command[None], rotation, parameters, cfg).position, trapezoid.position
    )


def test_current_metric_is_sdf_and_historical_dcol_requires_its_explicit_reference():
    assert ZhangTrainCfg().collision_metric == "sdf"
    assert ZhangTrainCfg().loss_timing == "pre_reset"
    with pytest.raises(ValueError, match="SDF only"):
        ZhangTrainCfg(collision_metric="dcol").validate()
    DcolReferenceTrainCfg().validate()
    assert DcolReferenceTrainCfg().collision_metric == "dcol"
    with pytest.raises(ValueError, match="position_integration"):
        ZhangTrainCfg(position_integration="invalid").validate()


def test_four_cell_factorial_rejects_changed_initial_world(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    driver = runpy.run_path(str(root / "scripts/run_zhang_loss_timing.py"))
    plan = json.loads((root / "configs/experiments/zhang_collision_integration_512_20261031.json").read_text())
    assert {(r["collision_metric"], r["position_integration"]) for r in driver["matrix"](plan)} == {
        (m, i) for m in ("dcol", "sdf") for i in ("original", "trapezoid")
    }
    for run in plan["runs"]:
        directory = tmp_path / run["name"]
        directory.mkdir()
        config_type = DcolReferenceTrainCfg if run["collision_metric"] == "dcol" else ZhangTrainCfg
        cfg = config_type(position_integration=run["position_integration"], max_iterations=512)
        config = {
            "train": asdict(cfg),
            "environment": {"num_envs": 256, "height_bounds_reset": True},
            "parent_checkpoint": None,
        }
        (directory / "config.json").write_text(json.dumps(config))
        (directory / "initialization.json").write_text(
            json.dumps({"model_sha256": "same", "policy_class": "ZhangPolicy", "perception": "depth", "fresh": True})
        )
        (directory / "initial_rollout.json").write_text(json.dumps({"world_sha256": "same"}))
    assert driver["verify_factorial"](tmp_path, plan)["cells"] == 4
    (tmp_path / plan["runs"][-1]["name"] / "initial_rollout.json").write_text(json.dumps({"world_sha256": "different"}))
    with pytest.raises(ValueError, match="worlds"):
        driver["verify_factorial"](tmp_path, plan)
