"""Moving geometry, horizontal steering, episode EMA and fixed-scene logging."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_training_curriculum import transition
from test_zhang_shac_endpoints import EndpointEnv

from isaaclab_navrl.tasks.navigation.curriculum import TrainingCurriculum
from isaaclab_navrl.tasks.navigation.geometry import dynamic_signed_distance, nearest_obstacle_signed_distance
from isaaclab_navrl.training import record_curriculum, tensorboard_directory, write_training_scalars
from isaaclab_navrl.zhang.config import ZhangTrainCfg
from isaaclab_navrl.zhang.losses import VelocityEMA, relative_avoidance, risk_speed_scale
from isaaclab_navrl.zhang.runner import ZhangTrainer


def test_future_pose_corrects_wrong_side_and_matches_independent_queries():
    times = torch.linspace(0, 0.05, 10, dtype=torch.double)
    query = torch.stack((times, times * 0 + 0.002, times * 0 + 2.5), -1).requires_grad_()
    position = torch.tensor([[0.08, 0, 2.5], [2.0, 1.0, 2.5]], dtype=torch.double)
    velocity = torch.tensor([[-1.5, 0, 0], [0, 1.0, 0]], dtype=torch.double)
    sizes = torch.tensor([[1.0, 1.0, 5.0], [1.0, 1.0, 5.0]], dtype=torch.double)
    poses = position + velocity * times[:, None, None]
    aligned = dynamic_signed_distance(query, poses, sizes)
    stale = dynamic_signed_distance(query, position, sizes)
    expected = torch.stack([dynamic_signed_distance(q, p, sizes) for q, p in zip(query, poses, strict=True)])
    torch.testing.assert_close(aligned, expected)
    grad = torch.autograd.grad(aligned.sum(), query, retain_graph=True)[0]
    stale_grad = torch.autograd.grad(stale.sum(), query)[0]
    assert grad[-1, 0] > 0 and stale_grad[-1, 0] < 0
    assert float((poses[-1] - position).norm(dim=-1).max()) == pytest.approx(0.075)


def test_selected_velocity_is_zero_for_static_and_ground():
    query = torch.tensor([[0.8, 0.0, 2.5], [0.8, 0.0, 2.5], [10.0, 0.0, 0.2]])
    position = torch.tensor([[0.0, 0.0, 2.5]])
    sizes = torch.tensor([[1.0, 1.0, 5.0]])
    velocity = torch.tensor([[1.5, 0.0, 0.0]])
    distance, selected = nearest_obstacle_signed_distance(
        query, torch.tensor([10.0, 0.1, 10.0]), position, sizes, dynamic_velocity=velocity
    )
    torch.testing.assert_close(distance, torch.tensor([0.3, 0.1, 0.19]))
    torch.testing.assert_close(selected, torch.cat((velocity, torch.zeros(2, 3))))
    torch.testing.assert_close(
        dynamic_signed_distance(query, position, sizes),
        dynamic_signed_distance(query, position.expand(3, -1, -1), sizes),
    )


def test_lateral_reward_reduces_approach_without_rewarding_ascent():
    velocity = torch.tensor([[[2.0, 1.0, 4.0]]], requires_grad=True)
    normal = torch.tensor([[[-1.0, 0.0, 0.0]]])
    closing, loss = relative_avoidance(velocity, normal, torch.zeros(1, 1))
    assert closing.item() == 2
    gradient = torch.autograd.grad(loss.sum(), velocity)[0]
    assert gradient[0, 0, 0] > 0 and gradient[0, 0, 1] < 0
    assert gradient[0, 0, 2] == 0
    _, departing = relative_avoidance(-velocity, normal, torch.zeros(1, 1))
    assert departing.item() == 0
    _, ground = relative_avoidance(velocity, torch.tensor([[[0.0, 0.0, -1.0]]]), torch.zeros(1, 1))
    assert ground.item() == 4  # No reward for horizontal motion along a height boundary.


def test_risk_target_is_bounded_detached_and_distinguishes_receding():
    closing = torch.tensor([[4.0], [-4.0]], requires_grad=True)
    scale = risk_speed_scale(closing, torch.zeros(2, 1), torch.ones(2, 1) * 4, 0.25)
    torch.testing.assert_close(scale, torch.tensor([[0.25], [1.0]]))
    assert not scale.requires_grad
    assert risk_speed_scale(closing, torch.ones(2, 1) * 10, torch.ones(2, 1) * 4, 0.25).min() > 0.999


def test_ema_reset_detach_formula_and_independent_random_stream():
    rng = torch.get_rng_state().clone()
    ema = VelocityEMA(torch.zeros(2, 3), seed=7)
    assert torch.equal(rng, torch.get_rng_state())
    assert ((ema.alpha >= 0.08) & (ema.alpha <= 0.12)).all()
    alpha = ema.alpha.clone()
    first = torch.ones(2, 3, requires_grad=True)
    ema.update(first)
    torch.testing.assert_close(ema.value, alpha.expand(2, 3))
    ema.reset(torch.tensor([True, False]))
    assert (ema.value[0] == 0).all() and ema.alpha[1] == alpha[1]
    second = torch.full((2, 3), 2.0, requires_grad=True)
    value = ema.update(second)
    torch.testing.assert_close(value[0], (2 * ema.alpha[0]).expand(3))
    grad = torch.autograd.grad(value.sum(), first)[0]
    assert (grad[0] == 0).all() and (grad[1] > 0).all()
    ema.detach()
    assert not ema.value.requires_grad


@pytest.mark.parametrize(
    "changes",
    [
        {"velocity_loss_mode": "ema"},
        {"velocity_loss_mode": "ema", "algorithm": "shac"},
        {"approach_mode": "relative", "lateral_avoidance": 1.0, "risk_speed_floor": 0.25},
        {"collision_metric": "sdf"},
    ],
)
def test_real_runner_new_modes_keep_terminal_poses_and_reset_graphs(tmp_path, changes):
    env = EndpointEnv(trigger_step=1)
    env.motion = SimpleNamespace(
        position=torch.tensor([[2.0, 0, 2.0]]), velocity=torch.tensor([[-1.0, 0, 0]]), size=torch.ones(1, 3)
    )
    original_step = env.step

    def step(command):
        result = original_step(command)
        times = torch.arange(1, 11)[:, None, None] * env.physics_dt
        poses = env.motion.position + env.motion.velocity * times
        result[-1]["transition"]["dynamic_positions_by_substep"] = poses
        env.motion.position = poses[-1].clone()
        return result

    env.step = step
    cfg = ZhangTrainCfg(
        physics_dt=0.01,
        policy_dt=0.1,
        timesteps=3,
        max_iterations=2,
        critic_iterations=1,
        critic_max_iterations=1,
        loss_timing="deferred",  # This fixture injects terminal poses after step returns.
        **changes,
    )
    trainer = ZhangTrainer(env, cfg, tmp_path)
    original_gap = trainer._dcol_gap
    samples = []

    def gap(query, radius, positions, **kwargs):
        samples.append(positions.clone())
        return original_gap(query, radius, positions, **kwargs)

    trainer._dcol_gap = gap
    result = trainer.run()
    assert result["completed_updates"] == 2 and result["nonfinite_grad_events"] == 0
    torch.testing.assert_close(samples[0][0, :, 0, 0], 2.0 - torch.linspace(0, 0.1, 10)[1:-1])
    torch.testing.assert_close(samples[1][:, 0, 0], torch.full((2,), 1.9))


def test_stage_independent_tags_and_shared_event_directory(tmp_path, monkeypatch):
    tags = []
    writer = SimpleNamespace(add_scalar=lambda tag, value, update: tags.append(tag), flush=lambda: None)
    curriculum = TrainingCurriculum(2, "cpu", stage="finetune_s1")
    curriculum.observe(transition([1, 2], [True, True]))
    record_curriculum(curriculum, 1, tmp_path, writer, lambda path: None)
    assert "Train/success_rate" in tags and "Train/episodes" in tags
    assert all("finetune" not in tag and "pretrain" not in tag for tag in tags)
    write_training_scalars(writer, {"total": 1, "Optim/head/vx/update_norm": 3, "update_s": 2}, 1)
    assert "Loss/total" in tags and "Timing/update_s" in tags
    assert not any("head" in tag for tag in tags)
    monkeypatch.setenv("NAVRL_TENSORBOARD_DIR", str(tmp_path / "shared"))
    assert tensorboard_directory(tmp_path / "finetune_s1") == tensorboard_directory(tmp_path / "finetune_s2")


def test_six_comparisons_fix_finetune_s1_and_preserve_retention(tmp_path):
    root = Path(__file__).resolve().parents[1]
    queue = runpy.run_path(str(root / "scripts/run_comparison_queue.py"))
    protocol = runpy.run_path(str(root / "scripts/run_acceptance.py"))
    plan = json.loads((root / "configs/experiments/zhang_finetune_s1.json").read_text())
    assert len(plan["runs"]) == 6
    assert protocol["selected_stages"]("finetune_s1", screening=True, fixed_stage=True) == ("finetune_s1",)
    with pytest.raises(ValueError, match="screening"):
        protocol["selected_stages"](fixed_stage=True)
    for run in plan["runs"]:
        command = queue["command_for"](root, tmp_path, plan, run, "cuda:0")
        assert command[command.index("--start_stage") + 1] == "finetune_s1" and "--fixed_stage" in command
        values = protocol["parse_train_args"](queue["train_tokens"](run))
        for name in ("timing",):
            values.pop(name)
        ZhangTrainCfg(**values).validate()
        command = protocol["build_command"](
            method="zhang",
            stage="finetune_s1",
            log_dir=tmp_path,
            scripts_dir=root / "scripts",
            num_envs=256,
            seed=0,
            device="cuda:0",
            finetune_lr=5e-5,
            train_args={},
            parent_checkpoint=None,
            fresh_stage=True,
            fixed_stage=True,
        )
        assert "--stop_after_acceptance" not in command and "--fresh_stage" in command
