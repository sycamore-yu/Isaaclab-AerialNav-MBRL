"""Behavioral contracts for the differentiable training path (CPU only)."""

import importlib.util
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from isaaclab_navrl.tasks.navigation.randomization import PolicyPerturbation, SimToRealCfg
from isaaclab_navrl.zhang.config import ZhangTrainCfg
from isaaclab_navrl.zhang.dynamics import (
    PointMassState,
    align_transition,
    disturbance_acceleration,
    point_mass_step,
)
from isaaclab_navrl.zhang.events import randomize_zhang_parameters, update_zhang_disturbance
from isaaclab_navrl.zhang.inference import load_zhang_policy
from isaaclab_navrl.zhang.losses import velocity_window_loss, zhang_losses
from isaaclab_navrl.zhang.model import ZhangPolicy, decode_action, policy_state
from isaaclab_navrl.zhang.runner import ZhangTrainer, cosine_learning_rate


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def nominal_parameters(batch):
    return {
        "max_speed": torch.full((batch, 1), 6.0), "margin": torch.full((batch,), 0.3),
        "thr_est_error": torch.ones(batch, 1), "pitch_ctl_delay": torch.full((batch, 1), 12.0),
        "yaw_ctl_delay": torch.full((batch, 1), 6.0), "drag_2": torch.zeros(batch, 2),
        "z_drag_coef": torch.ones(batch, 1), "wind": torch.zeros(batch, 3), "dg": torch.zeros(batch, 3),
    }


def test_original_network_matches_published_model():
    source = Path("/home/tong/tongworkspace/reference_repos/DiffPhysDrone/model.py")
    if not source.exists():
        pytest.skip("Original read-only source checkout is not mounted")
    spec = importlib.util.spec_from_file_location("diffphysdrone_reference_model", source)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    reference = reference_module.Model(10, 6)
    migrated = ZhangPolicy()
    migrated.load_state_dict(reference.state_dict(), strict=True)
    depth, state, hidden = torch.randn(3, 1, 12, 16), torch.randn(3, 10), torch.randn(3, 192)
    expected, _, expected_hidden = reference(depth, state, hidden)
    actual, actual_hidden = migrated(depth, state, hidden)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)


def test_activation_checkpointing_preserves_recurrent_outputs_and_gradients():
    import copy

    torch.manual_seed(29)
    original = ZhangPolicy()
    checkpointed = copy.deepcopy(original)
    depth = torch.randn(4, 1, 12, 16)
    state = torch.randn(4, 10)
    outputs, gradients, saved_counts = [], [], []
    for model, forward in ((original, original.forward), (checkpointed, checkpointed.forward_checkpointed)):
        state_input = state.clone().requires_grad_()
        hidden = torch.zeros(4, 192)
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
        torch.zeros(1, 3), torch.tensor([[1.0, 2.0, 3.0]]), quaternion,
        torch.tensor([[0.0, 20.0, 0.0]]), torch.tensor([[9.0]]), torch.tensor([0.3]),
    )
    torch.testing.assert_close(state, torch.tensor([[2.0, -1.0, 3.0, 9.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3]]), atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(target, torch.tensor([[0.0, 9.0, 0.0]]))
    command, predicted_velocity = decode_action(torch.tensor([[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]]), heading, torch.ones(1, 1))
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
    torch.testing.assert_close(p_gradient, torch.full((1, 3), 0.4 ** 0.01))
    torch.testing.assert_close(v_gradient, torch.full((1, 3), 0.01))
    action_gradient = torch.autograd.grad(predicted.velocity.sum(), command)[0]
    torch.testing.assert_close(action_gradient, torch.full((1, 3), 0.005 * expected_actuator))


def test_state_alignment_keeps_terminal_gradients_and_cuts_reset_history():
    predicted_position = torch.randn(2, 3, requires_grad=True)
    predicted_velocity = torch.randn(2, 3, requires_grad=True)
    predicted = PointMassState(predicted_position, predicted_velocity, torch.zeros(2, 3), torch.zeros(2, 3))
    measured_position, measured_velocity = torch.full((2, 3), 5.0), torch.full((2, 3), 2.0)
    terminal, next_state = align_transition(predicted, measured_position, measured_velocity, torch.zeros(2, 3),
        torch.tensor([True, False]), torch.full((2, 3), -5.0), torch.zeros(2, 3), 0.1)
    torch.testing.assert_close(terminal.position, measured_position)
    torch.testing.assert_close(next_state.position, torch.tensor([[-5.0] * 3, [5.0] * 3]))
    terminal_gradient = torch.autograd.grad(terminal.position.sum(), predicted_position, retain_graph=True)[0]
    next_gradient = torch.autograd.grad(next_state.position.sum(), predicted_position)[0]
    torch.testing.assert_close(terminal_gradient, torch.ones(2, 3))
    torch.testing.assert_close(next_gradient, torch.tensor([[0.0] * 3, [1.0] * 3]))


def test_drag_matches_source_body_axis_formula():
    parameters = nominal_parameters(1)
    parameters["drag_2"][:] = torch.tensor([0.2, 0.3])
    parameters["wind"][:] = torch.tensor([0.1, 0.2, 0.3])
    parameters["dg"][:] = torch.tensor([0.01, 0.02, 0.03])
    parameters["z_drag_coef"][:] = 0.8
    relative = torch.tensor([[0.9, -2.2, 2.7]])
    expected = parameters["dg"] - (0.2 * relative * relative.abs() + 0.3 * relative) * torch.tensor([1, 1, 0.8])
    actual = disturbance_acceleration(torch.tensor([[1.0, -2.0, 3.0]]), torch.tensor([[0.0, 0.0, 0.0, 1.0]]), parameters)
    torch.testing.assert_close(actual, expected)


def test_velocity_loss_matches_source_forward_window_and_rejects_cross_episode():
    torch.manual_seed(7)
    velocity, target = torch.randn(90, 2, 3), torch.randn(90, 2, 3)
    cumulative = velocity.cumsum(0)
    average = (cumulative[60:] - cumulative[:-60]) / 60
    distance = (average - target[1:1 - 60]).norm(dim=-1)
    expected = F.smooth_l1_loss(distance, torch.zeros_like(distance))
    actual = velocity_window_loss(velocity, target, torch.zeros(90, 2, dtype=torch.long), 60)
    torch.testing.assert_close(actual, expected)
    short_velocity = torch.randn(5, 1, 3, requires_grad=True)
    mixed = velocity_window_loss(short_velocity, torch.zeros_like(short_velocity), torch.tensor([[0], [0], [0], [1], [1]]), 3)
    assert mixed == 0
    mixed.backward()
    torch.testing.assert_close(short_velocity.grad, torch.zeros_like(short_velocity))


def test_avoidance_speed_and_jerk_use_policy_time_units():
    cfg = ZhangTrainCfg(velocity_window_s=0.2)
    vectors = torch.zeros(1, 10, 3)
    vectors[0, :, 0] = torch.linspace(1.2, 0.3, 10)
    commands = [torch.tensor([[0.1 * i, 0.0, 0.0]]) for i in range(4)]
    history = {
        "velocity": [torch.zeros(1, 3)] * 4,
        "target_velocity": [torch.zeros(1, 3)] * 4,
        "episode_id": [torch.zeros(1, dtype=torch.long)] * 4,
        "obstacle_vector": [vectors] * 4, "margin": [torch.full((1,), 0.3)] * 4,
        "command": commands, "previous_command": [commands[0], *commands[:-1]],
        "velocity_prediction": [torch.zeros(1, 3)] * 4,
        "observed_velocity": [torch.zeros(1, 3)] * 4,
    }
    losses, _ = zhang_losses(history, cfg)
    clearance = vectors.norm(dim=-1) - 0.3
    expected_speed = torch.full((1, 9), 9.0)
    torch.testing.assert_close(losses["obstacle_avoidance"], (expected_speed * (1 - clearance[:, 1:]).relu().square()).mean())
    torch.testing.assert_close(losses["collision"], (expected_speed * F.softplus(-32 * clearance[:, 1:])).mean())
    torch.testing.assert_close(losses["jerk"], torch.tensor(0.75))
    assert set(losses) == {"velocity", "velocity_prediction", "obstacle_avoidance", "collision", "acceleration", "jerk", "total"}


def test_pre_step_lookahead_preserves_gradient_for_a_crossed_obstacle():
    velocity = torch.tensor([[40.0, 0.0, 0.0]], requires_grad=True)
    start = torch.tensor([[0.0, 0.1, 2.0]])
    times = torch.linspace(0, 0.1, 10)[None, :, None]
    obstacle = torch.tensor([[[2.2, 0.0, 2.0]]])
    pre_vectors = obstacle - (start[:, None] + velocity[:, None] * times)
    post_vectors = obstacle - (start[:, None] + velocity[:, None] * (times + 0.1))
    base = {
        "velocity": [torch.zeros(1, 3)], "target_velocity": [torch.zeros(1, 3)],
        "episode_id": [torch.zeros(1, dtype=torch.long)], "margin": [torch.tensor([0.3])],
        "command": [torch.zeros(1, 3)], "previous_command": [torch.zeros(1, 3)],
        "velocity_prediction": [torch.zeros(1, 3)], "observed_velocity": [torch.zeros(1, 3)],
    }
    pre_loss, _ = zhang_losses(dict(base, obstacle_vector=[pre_vectors]), ZhangTrainCfg())
    post_loss, _ = zhang_losses(dict(base, obstacle_vector=[post_vectors]), ZhangTrainCfg())
    assert pre_loss["obstacle_avoidance"] > 0
    assert post_loss["obstacle_avoidance"] == 0
    pre_loss["total"].backward()
    assert velocity.grad[0, 1] < 0  # Gradient descent steers farther from the obstacle in +Y.


def test_unlimited_training_and_budget_independent_cosine_period():
    cfg = ZhangTrainCfg()
    cfg.validate()
    assert cfg.max_iterations is None
    assert cfg.cosine_period_updates == 400
    assert cfg.velocity_window_steps == 60
    assert cfg.timesteps * cfg.policy_dt == 15
    for update in (0, 400, 800, 1200, 1600, 2000, 40000):
        assert cosine_learning_rate(update, cfg) == pytest.approx(0.001)
        assert cosine_learning_rate(update, ZhangTrainCfg(max_iterations=2)) == pytest.approx(0.001)
    assert cosine_learning_rate(399, cfg) < 0.000011
    with pytest.raises(ValueError, match="policy steps"):
        ZhangTrainCfg(velocity_window_s=0.15).validate()


class AnalyticEnv:
    """Independent forward environment exercising the real trainer's reset contract."""
    num_envs = 2
    device = "cpu"
    physics_dt = 0.01
    step_dt = 0.1

    def __init__(self):
        self.cfg = SimpleNamespace(seed=7, terrain_seed=0, num_static_obstacles=350,
                                   num_dynamic_obstacles=100, evaluation=False, training_stage=None,
                                   sim_to_real=SimToRealCfg())
        self.perturbation = PolicyPerturbation(self.cfg.sim_to_real, 2, "cpu", 7, nominal_action_delay=1)
        self.zhang_parameters = nominal_parameters(2)
        self.position = torch.zeros(2, 3)
        self.velocity = torch.zeros(2, 3)
        self.episode_id = torch.zeros(2, dtype=torch.long)
        self.depth = torch.ones(2, 1, 12, 16)
        self.total_steps = 0
        self.perturbation.reset(torch.arange(2))
        self.queried_at_steps = []
        self.nonfinite_events = torch.zeros((), dtype=torch.long)
        self.curriculum = None

    def reset(self):
        self.perturbation.reset(torch.arange(2))
        self.position.zero_()
        self.position[:, 2] = 2
        self.velocity.zero_()
        self.episode_id += 1
        self.total_steps = 0

    def get_zhang_observation(self):
        observation = {"position": self.position.clone(), "velocity": self.velocity.clone(),
                "quaternion": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, -1),
                "target": torch.tensor([[10.0, 0.0, 2.0]]).expand(2, -1),
                "episode_id": self.episode_id, "depth_features": self.depth}
        observation.update({f"policy_{name}": observation[name] for name in (
            "position", "velocity", "quaternion", "target", "depth_features"
        )})
        observation["input_delay_steps"] = torch.zeros(2, dtype=torch.long)
        return observation

    def nearest_obstacle_vector(self, query):
        self.queried_at_steps.append(self.total_steps)
        torch.testing.assert_close(query[:, 0], self.position)
        surface = torch.zeros_like(query)
        surface[:, :, 0] = 12.0
        surface[:, :, 2] = 2.0
        return surface - query

    def step(self, command):
        command = self.perturbation.execute(command)
        self.total_steps += 1
        self.velocity += command * self.step_dt
        self.position += self.velocity * self.step_dt
        transition = {"position": self.position.clone(), "velocity": self.velocity.clone(),
                      "executed_action": command.clone(), "actuator": command.clone()}
        done = torch.tensor([self.total_steps % 2 == 0, False])
        if done[0]:
            self.position[0] = torch.tensor([0.0, 0.0, 2.0])
            self.velocity[0] = 0
            self.episode_id[0] += 1
            # Explicitly exercise mutation of an EventManager-owned parameter buffer.
            self.zhang_parameters["pitch_ctl_delay"][0] = 11.0 + self.total_steps / 100
            self.perturbation.reset(torch.tensor([0]))
        self.depth.fill_(1 + self.total_steps / 100)
        return {}, torch.zeros(2), done, torch.zeros(2, dtype=torch.bool), {"transition": transition}


def test_two_bptt_updates_survive_resets_and_save_independent_inference(tmp_path):
    torch.manual_seed(3)
    env = AnalyticEnv()
    cfg = ZhangTrainCfg(timesteps=4, max_iterations=2, velocity_window_s=0.2)
    trainer = ZhangTrainer(env, cfg, tmp_path)
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
    policy = load_zhang_policy(env, checkpoint)
    torch.testing.assert_close(env.step(policy())[-1]["transition"]["executed_action"], torch.zeros(2, 3))
    assert torch.isfinite(policy()).all()
    env.episode_id[0] += 1
    after_reset = policy()
    assert torch.isfinite(after_reset).all()


def test_stage_handoff_preserves_weights_optimizer_and_counts_actual_transitions(tmp_path):
    checkpoint = None
    previous_weights = None
    expected_steps = 0
    for index, stage in enumerate(("pretrain_s1", "pretrain_s2", "finetune_s1")):
        env = AnalyticEnv()
        env.cfg.training_stage = stage
        env.cfg.sim_to_real.enabled = stage.startswith("finetune")
        cfg = ZhangTrainCfg(timesteps=4 + index, max_iterations=1, velocity_window_s=0.2)
        trainer = ZhangTrainer(env, cfg, tmp_path / stage)
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


def test_bptt_curriculum_resets_temporal_history_without_recreating_optimizer(tmp_path):
    from isaaclab_navrl.tasks.navigation.curriculum import TrainingCurriculum

    class CurriculumEnv(AnalyticEnv):
        def __init__(self):
            super().__init__()
            self.curriculum = TrainingCurriculum(self.num_envs, self.device)
            self.cfg.num_dynamic_obstacles = 60

        def step(self, command):
            result = super().step(command)
            self.curriculum.observe({
                "first_event_code": result[2].long(),
                "first_event_time_s": torch.full((2,), self.total_steps * self.step_dt),
                "terminated": result[2], "timeout": result[3],
                "episode_return": torch.zeros(2), "step": torch.full((2,), self.total_steps),
            })
            return result

        def advance_curriculum(self):
            self.cfg.num_dynamic_obstacles = self.curriculum.advance()
            self.reset()

    env = CurriculumEnv()
    cfg = ZhangTrainCfg(timesteps=4, max_iterations=3, velocity_window_s=0.2)
    trainer = ZhangTrainer(env, cfg, tmp_path)
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

        def apply_zhang_camera_randomization(self, env_ids):
            self.camera_updated = env_ids.clone()

    env = EventEnv()
    randomize_zhang_parameters(env, None, body_radius=0.3)
    parameters = env.zhang_parameters
    assert bool(((parameters["max_speed"] >= 2.25) & (parameters["max_speed"] <= 9.75)).all())
    assert bool(((parameters["margin"] >= 0.3) & (parameters["margin"] <= 0.5)).all())
    untouched = {key: value[1:].clone() for key, value in parameters.items()}
    randomize_zhang_parameters(env, torch.tensor([0]), body_radius=0.3)
    for key, value in untouched.items():
        torch.testing.assert_close(parameters[key][1:], value)
    env.cfg.evaluation = True
    randomize_zhang_parameters(env, None, body_radius=0.3)
    update_zhang_disturbance(env, None)
    for key in ("wind", "dg", "drag_2", "camera_pitch"):
        torch.testing.assert_close(parameters[key], torch.zeros_like(parameters[key]))
    torch.testing.assert_close(parameters["margin"], torch.full((16,), 0.3))
