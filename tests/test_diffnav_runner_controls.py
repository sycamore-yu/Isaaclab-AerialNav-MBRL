"""Restore learned state without rewinding the experiment's observed history."""

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.diffnav.runner import DiffNavTrainer
from aerialnav.tasks.navigation.curriculum import TrainingCurriculum


@dataclass
class _Randomization:
    enabled: bool = False


class _Writer:
    def add_scalar(self, *args):
        pass


def _trainer(path, schedule="cosine", stage="pretrain_s0", algorithm="bptt", **cfg_changes):
    settings = dict(
        physics_dt=0.01,
        policy_dt=0.1,
        velocity_loss_mode="variant_default",
        variant="dcol",
        algorithm=algorithm,
        observation_normalization=True,
        lr_schedule=schedule,
        plateau_patience=2,
        plateau_cooldown=0,
    )
    settings.update(cfg_changes)
    cfg = DiffNavTrainCfg(**settings)
    path.mkdir(parents=True, exist_ok=True)
    env = SimpleNamespace(
        physics_dt=0.01,
        step_dt=0.1,
        device="cpu",
        num_envs=2,
        nonfinite_events=0,
        cfg=SimpleNamespace(
            variant="dcol",
            training_stage=stage,
            seed=0,
            terrain_seed=0,
            num_static_obstacles=300,
            num_dynamic_obstacles=0,
            sim_to_real=_Randomization(),
            height_target_range=cfg.height_target_range,
        ),
        curriculum=TrainingCurriculum(2, "cpu", stage),
    )
    return DiffNavTrainer(env, cfg, path)


def _assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected, strict=True):
            _assert_state_equal(first, second)
    else:
        assert actual == expected


def _report(trainer, update, rate):
    trainer.completed_updates = update
    trainer.environment_steps = update * 300
    return {"stage": trainer.env.cfg.training_stage, "update": update, "episodes": 1024, "success_rate": rate}


@pytest.mark.parametrize(
    ("name", "optimizer_type", "decay"),
    (
        ("adamw", torch.optim.AdamW, 0.01),
        ("adamw", torch.optim.AdamW, 0.0),
        ("adam", torch.optim.Adam, 0.0),
        ("adam", torch.optim.Adam, 0.01),
    ),
)
def test_runner_constructs_the_requested_optimizer_without_changing_its_weight_decay(
    tmp_path, name, optimizer_type, decay
):
    trainer = _trainer(tmp_path, optimizer=name, weight_decay=decay)
    assert type(trainer.optimizer) is optimizer_type
    assert all(group["weight_decay"] == decay for group in trainer.optimizer.param_groups)
    assert trainer.cfg.height_loss_mode == "goal_height_band"
    assert trainer.cfg.resolved_velocity_loss_mode == "truncated_window"


def test_runner_uses_current_height_identity_and_ahac_limit(tmp_path):
    trainer = _trainer(tmp_path, algorithm="ahac")
    assert trainer.horizon_controller.settings.max_horizon == 300
    assert trainer.horizon_controller.horizon == 32
    trainer.env.cfg.height_target_range = (0.6, 2.4)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="height"):
        DiffNavTrainer(trainer.env, trainer.cfg, tmp_path / "wrong_height")
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


def test_recovery_keeps_each_level_file_immutable_and_restores_the_highest_same_stage_state(tmp_path):
    trainer = _trainer(tmp_path)
    for update, success in enumerate((0.06, 0.065, 0.07), 1):
        assert not trainer._report_controls(_report(trainer, update, success), _Writer())
    assert set(trainer.recovery.stable_checkpoints) == {0.05}
    first = Path(trainer.recovery.stable_checkpoint)
    first_digest = hashlib.sha256(first.read_bytes()).hexdigest()
    assert first.name == "checkpoint_stable_pretrain_s0_u3.pt"
    assert trainer.recovery.reference_success == 0.06
    first_record = copy.deepcopy(trainer.recovery.stable_checkpoints[0.05])

    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(0.1)
    high_state = copy.deepcopy(trainer.model.state_dict())
    for update, success in enumerate((0.16, 0.17, 0.18), 4):
        assert not trainer._report_controls(_report(trainer, update, success), _Writer())
    assert set(trainer.recovery.stable_checkpoints) == {0.05, 0.10, 0.15}
    assert trainer.recovery.stable_checkpoints[0.05] == first_record
    assert hashlib.sha256(first.read_bytes()).hexdigest() == first_digest
    assert trainer.recovery.reference_success == 0.16
    for level in (0.10, 0.15):
        record = trainer.recovery.stable_checkpoints[level]
        path = Path(record["checkpoint"])
        assert path.parent == tmp_path.resolve()
        assert path.name == "checkpoint_stable_pretrain_s0_u6.pt"
        assert record["stage"] == "pretrain_s0" and record["update"] == 6
        assert record["reference_success"] == 0.16
        payload = torch.load(path, map_location="cpu", weights_only=True)
        _assert_state_equal(payload["model_state_dict"], high_state)
        assert tuple(payload["environment"]["height_target_range"]) == (0.5, 2.5)
    assert len(list(tmp_path.glob("checkpoint_stable_*.pt"))) == 2
    highest_path = trainer.recovery.stable_checkpoint
    high_digest = hashlib.sha256(Path(highest_path).read_bytes()).hexdigest()

    # Further improvement within an existing level cannot replace that file
    # or change the reference used by subsequent degradation decisions.
    assert not trainer._report_controls(_report(trainer, 7, 0.19), _Writer())
    assert trainer.recovery.reference_success == 0.16
    assert hashlib.sha256(Path(highest_path).read_bytes()).hexdigest() == high_digest
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(0.4)
    for update in range(8, 11):
        assert not trainer._report_controls(_report(trainer, update, 0.08), _Writer())
    assert trainer.recovery.state == "observing"
    for update in range(11, 14):
        restored = trainer._report_controls(_report(trainer, update, 0.08), _Writer())
        assert restored is (update == 13)
    _assert_state_equal(trainer.model.state_dict(), high_state)
    assert trainer.recovery.stable_checkpoint == highest_path
    assert trainer.recovery.last_report_update == 13
    assert trainer.completed_updates == 13 and trainer.environment_steps == 3900
    assert hashlib.sha256(first.read_bytes()).hexdigest() == first_digest
    assert hashlib.sha256(Path(highest_path).read_bytes()).hexdigest() == high_digest
    records = [json.loads(line) for line in (tmp_path / "optimization_events.jsonl").read_text().splitlines()]
    assert records[-1]["action"] == "restore" and records[-1]["checkpoint"] == highest_path
    assert records[-1]["reference_success"] == 0.16


def test_recovery_refuses_to_overwrite_an_existing_level_file_before_acknowledging(tmp_path):
    trainer = _trainer(tmp_path)
    path = tmp_path / "checkpoint_stable_pretrain_s0_u3.pt"
    original = b"Synthetic pre-existing history; not a trained policy checkpoint"
    path.write_bytes(original)
    for update in (1, 2):
        assert not trainer._report_controls(_report(trainer, update, 0.10), _Writer())
    with pytest.raises(FileExistsError, match="immutable"):
        trainer._report_controls(_report(trainer, 3, 0.10), _Writer())
    assert path.read_bytes() == original
    assert not (tmp_path / "checkpoint_stable_pretrain_s0_10.pt").exists()
    assert trainer.recovery.stable_checkpoints == {}
    assert trainer.recovery.stable_checkpoint is None
    assert trainer.recovery.pending.action == "save_stable"
    assert trainer.recovery.pending.levels == (0.05, 0.10)


def test_actual_restore_keeps_counters_rng_and_reduced_cosine_lr(tmp_path):
    trainer = _trainer(tmp_path)
    # Populate real AdamW state; a fake parameter-only restore would miss it.
    sum(parameter.square().sum() for parameter in trainer.model.parameters()).backward()
    trainer.optimizer.step()
    trainer.normalizer.observe(torch.ones(2, 1, 12, 16), torch.ones(2, 10))
    expected = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    for update, success in enumerate((0.50, 0.51, 0.52), 1):
        assert not trainer._report_controls(_report(trainer, update, success), _Writer())
    checkpoint = trainer.recovery.stable_checkpoint
    assert checkpoint is not None
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(0.2)
    trainer.normalizer.observe(torch.full((2, 1, 12, 16), 99.0), torch.full((2, 10), 99.0))
    for update in range(4, 7):
        assert not trainer._report_controls(_report(trainer, update, 0.2), _Writer())
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0005)
    # A cosine restart cannot silently undo the reduction.
    trainer.schedule_updates = 400
    assert trainer._learning_rate() == pytest.approx(0.0005)
    trainer.elapsed_s = 1234.0
    rng = torch.get_rng_state().clone()
    for update in range(7, 10):
        restored = trainer._report_controls(_report(trainer, update, 0.2), _Writer())
        assert restored is (update == 9)
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert trainer.completed_updates == 9 and trainer.environment_steps == 2700
    assert trainer.schedule_updates == 400 and trainer.elapsed_s == 1234.0
    assert trainer.recovery.last_report_update == 9 and trainer.recovery.restore_count == 1
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0005)
    torch.testing.assert_close(trainer.normalizer.state_moments.mean, torch.ones(10, dtype=torch.float64))
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert all(int(state["step"]) == 1 for state in trainer.optimizer.state.values())


def test_one_report_cannot_reduce_lr_twice_and_old_stage_cannot_restore(tmp_path):
    trainer = _trainer(tmp_path, "plateau")
    for update, success in enumerate((0.50, 0.51, 0.52, 0.20, 0.20, 0.20), 1):
        trainer._report_controls(_report(trainer, update, success), _Writer())
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0005)
    assert trainer.recovery.state == "observing"
    checkpoint = trainer.recovery.stable_checkpoint
    trainer.env.cfg.training_stage = "pretrain_s1"
    with pytest.raises(ValueError, match="exact same run"):
        trainer._restore_stable(checkpoint)
    with pytest.raises(ValueError, match="log directory"):
        trainer._restore_stable(tmp_path.parent / "another_run.pt")


def test_new_rollout_discards_gru_delays_window_and_incomplete_episode_state(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.velocity_history = {"sentinel": True}
    env = trainer.env
    env.perturbation = SimpleNamespace(
        action_buffer=SimpleNamespace(history_length=2), observation_buffer=SimpleNamespace(history_length=5)
    )
    env.curriculum.outcome.fill_(2)
    env.reset = env.curriculum.reset_episodes
    env.get_diffnav_observation = lambda: {"position": torch.full((2, 3), 2.0), "velocity": torch.zeros(2, 3)}
    _, _, state, hidden, previous, action_delay, observation_delay, window = trainer._initialize_rollout()
    assert trainer.velocity_history is None
    assert not hidden.any() and not previous.any() and not state.actuator.any()
    assert not action_delay.history and not observation_delay.history and not window.features().any()
    assert not env.curriculum.outcome.any()


def test_accepted_stage_checkpoint_keeps_normalization_and_resets_report_history(tmp_path):
    original = _trainer(tmp_path)
    original.env.curriculum.stage_passed = True
    original.completed_updates, original.environment_steps, original.schedule_updates = 12, 3600, 12
    original.normalizer.observe(torch.ones(2, 1, 12, 16), torch.full((2, 10), 2.0))
    checkpoint = tmp_path / "checkpoint_stage.pt"
    original._checkpoint(checkpoint)
    next_stage = _trainer(tmp_path / "next", stage="pretrain_s1")
    next_stage.load_stage_checkpoint(checkpoint)
    assert next_stage.completed_updates == 12 and next_stage.schedule_updates == 12
    assert next_stage.recovery.stable_checkpoint is None
    assert next_stage.recovery.stable_checkpoints == {}
    assert next_stage.recovery.stage == "pretrain_s1"
    # Passing a stage without three stable 80% reports does not manufacture
    # a stable80 checkpoint in either stage's immutable history.
    assert not list(tmp_path.glob("checkpoint_stable_*_80.pt"))
    torch.testing.assert_close(next_stage.normalizer.state_moments.mean, torch.full((10,), 2.0, dtype=torch.float64))


@pytest.mark.parametrize(
    ("algorithm", "changed_field", "changed_value"),
    [
        ("bptt", "normalization_variance_floor", 0.002),
        ("shac", "discount", 0.95),
        ("shac", "td_lambda", 0.8),
        ("ahac", "ahac_risk_budget", 0.2),
        ("ahac", "ahac_max_horizon", 150),
        ("bptt", "velocity_loss_mode", "full_window"),
        ("bptt", "height_loss_mode", "boundary_huber"),
        ("bptt", "height_target_range", (0.6, 2.4)),
        ("bptt", "optimizer", "adam"),
        ("bptt", "weight_decay", 0.0),
        ("bptt", "plateau_patience", 0),
        ("bptt", "gradient_projection", "height_priority"),
    ],
)
def test_stage_load_rejects_changed_learning_configuration_before_mutating_state(
    tmp_path, algorithm, changed_field, changed_value
):
    original = _trainer(tmp_path, algorithm=algorithm)
    original.env.curriculum.stage_passed = True
    original.completed_updates = 12
    original.normalizer.observe(torch.ones(2, 1, 12, 16), torch.full((2, 10), 2.0))
    checkpoint = tmp_path / "checkpoint_stage.pt"
    original._checkpoint(checkpoint)
    next_stage = _trainer(tmp_path / "next", stage="pretrain_s1", algorithm=algorithm, **{changed_field: changed_value})
    model_before = copy.deepcopy(next_stage.model.state_dict())
    normalization_before = copy.deepcopy(next_stage.normalizer.state_dict())
    with pytest.raises(ValueError, match=f"cannot change {changed_field}"):
        next_stage.load_stage_checkpoint(checkpoint)
    _assert_state_equal(next_stage.model.state_dict(), model_before)
    _assert_state_equal(next_stage.normalizer.state_dict(), normalization_before)
    assert next_stage.completed_updates == 0 and next_stage.parent_checkpoint is None


@pytest.mark.parametrize("schedule", ("cosine", "plateau", "constant"))
def test_stage_load_allows_only_the_intended_stage_control_overrides(tmp_path, schedule):
    original = _trainer(tmp_path, schedule=schedule)
    original.env.curriculum.stage_passed = True
    original.completed_updates, original.environment_steps, original.schedule_updates = 12, 3600, 12
    checkpoint = tmp_path / "checkpoint_stage.pt"
    original._checkpoint(checkpoint)
    next_stage = _trainer(
        tmp_path / "next",
        schedule=schedule,
        stage="pretrain_s1",
        lr=0.0005,
        max_iterations=2,
        stop_after_acceptance=True,
        gradient_diagnostic_updates=(17, 19),
    )
    next_stage.load_stage_checkpoint(checkpoint)
    assert next_stage.cfg.lr == 0.0005 and next_stage.cfg.max_iterations == 2
    assert next_stage.cfg.stop_after_acceptance is True
    assert next_stage.cfg.gradient_diagnostic_updates == (17, 19)
    assert next_stage.completed_updates == 12 and next_stage.environment_steps == 3600
    assert next_stage.parent_checkpoint == str(checkpoint.resolve())
    assert next_stage.optimizer.param_groups[0]["lr"] == next_stage.cfg.lr
    assert next_stage._learning_rate() <= next_stage.cfg.lr
    _assert_state_equal(next_stage.model.state_dict(), original.model.state_dict())


@pytest.mark.parametrize("algorithm", ("shac", "ahac"))
def test_same_stage_restore_keeps_critic_progress_rng_and_horizon_progress(tmp_path, algorithm):
    trainer = _trainer(tmp_path, algorithm=algorithm, critic_iterations=1, critic_max_iterations=1)
    trainer._initialize_critic(torch.zeros(2, 3))
    critic = trainer.actor_critic
    observations = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) / 7
    costs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    next_values = torch.zeros_like(costs)
    trace_cut = torch.zeros_like(costs, dtype=torch.bool)
    critic.fit(observations, costs, next_values, trace_cut)
    controller = trainer.horizon_controller
    if controller is not None:
        controller.update(torch.full((controller.horizon, 2), 0.3))
    saved_critic = copy.deepcopy(critic.state_dict())
    saved_horizon = copy.deepcopy(controller.state_dict()) if controller is not None else None
    assert saved_critic["optimizer"]["state"]  # Exercise nonempty AdamW moments.
    checkpoint = tmp_path / "checkpoint_stable.pt"
    trainer._checkpoint(checkpoint)

    # Real further fits advance both the private shuffle generator and its
    # cumulative counter while changing live and frozen critic parameters.
    for offset in (1.0, 2.0):
        critic.fit(observations, costs + offset, next_values, trace_cut)
        if controller is not None:
            controller.update(torch.full((controller.horizon, 2), 0.3))
    current_critic = copy.deepcopy(critic.state_dict())
    assert current_critic["completed_fits"] > saved_critic["completed_fits"]
    assert not torch.equal(current_critic["shuffle_rng_state"], saved_critic["shuffle_rng_state"])
    assert any(
        not torch.equal(value, saved_critic["critics"][name]) for name, value in current_critic["critics"].items()
    )
    current_horizon_updates = controller.completed_updates if controller is not None else None
    if controller is not None:
        assert controller.horizon != saved_horizon["horizon"]
        assert controller.dual != saved_horizon["dual"]
    trainer.completed_updates, trainer.environment_steps, trainer.schedule_updates = 25, 7500, 25
    trainer.elapsed_s = 123.0
    trainer.lr_scale, trainer.lr_cap = 0.2, 0.0002
    for group in trainer.optimizer.param_groups:
        group["lr"] = 0.0002
    global_rng = torch.get_rng_state().clone()
    trainer._restore_stable(checkpoint)

    # Learned weights, target copies and optimizer moments come from the
    # stable checkpoint; cumulative work and future minibatch randomness do not.
    expected_critic = copy.deepcopy(saved_critic)
    expected_critic["completed_fits"] = current_critic["completed_fits"]
    expected_critic["shuffle_rng_state"] = current_critic["shuffle_rng_state"]
    _assert_state_equal(critic.state_dict(), expected_critic)
    if controller is not None:
        expected_horizon = {**saved_horizon, "completed_updates": current_horizon_updates}
        _assert_state_equal(controller.state_dict(), expected_horizon)
    assert (trainer.completed_updates, trainer.environment_steps, trainer.schedule_updates) == (25, 7500, 25)
    assert trainer.elapsed_s == 123.0 and trainer.lr_scale == 0.2 and trainer.lr_cap == 0.0002
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0002)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
