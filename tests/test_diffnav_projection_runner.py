"""Exercise probe timing and the real runner's raw/projection/clip/step sequence."""

import copy
import json
from dataclasses import replace

import pytest
import torch
from test_diffnav_contract import AnalyticEnv

from aerialnav.diffnav import runner as runner_module
from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.diffnav.runner import DiffNavTrainer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class _TrackedEnv(AnalyticEnv):
    def __init__(self):
        super().__init__()
        self.signed_queries = []

    def nearest_obstacle_signed_distance(self, query):
        distance = super().nearest_obstacle_signed_distance(query)
        self.signed_queries.append(torch.minimum(distance, query[..., 2] - 0.01).detach().clone())
        return distance


def _flat_grad(model):
    return torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()])


def _equal_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected, strict=True):
            _equal_tree(first, second)
    else:
        assert actual == expected


def _trainer(path, cfg):
    torch.manual_seed(103)
    trainer = DiffNavTrainer(_TrackedEnv(), cfg, path)
    # A nonzero global counter verifies that diagnostic IDs are not interpreted
    # as local indices. These small synthetic runs are never acceptance runs.
    trainer.completed_updates = 5
    trainer.schedule_updates = 5
    trainer.environment_steps = 40
    return trainer


@pytest.mark.parametrize("projection", ("none", "height_priority"))
@pytest.mark.parametrize("recurrent_model", ("gru", "sru_lstm_gated"))
def test_runner_probe_uses_pre_call_state_and_changes_no_training_update(
    tmp_path, monkeypatch, projection, recurrent_model
):
    cfg = DiffNavTrainCfg(
        recurrent_model=recurrent_model,
        physics_dt=0.01,
        policy_dt=0.1,
        variant="step",
        velocity_loss_mode="variant_default",
        timesteps=4,
        max_iterations=2,
        velocity_window_s=0.2,
        max_grad_norm=0.05,
        observation_normalization=True,
        recovery_enabled=False,
        gradient_projection=projection,
    )
    baseline = _trainer(tmp_path / "baseline", cfg)
    baseline_result = baseline.run()
    expected_model = copy.deepcopy(baseline.model.state_dict())
    expected_optimizer = copy.deepcopy(baseline.optimizer.state_dict())
    expected_normalizer = copy.deepcopy(baseline.normalizer.state_dict())
    expected_rng = torch.get_rng_state().clone()
    assert not (baseline.log_dir / "gradient_diagnostics.jsonl").exists()

    measured = _trainer(tmp_path / "measured", replace(cfg, gradient_diagnostic_updates=(7,)))
    calls, events, observed = [], [], {}
    checkpointed_forward = measured.model.forward_checkpointed

    def record_forward(depth, features, hidden):
        entry = {
            name: value.detach().clone()
            for name, value in (
                ("depth", depth),
                ("features", features),
                ("hidden", hidden),
            )
        }
        result = checkpointed_forward(depth, features, hidden)
        entry["hidden_after_call"] = result[1].detach().clone()
        calls.append(entry)
        return result

    monkeypatch.setattr(measured.model, "forward_checkpointed", record_forward)
    original_diagnostic = runner_module.GradientDiagnostic
    original_probe = runner_module.ActionUpdateProbe
    original_projection = runner_module.height_priority_gradient
    original_clip = torch.nn.utils.clip_grad_norm_

    class OrderedDiagnostic(original_diagnostic):
        def component_gradient(self, name):
            result = super().component_gradient(name)
            if name == "height_boundary":
                observed["height"] = result.clone()
            return result

        def check_total_gradient(self):
            events.append("raw_check")
            super().check_total_gradient()

    class OrderedProbe(original_probe):
        def __init__(self, model, optimizer, **kwargs):
            events.append("probe")
            torch.testing.assert_close(_flat_grad(model), kwargs["total_flat"], rtol=0, atol=0)
            torch.testing.assert_close(kwargs["height_flat"], observed["height"], rtol=0, atol=0)
            observed["raw"] = kwargs["total_flat"].clone()
            observed["probe_inputs"] = {name: value.detach().clone() for name, value in kwargs["probe"].items()}
            super().__init__(model, optimizer, **kwargs)

        def finish(self, model, **kwargs):
            events.append("finish")
            return super().finish(model, **kwargs)

    def record_projection(total, height, weight):
        events.append("projection")
        torch.testing.assert_close(_flat_grad(measured.model), total, rtol=0, atol=0)
        result = original_projection(total, height, weight)
        observed["projected"] = result[0].clone()
        return result

    def record_clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        if parameters and parameters[0] is next(measured.model.parameters()):
            events.append("clip")
            if projection == "height_priority":
                torch.testing.assert_close(_flat_grad(measured.model), observed["projected"], rtol=0, atol=0)
            elif "raw" in observed:
                torch.testing.assert_close(_flat_grad(measured.model), observed["raw"], rtol=0, atol=0)
        return original_clip(parameters, *args, **kwargs)

    monkeypatch.setattr(runner_module, "GradientDiagnostic", OrderedDiagnostic)
    monkeypatch.setattr(runner_module, "ActionUpdateProbe", OrderedProbe)
    monkeypatch.setattr(runner_module, "height_priority_gradient", record_projection)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    result = measured.run()

    _equal_tree(measured.model.state_dict(), expected_model)
    _equal_tree(measured.optimizer.state_dict(), expected_optimizer)
    _equal_tree(measured.normalizer.state_dict(), expected_normalizer)
    _equal_tree(torch.get_rng_state(), expected_rng)
    assert result["completed_updates"] == baseline_result["completed_updates"] == 7
    assert result["environment_steps"] == baseline_result["environment_steps"] == 56
    assert measured.env.total_steps == baseline.env.total_steps == 8
    assert not result["acceptance_passed"]
    records = [json.loads(line) for line in (measured.log_dir / "gradient_diagnostics.jsonl").read_text().splitlines()]
    assert len(records) == 1 and records[0]["update"] == 7
    record = records[0]
    assert record["groups"]["trunk"]["parameter_count"] == (
        sum(parameter.numel() for parameter in measured.model.parameters()) - measured.model.fc.weight.numel()
    )
    assert record["applied_gradient_projection"] == projection
    assert record["total_gradient_check"]["relative_l2_error"] < 2e-5
    probe_record = record["optimizer_action_probe"]
    assert probe_record["actual_step"]["applied_projection"] == projection
    assert probe_record["actual_step"]["parameter_max_abs_error"] == 0
    assert probe_record["actual_step"]["command_match_error"]["max_abs"] == 0
    expected_events = ["raw_check", "probe"]
    if projection == "height_priority":
        expected_events.append("projection")
    expected_events.extend(("clip", "finish"))
    assert events[-len(expected_events) :] == expected_events
    assert events.count("probe") == events.count("finish") == 1

    assert len(calls) == 8  # No extra environment rollout or checkpointed policy call.
    clearance = (measured.env.signed_queries[-1] - measured.env.diffnav_parameters["margin"][:, None]).amin(dim=1)
    expected_ids = clearance.topk(min(16, measured.env.num_envs), largest=False).indices
    selection = probe_record["probe_selection"]
    assert selection["selection"] == "smallest_minimum_pre_step_lookahead_clearance"
    assert selection["environment_ids"] == expected_ids.tolist()
    assert selection["clearance_m"] == clearance[expected_ids].tolist()
    for name in ("depth", "features", "hidden"):
        torch.testing.assert_close(observed["probe_inputs"][name], calls[-1][name][expected_ids], rtol=0, atol=0)
    # Recurrent state after the selected call is distinct, so the test would
    # detect accidentally probing one extra recurrent transition.
    assert not torch.equal(observed["probe_inputs"]["hidden"], calls[-1]["hidden_after_call"][expected_ids])
    assert not any(value.requires_grad for value in observed["probe_inputs"].values())
