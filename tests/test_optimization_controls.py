"""Actual optimizer measurements and decisions based only on fresh reports."""

import copy
import json

import pytest
import torch
from torch import nn

from aerialnav.diffnav.optimization import (
    OptimizerStepMonitor,
    RecoveryConfig,
    RecoveryController,
    make_plateau_scheduler,
)


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(3, 4)
        self.fc = nn.Linear(4, 6)

    def forward(self, value):
        return self.fc(self.trunk(value).tanh())


def _flat(model):
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])


def _compare_tree(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _compare_tree(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            _compare_tree(left, right)
    else:
        assert first == second


def test_monitor_reports_actual_all_layer_and_six_head_updates_without_changing_adamw():
    torch.manual_seed(5)
    model = _Policy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    inputs = torch.arange(9, dtype=torch.float32).reshape(3, 3) / 4
    model(inputs).square().sum().backward()
    optimizer.step()  # Nonempty optimizer moments.
    expected_model = copy.deepcopy(model)
    expected_optimizer = torch.optim.AdamW(expected_model.parameters(), lr=0.001)
    expected_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    monitor = OptimizerStepMonitor(model)
    before = _flat(model).clone()
    before_head = torch.cat((model.fc.weight.detach(), model.fc.bias.detach()[:, None]), dim=1)
    gradients = []
    for policy, optim in ((model, optimizer), (expected_model, expected_optimizer)):
        optim.zero_grad(set_to_none=True)
        policy(inputs).sin().sum().backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.2)
        if policy is model:
            gradients = [parameter.grad.clone() for parameter in policy.parameters()]
            monitor.before_step()
        optim.step()
    record = monitor.after_step()
    _compare_tree(model.state_dict(), expected_model.state_dict())
    _compare_tree(optimizer.state_dict(), expected_optimizer.state_dict())
    for actual, expected in zip(model.parameters(), gradients, strict=True):
        torch.testing.assert_close(actual.grad, expected, rtol=0, atol=0)
    expected_norm = float(before.double().norm())
    expected_delta = float((_flat(model) - before).double().norm())
    assert record["Optim/param_norm"] == pytest.approx(expected_norm, rel=1e-7)
    assert record["Optim/update_norm"] == pytest.approx(expected_delta, rel=1e-7)
    assert record["Optim/update_ratio"] == pytest.approx(expected_delta / expected_norm, rel=1e-7)
    assert "Optim/layers/trunk/update_ratio" in record and "Optim/layers/fc/update_ratio" in record
    after_head = torch.cat((model.fc.weight.detach(), model.fc.bias.detach()[:, None]), dim=1)
    for index, label in enumerate(("ax", "vx", "ay", "vy", "az", "vz")):
        assert record[f"Optim/head/{label}/update_norm"] == pytest.approx(
            float((after_head[index] - before_head[index]).double().norm()), rel=1e-7
        )
    json.dumps(record, allow_nan=False)


def test_plateau_is_off_by_default_uses_success_max_and_resumes_official_state():
    parameter = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    assert make_plateau_scheduler(optimizer) is None
    assert make_plateau_scheduler(optimizer, "constant") is None
    scheduler = make_plateau_scheduler(optimizer, "plateau", patience=1, cooldown=0)
    assert type(scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau
    rates = []
    for success in (0.1, 0.102, 0.103):
        scheduler.step(success)
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates == pytest.approx([0.01, 0.01, 0.005])
    restored_optimizer = torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=0.01)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    restored = make_plateau_scheduler(restored_optimizer, "plateau", patience=1, cooldown=0)
    restored.load_state_dict(copy.deepcopy(scheduler.state_dict()))
    for success in (0.2, 0.199, 0.198):
        scheduler.step(success)
        restored.step(success)
        assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0025)
    _compare_tree(restored.state_dict(), scheduler.state_dict())


def test_plateau_default_patience_is_two_fresh_bad_reports_before_third_reduces():
    optimizer = torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=0.01)
    scheduler = make_plateau_scheduler(optimizer, "plateau")
    assert scheduler.patience == 2
    for success in (0.1, 0.1, 0.1):
        scheduler.step(success)
        assert optimizer.param_groups[0]["lr"] == 0.01
    scheduler.step(0.1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)


def _report(update, success, stage="pretrain_s1"):
    return {"update": update, "stage": stage, "episodes": 1024, "success_rate": success, "stage_passed": success >= 0.8}


def _save_requested(controller, decision, suffix=""):
    checkpoints = {
        level: f"/run/{decision.stage}/stable_{level:.2f}_u{decision.update}{suffix}.pt" for level in decision.levels
    }
    controller.acknowledge("save_stable", checkpoints=checkpoints)
    return checkpoints


def _armed_controller():
    controller = RecoveryController("pretrain_s1", RecoveryConfig(enabled=True))
    for update, success in enumerate((0.50, 0.53, 0.52), 1):
        decision = controller.observe(_report(update, success))
    assert decision.action == "save_stable" and decision.reference_success == 0.5
    _save_requested(controller, decision)
    return controller


def test_recovery_requests_lr_then_restore_with_ack_and_never_rewinds_report_updates():
    controller = _armed_controller()
    for update in range(4, 7):
        report = _report(update, 0.2)
        original = copy.deepcopy(report)
        decision = controller.observe(report)
        assert report == original
    assert decision.action == "reduce_lr" and decision.lr_factor == 0.5
    with pytest.raises(RuntimeError, match="Acknowledge"):
        controller.observe(_report(7, 0.2))
    controller.acknowledge("reduce_lr")
    for update in range(7, 10):
        decision = controller.observe(_report(update, 0.2))
    assert decision.action == "restore" and decision.checkpoint.endswith("stable_0.50_u3.pt")
    catalogue = copy.deepcopy(controller.stable_checkpoints)
    controller.acknowledge("restore")
    assert controller.last_report_update == 9 and controller.restore_count == 1
    assert not controller.window and controller.bad_reports == controller.observed_reports == 0
    assert controller.stable_checkpoints == catalogue
    for update in range(10, 13):
        assert controller.observe(_report(update, 0.1)).action == "none"
    assert controller.state == "armed" and controller.last_report_update == 12
    with pytest.raises(ValueError, match="strictly increasing"):
        controller.observe(_report(3, 0.5))


def test_recovery_ignores_early_hover_and_abandons_restore_if_reduction_recovers():
    warming = RecoveryController("pretrain_s1", RecoveryConfig(enabled=True))
    for update in range(1, 8):
        assert warming.observe(_report(update, 0.049)).action == "none"
    assert warming.stable_checkpoint is None
    controller = _armed_controller()
    for update in range(4, 7):
        controller.observe(_report(update, 0.2))
    controller.acknowledge("reduce_lr")
    for update, success in zip(range(7, 10), (0.2, 0.48, 0.5), strict=True):
        decision = controller.observe(_report(update, success))
    assert decision.action == "none" and controller.state == "armed"
    assert controller.restore_count == 0


def test_recovery_state_roundtrip_stage_isolation_and_disabled_behavior():
    controller = _armed_controller()
    controller.observe(_report(4, 0.2))
    payload = json.loads(json.dumps(controller.state_dict(), allow_nan=False))
    restored = RecoveryController("pretrain_s1", RecoveryConfig(enabled=True))
    restored.load_state_dict(payload)
    for update in (5, 6):
        assert restored.observe(_report(update, 0.2)) == controller.observe(_report(update, 0.2))
    assert restored.state_dict() == controller.state_dict()
    restored.acknowledge("reduce_lr")
    restored.reset_stage("pretrain_s2")
    assert restored.stable_checkpoint is None and restored.reference_success is None
    assert restored.stable_checkpoints == {}
    assert restored.last_report_update == 6
    with pytest.raises(ValueError, match="same stage"):
        restored.observe(_report(7, 0.2))
    with pytest.raises(ValueError, match="same version, stage"):
        restored.load_state_dict(payload)
    disabled = RecoveryController("pretrain_s1")
    for update, success in enumerate((0.8, 0.8, 0.8, 0.0, 0.0, 0.0), 1):
        assert disabled.observe(_report(update, success)).action == "none"
    assert disabled.stable_checkpoint is None


@pytest.mark.parametrize("reference,threshold", ((0.05, 0.03), (0.10, 0.07), (0.15, 0.105), (0.20, 0.15), (0.80, 0.75)))
def test_recovery_strict_degradation_uses_relative_margin_with_two_and_five_pp_bounds(reference, threshold):
    controller = RecoveryController("pretrain_s1", RecoveryConfig(enabled=True))
    for update in range(1, 4):
        decision = controller.observe(_report(update, reference))
    _save_requested(controller, decision)
    assert controller.degradation_threshold == threshold
    for update in range(4, 7):
        assert controller.observe(_report(update, threshold)).action == "none"
    assert controller.bad_reports == 0
    for update in range(7, 10):
        decision = controller.observe(_report(update, threshold - 0.001))
    assert decision.action == "reduce_lr"


def test_recovery_levels_are_exact_grid_cross_multiple_levels_once_and_keep_low_history():
    controller = RecoveryController("pretrain_s1", RecoveryConfig(enabled=True))
    assert controller.levels == tuple(value / 100 for value in range(5, 81, 5))
    for update, rate in enumerate((0.15, 0.16, 0.17), 1):
        decision = controller.observe(_report(update, rate))
    assert decision.levels == (0.05, 0.10, 0.15)
    initial_paths = _save_requested(controller, decision)
    assert controller.reference_success == 0.15
    first_history = copy.deepcopy(controller.stable_checkpoints)
    for update, rate in enumerate((0.21, 0.22, 0.23), 4):
        decision = controller.observe(_report(update, rate))
    assert decision.action == "save_stable" and decision.levels == (0.20,)
    new_paths = _save_requested(controller, decision)
    assert controller.reference_success == 0.21
    assert controller.stable_checkpoint == new_paths[0.20]
    for level, path in initial_paths.items():
        assert controller.stable_checkpoints[level] == first_history[level]
        assert controller.stable_checkpoints[level]["checkpoint"] == path
    # Improving inside an already saved level does not replace its historical
    # state or silently move its reference away from the saved window minimum.
    for update in range(7, 10):
        assert controller.observe(_report(update, 0.24)).action == "none"
    assert controller.reference_success == 0.21


def test_recovery_ack_mapping_is_atomic_requires_all_new_levels_and_preserves_older_paths():
    controller = RecoveryController("pretrain_s1", RecoveryConfig(enabled=True))
    for update in range(1, 4):
        decision = controller.observe(_report(update, 0.10))
    for invalid in (
        {0.05: "/five.pt"},
        {0.05: "/five.pt", 0.10: ""},
        {0.05: "/five.pt", 0.10: "/ten.pt", 0.15: "/extra.pt"},
    ):
        with pytest.raises(ValueError, match="mapping|path"):
            controller.acknowledge("save_stable", checkpoints=invalid)
        assert controller.stable_checkpoints == {} and controller.pending == decision
    # A single trained state can serve several newly crossed levels.
    controller.acknowledge("save_stable", checkpoints={0.05: "/u3.pt", 0.10: "/u3.pt"})
    for update in range(4, 7):
        decision = controller.observe(_report(update, 0.15))
    assert decision.levels == (0.15,)
    with pytest.raises(ValueError, match="overwrite"):
        controller.acknowledge("save_stable", checkpoints={0.15: "/u3.pt"})
    assert set(controller.stable_checkpoints) == {0.05, 0.10}


def test_recovery_three_report_stability_and_spread_are_independent_of_stage_pass():
    controller = RecoveryController("pretrain_s1", RecoveryConfig(enabled=True))
    for update, rate in enumerate((0.04, 0.10, 0.04, 0.10), 1):
        assert controller.observe(_report(update, rate)).action == "none"
    assert controller.stable_checkpoint is None
    for update, rate in enumerate((0.50, 0.55, 0.52), 5):
        decision = controller.observe(_report(update, rate))
    assert decision.action == "save_stable" and decision.reference_success == 0.50
    _save_requested(controller, decision)
    assert controller.observe(_report(8, 0.8)).action == "none"
    assert 0.8 not in controller.stable_checkpoints
    assert controller.observe(_report(9, 0.8)).action == "none"
    decision = controller.observe(_report(10, 0.8))
    assert decision.action == "save_stable" and decision.levels[-1] == 0.8
    assert decision.reference_success == 0.8


def test_recovery_nonconsecutive_drops_do_not_trigger_and_restore_selects_highest_level():
    controller = _armed_controller()
    for update, rate in enumerate((0.42, 0.43, 0.45, 0.42, 0.45), 4):
        assert controller.observe(_report(update, rate)).action == "none"
    for update in range(9, 12):
        decision = controller.observe(_report(update, 0.61))
    assert decision.levels == (0.55, 0.60)
    _save_requested(controller, decision)
    for update in range(12, 15):
        decision = controller.observe(_report(update, 0.5))
    assert decision.action == "reduce_lr"
    controller.acknowledge("reduce_lr")
    for update in range(15, 18):
        decision = controller.observe(_report(update, 0.5))
    assert decision.action == "restore"
    assert decision.reference_success == 0.61
    assert decision.checkpoint == controller.stable_checkpoints[0.60]["checkpoint"]
    controller.acknowledge("restore")
    assert controller.last_report_update == 17 and controller.restore_count == 1
    assert 0.05 in controller.stable_checkpoints and 0.60 in controller.stable_checkpoints


def test_recovery_pending_multi_level_save_roundtrips_json_and_rejects_old_state():
    controller = RecoveryController("pretrain_s1", RecoveryConfig(enabled=True))
    for update in range(1, 4):
        decision = controller.observe(_report(update, 0.15))
    payload = json.loads(json.dumps(controller.state_dict(), allow_nan=False))
    restored = RecoveryController("pretrain_s1", RecoveryConfig(enabled=True))
    restored.load_state_dict(payload)
    assert restored.pending == decision
    assert restored.state_dict() == controller.state_dict()
    _save_requested(restored, restored.pending)
    assert restored.reference_success == 0.15 and len(restored.stable_checkpoints) == 3
    payload["version"] = 1
    with pytest.raises(ValueError, match="same version"):
        restored.load_state_dict(payload)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"stable_min_success": 0},
        {"stable_level_step": 0},
        {"stable_max_success": 0.04},
        {"degrade_min_drop": 0.06},
        {"degrade_abs_drop": 0},
        {"degrade_relative_drop": 0},
    ),
)
def test_recovery_rejects_invalid_level_or_degradation_ranges(kwargs):
    with pytest.raises(ValueError):
        RecoveryController("pretrain_s1", RecoveryConfig(**kwargs))
