"""Actual optimizer measurements and decisions based only on fresh reports."""

import copy
import json

import pytest
import torch
from torch import nn

from isaaclab_navrl.zhang.optimization import (
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


def _report(update, success, stage="pretrain_s1"):
    return {"update": update, "stage": stage, "episodes": 1024, "success_rate": success,
            "stage_passed": success >= 0.8}


def _armed_controller():
    controller = RecoveryController("pretrain_s1", RecoveryConfig(enabled=True))
    for update, success in enumerate((0.50, 0.53, 0.52), 1):
        decision = controller.observe(_report(update, success))
    assert decision.action == "save_stable" and decision.reference_success == 0.5
    controller.acknowledge("save_stable", checkpoint="/run/pretrain_s1/stable_u3.pt")
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
    assert decision.action == "restore" and decision.checkpoint.endswith("stable_u3.pt")
    controller.acknowledge("restore")
    assert controller.last_report_update == 9 and controller.restore_count == 1
    for update in range(10, 13):
        assert controller.observe(_report(update, 0.1)).action == "none"
    assert controller.state == "armed" and controller.last_report_update == 12
    with pytest.raises(ValueError, match="strictly increasing"):
        controller.observe(_report(3, 0.5))


def test_recovery_ignores_early_hover_and_abandons_restore_if_reduction_recovers():
    warming = RecoveryController("pretrain_s1", RecoveryConfig(enabled=True))
    for update in range(1, 8):
        assert warming.observe(_report(update, 0.05)).action == "none"
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
    assert restored.last_report_update == 6
    with pytest.raises(ValueError, match="same stage"):
        restored.observe(_report(7, 0.2))
    with pytest.raises(ValueError, match="same version, stage"):
        restored.load_state_dict(payload)
    disabled = RecoveryController("pretrain_s1")
    for update, success in enumerate((0.8, 0.8, 0.8, 0.0, 0.0, 0.0), 1):
        assert disabled.observe(_report(update, success)).action == "none"
    assert disabled.stable_checkpoint is None
