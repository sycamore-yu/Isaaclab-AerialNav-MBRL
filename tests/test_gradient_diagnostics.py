"""Gradient measurements must leave checkpointed BPTT and AdamW unchanged."""

import copy
import json
from pathlib import Path

import pytest
import torch

from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.diffnav.diagnostics import LOSS_COEFFICIENTS, GradientDiagnostic
from aerialnav.diffnav.losses import diffnav_transition_costs
from aerialnav.diffnav.model import DiffNavPolicy, decode_action


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _losses(model, cfg):
    """Use the real checkpointed policy and seven losses on one short graph."""
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(71)
    depth = torch.randn(4, 2, 1, 12, 16, generator=generator).to(device)
    velocity = torch.tensor([[0.8, -0.2, 0.1], [0.6, 0.3, -0.1]], device=device)
    position = torch.tensor([[0.4, 0.0, 3.8], [0.5, 0.1, 3.7]], device=device)
    target = torch.tensor([[2.0, 0.1, -0.4], [1.5, -0.2, -0.3]], device=device)
    up = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(2, -1)
    margin = torch.full((2,), 0.3, device=device)
    hidden, previous = torch.zeros(2, 192, device=device), torch.zeros(2, 3, device=device)
    heading = torch.eye(3, device=device).expand(2, -1, -1)
    lookahead = torch.linspace(0, 0.1, 10, device=device)[None, :, None]
    history = {
        name: []
        for name in (
            "velocity",
            "target_velocity",
            "episode_id",
            "position",
            "obstacle_signed_distance",
            "height_target",
            "height_range",
            "margin",
            "command",
            "previous_command",
            "velocity_prediction",
            "observed_velocity",
        )
    }
    for frame in depth:
        features = torch.cat((velocity, target, up, margin[:, None]), dim=-1)
        output, hidden = model.forward_checkpointed(frame, features, hidden)
        command, prediction = decode_action(output, heading, torch.ones(2, 1, device=device))
        query = position[:, None] + velocity[:, None] * lookahead
        closest = query.detach().clone()
        closest[..., 0] = 1.0
        history["obstacle_signed_distance"].append((closest - query).norm(dim=-1))
        history["height_target"].append(torch.full((2,), 2.0, device=device))
        history["height_range"].append(torch.full((2, 2), 2.0, device=device))
        history["observed_velocity"].append(velocity.detach())
        position = position + velocity * 0.1 + command * 0.005
        velocity = velocity + command * 0.1
        for name, value in (
            ("velocity", velocity),
            ("target_velocity", target),
            ("position", position),
            ("episode_id", torch.zeros(2, dtype=torch.long, device=device)),
            ("margin", margin),
            ("command", command),
            ("previous_command", previous),
            ("velocity_prediction", prediction),
        ):
            history[name].append(value)
        previous = command
    return diffnav_transition_costs(history, cfg)[0]


def _step(model, optimizer, cfg, diagnose):
    losses = _losses(model, cfg)
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    rng_before = torch.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    diagnostic = GradientDiagnostic(model, losses, cfg) if diagnose else None
    assert all(parameter.grad is None for parameter in model.parameters())
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    if cuda_rng_before is not None:
        torch.testing.assert_close(torch.cuda.get_rng_state(device), cuda_rng_before, rtol=0, atol=0)
    losses["total"].backward()
    original_gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
    if diagnostic is not None:
        diagnostic.check_total_gradient()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    record = diagnostic.finish(optimizer, grad_norm, cfg.max_grad_norm) if diagnostic is not None else None
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    if cuda_rng_before is not None:
        torch.testing.assert_close(torch.cuda.get_rng_state(device), cuda_rng_before, rtol=0, atol=0)
    return original_gradients, grad_norm, record


def _compare_tree(actual, expected, rtol, atol):
    """Report errors for the same comparisons used by pytest and the CUDA entry."""
    result = {"passed": True, "tensor_count": 0, "max_abs_error": 0.0}
    difference_squared, reference_squared = 0.0, 0.0

    def compare(left, right):
        nonlocal difference_squared, reference_squared
        if isinstance(left, torch.Tensor):
            assert isinstance(right, torch.Tensor) and left.shape == right.shape and left.dtype == right.dtype
            difference = left.detach().double() - right.detach().double()
            difference_squared += float(difference.square().sum())
            reference_squared += float(right.detach().double().square().sum())
            result["tensor_count"] += 1
            result["max_abs_error"] = max(result["max_abs_error"], float(difference.abs().max()))
            result["passed"] &= bool(torch.allclose(left, right, rtol=rtol, atol=atol))
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                compare(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for first, second in zip(left, right, strict=True):
                compare(first, second)
        else:
            result["passed"] &= left == right

    compare(actual, expected)
    result["relative_l2_error"] = difference_squared**0.5 / max(reference_squared**0.5, 1e-12)
    return result


def _memory_start(device):
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_allocated(device)


def _memory_finish(device, allocated_before):
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return {
        "allocated_before_bytes": allocated_before,
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "additional_peak_allocated_bytes": peak - allocated_before,
    }


def run_equivalence(device):
    """N=2/H=4 smoke on CPU or CUDA; no simulator, rollout copy, or alternate losses."""
    torch.manual_seed(63)
    cfg = DiffNavTrainCfg(
        physics_dt=0.01,
        policy_dt=0.1,
        variant="step",
        velocity_loss_mode="variant_default",
        timesteps=4,
        max_grad_norm=0.2,
    )
    baseline = DiffNavPolicy().to(device)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=cfg.lr)
    _step(baseline, baseline_optimizer, cfg, diagnose=False)  # Populate AdamW moments.
    measured = copy.deepcopy(baseline)
    measured_optimizer = torch.optim.AdamW(measured.parameters(), lr=cfg.lr)
    measured_optimizer.load_state_dict(copy.deepcopy(baseline_optimizer.state_dict()))
    before = torch.cat([parameter.detach().reshape(-1) for parameter in measured.parameters()])

    allocated = _memory_start(device)
    expected_gradients, expected_norm, _ = _step(baseline, baseline_optimizer, cfg, diagnose=False)
    baseline_memory = _memory_finish(device, allocated)
    allocated = _memory_start(device)
    actual_gradients, actual_norm, record = _step(measured, measured_optimizer, cfg, diagnose=True)
    diagnostic_memory = _memory_finish(device, allocated)

    # CPU remains exact. CUDA reports its actual errors against this fixed gate.
    rtol, atol = (1e-5, 1e-7) if device.type == "cuda" else (0.0, 0.0)
    closure_tolerance = 2e-5 if device.type == "cuda" else 2e-6
    expected_after = torch.cat([parameter.detach().reshape(-1) for parameter in baseline.parameters()])
    after = torch.cat([parameter.detach().reshape(-1) for parameter in measured.parameters()])
    checks = {
        "original_gradients": _compare_tree(actual_gradients, expected_gradients, rtol, atol),
        "pre_clip_norm": _compare_tree(actual_norm, expected_norm, rtol, atol),
        "model_state": _compare_tree(measured.state_dict(), baseline.state_dict(), rtol, atol),
        "parameter_update": _compare_tree(after - before, expected_after - before, rtol, atol),
        "optimizer_state": _compare_tree(measured_optimizer.state_dict(), baseline_optimizer.state_dict(), rtol, atol),
        "clipping_exercised": bool(actual_norm > cfg.max_grad_norm),
        "component_sum_closed": record["total_gradient_check"]["relative_l2_error"] < closure_tolerance,
        "cpu_rng_unchanged": True,  # Checked before and after both paths in _step.
        "cuda_rng_unchanged": True if device.type == "cuda" else None,
    }
    assert record["diagnostic_s"] >= 0
    assert set(record["groups"]) == {"all", "trunk", "z_head"}
    assert record["groups"]["z_head"]["parameter_count"] == 2 * measured.fc.weight.shape[1]
    displacement = (after - before).double()
    actual_total = torch.cat([gradient.reshape(-1) for gradient in actual_gradients]).double()
    projected = sum(
        weight * value
        for weight, value in zip(record["coefficients"], record["groups"]["all"]["linearized_delta_loss"], strict=True)
    )
    projection_reference = float(actual_total @ displacement)
    checks["actual_update_projection_closed"] = abs(projected - projection_reference) <= (
        1e-10 + closure_tolerance * abs(projection_reference)
    )
    # Serialization rejects tensors, graphs, and nonfinite pseudo-cosines.
    assert json.loads(json.dumps(record, allow_nan=False))["term_order"] == list(record["term_order"])
    return {
        "passed": all(value["passed"] if isinstance(value, dict) else value is not False for value in checks.values()),
        "device": str(device),
        "torch_version": torch.__version__,
        "num_envs": 2,
        "rollout_steps": 4,
        "rtol": rtol,
        "atol": atol,
        "closure_relative_tolerance": closure_tolerance,
        "checks": checks,
        "baseline_memory": baseline_memory,
        "diagnostic_memory": diagnostic_memory,
        "memory_scope": "Torch allocations; both model/optimizer copies are resident; not a throughput benchmark",
        "gradient_diagnostic": record,
    }


def test_diagnostic_preserves_checkpointed_gradients_update_and_nonempty_adamw_state():
    report = run_equivalence(torch.device("cpu"))
    assert report["passed"], report["checks"]


def test_zero_gradient_cosines_are_null_even_with_adamw_weight_decay():
    torch.manual_seed(91)
    model, cfg = (
        DiffNavPolicy(),
        DiffNavTrainCfg(physics_dt=0.01, policy_dt=0.1, variant="step", velocity_loss_mode="variant_default"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    zero = sum(parameter.sum() for parameter in model.parameters()) * 0.0
    losses = {name: zero for name, _ in LOSS_COEFFICIENTS}
    losses["total"] = sum(getattr(cfg, attribute) * losses[name] for name, attribute in LOSS_COEFFICIENTS)
    optimizer.zero_grad(set_to_none=True)
    diagnostic = GradientDiagnostic(model, losses, cfg)
    assert all(parameter.grad is None for parameter in model.parameters())
    losses["total"].backward()
    diagnostic.check_total_gradient()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    optimizer.step()
    record = diagnostic.finish(optimizer, norm, cfg.max_grad_norm)
    assert record["total_gradient_check"]["l2_error"] == 0
    assert record["groups"]["all"]["update_norm"] > 0  # Decoupled weight decay remains active.
    for group in record["groups"].values():
        assert group["raw_grad_norm"] == [0.0] * 7
        assert group["cosine"] == [[None] * 7 for _ in range(7)]
        assert group["cosine_with_actual_descent"] == [None] * 7
        assert group["linearized_delta_loss"] == [0.0] * 7
        assert group["height_weight_threshold_raw_gradient"] is None
    json.dumps(record, allow_nan=False)


def test_diagnostic_update_config_defaults_off_and_rejects_invalid_global_ids():
    assert (
        DiffNavTrainCfg(
            physics_dt=0.01, policy_dt=0.1, variant="step", velocity_loss_mode="variant_default"
        ).gradient_diagnostic_updates
        == ()
    )
    DiffNavTrainCfg(
        physics_dt=0.01,
        policy_dt=0.1,
        variant="step",
        velocity_loss_mode="variant_default",
        gradient_diagnostic_updates=(20, 22, 24, 26),
    ).validate()
    for updates in ((0,), (-1,), (1, 1), (1.0,), (True,)):
        with pytest.raises(ValueError, match="gradient_diagnostic_updates"):
            DiffNavTrainCfg(
                physics_dt=0.01,
                policy_dt=0.1,
                variant="step",
                velocity_loss_mode="variant_default",
                gradient_diagnostic_updates=updates,
            ).validate()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Small checkpointed gradient-diagnostic equivalence smoke")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new file")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("--device must select CPU or CUDA")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = run_equivalence(device)
    except Exception as error:
        report = {"passed": False, "device": str(device), "error": f"{type(error).__name__}: {error}"}
    finally:
        torch.set_num_threads(previous_threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, allow_nan=False, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "gradient_diagnostic"}, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


def test_bootstrap_diagnostics_measure_the_actual_objective_without_changing_gradients():
    model = DiffNavPolicy()
    cfg = DiffNavTrainCfg(algorithm="shac", gradient_diagnostic_updates=(1,))
    cfg.validate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    signal = sum(parameter.square().sum() for parameter in model.parameters())
    parts = {"discounted_cost": signal, "bootstrap": -0.4 * signal}
    diagnostic = GradientDiagnostic(model, {}, cfg, components=parts)
    sum(parts.values()).backward()
    diagnostic.check_total_gradient()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    record = diagnostic.finish(optimizer, norm, 5)
    assert record["total_gradient_check"]["relative_l2_error"] < 1e-6
    assert record["groups"]["all"]["cosine"][0][1] == pytest.approx(-1)
    assert record["groups"]["all"]["linearized_delta_loss"][0] < 0
    assert record["groups"]["all"]["linearized_delta_loss"][1] > 0


def test_critic_vertical_partials_include_physical_units_and_match_finite_difference():
    from aerialnav.diffnav.diagnostics import critic_vertical_probe

    class AnalyticCritic:
        def actor_value(self, features):
            return 2 * features[:, 2] + 3 * features[:, 5]

    features = torch.randn(3, 6, dtype=torch.double)
    result = critic_vertical_probe(
        AnalyticCritic(), features, torch.tensor([2.0, 3.7, 4.1]), torch.tensor([False, False, True])
    )
    assert result["dV_dz"] == pytest.approx([-0.04] * 3)
    assert result["dV_dvz"] == pytest.approx([0.3] * 3)
    assert result["fd_max_abs_error_dz"] < 1e-12
    assert result["fd_max_abs_error_dvz"] < 1e-12
