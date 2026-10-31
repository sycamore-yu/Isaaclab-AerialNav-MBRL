"""Raw height projection and complete optimizer probes must have distinct guarantees."""

import copy
import json

import pytest
import torch

from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.diffnav.diagnostics import LOSS_COEFFICIENTS, GradientDiagnostic
from aerialnav.diffnav.model import DiffNavPolicy
from aerialnav.diffnav.projection import ActionUpdateProbe, height_priority_gradient


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _flat(values):
    return torch.cat([value.detach().reshape(-1) for value in values])


def _assert_equal_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal_tree(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected, strict=True):
            _assert_equal_tree(first, second)
    else:
        assert actual == expected


def _inputs(device):
    generator = torch.Generator().manual_seed(81)
    heading = torch.tensor(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    return {
        "depth": torch.randn(2, 1, 12, 16, generator=generator).to(device),
        "features": torch.randn(2, 10, generator=generator).to(device),
        "hidden": torch.randn(2, 192, generator=generator).to(device),
        "heading": heading,
        "thrust_error": torch.tensor([[0.9], [1.1]], device=device),
    }


def _output(model, inputs):
    return model(inputs["depth"], inputs["features"], inputs["hidden"])[0]


def test_projection_removes_only_the_opposing_component_and_never_mutates_inputs():
    height = torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64, requires_grad=True)
    total = torch.tensor([-1.5, -1.0, 3.0], dtype=torch.float64, requires_grad=True)
    before_total, before_height = total.detach().clone(), height.detach().clone()
    projected, metrics = height_priority_gradient(total, height, 0.5)
    torch.testing.assert_close(projected, total.new_tensor([-0.3, 1.4, 3.0]), rtol=0, atol=1e-14)
    other_after = projected - 0.5 * height.detach()
    assert abs(float(other_after @ height.detach())) < 1e-14
    assert float(projected @ height.detach()) >= 2.5 - 1e-14
    # A perpendicular direction is not altered by the projection.
    perpendicular = total.new_tensor([2.0, -1.0, 1.0])
    torch.testing.assert_close(projected @ perpendicular, total.detach() @ perpendicular, rtol=0, atol=1e-14)
    torch.testing.assert_close(total, before_total, rtol=0, atol=0)
    torch.testing.assert_close(height, before_height, rtol=0, atol=0)
    assert not projected.requires_grad and projected.data_ptr() != total.data_ptr()
    assert metrics["applied"] and not metrics["height_gradient_zero"]
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("dtype,tiny", ((torch.float32, 1e-40), (torch.float64, 1e-200)))
def test_projection_handles_tiny_exponential_gradients_without_coefficient_overflow(dtype, tiny):
    total = torch.tensor([-1.0, 2.0], dtype=dtype)
    height = torch.tensor([tiny, 0.0], dtype=dtype)
    projected, metrics = height_priority_gradient(total, height, 1.0)
    torch.testing.assert_close(projected, torch.tensor([tiny, 2.0], dtype=dtype), rtol=0, atol=0)
    assert metrics["applied"] and metrics["height_gradient_norm"] > 0
    assert bool(torch.isfinite(projected).all())
    json.dumps(metrics, allow_nan=False)


def test_projection_is_identity_for_zero_height_or_aligned_other_gradient():
    total = torch.tensor([3.0, -2.0])
    for height in (torch.zeros_like(total), torch.tensor([1.0, 0.0])):
        projected, metrics = height_priority_gradient(total, height, 1.0)
        torch.testing.assert_close(projected, total, rtol=0, atol=0)
        assert projected.data_ptr() != total.data_ptr() and not metrics["applied"]
    with pytest.raises(ValueError, match="finite input"):
        height_priority_gradient(torch.tensor([float("nan")]), torch.ones(1), 1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        height_priority_gradient(torch.ones(1), torch.ones(1), -1.0)


def test_adam_momentum_can_reverse_the_raw_height_projection_guarantee():
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, betas=(0.9, 0.999), weight_decay=0.0)
    parameter.grad = torch.full_like(parameter, -10.0)
    optimizer.step()  # A previous gradient leaves a negative first moment.
    before = parameter.detach().clone()
    projected, metrics = height_priority_gradient(
        torch.tensor([-1.0], dtype=parameter.dtype), torch.ones_like(parameter), 1.0
    )
    assert metrics["applied"] and metrics["height_dot_total_after"] > 0
    parameter.grad = projected
    torch.nn.utils.clip_grad_norm_([parameter], 5.0)
    optimizer.step()
    # Positive raw height gradient would decrease height under plain SGD, but
    # this inherited Adam first moment still increases the local height loss.
    assert float((parameter.detach() - before).sum()) > 0


@pytest.mark.parametrize("optimizer_type", (torch.optim.Adam, torch.optim.AdamW))
@pytest.mark.parametrize("applied_projection", ("none", "height_priority"))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_probe_preserves_real_state_and_rng_and_matches_actual_complete_step(
    optimizer_type,
    applied_projection,
    device,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(73)
    model = DiffNavPolicy().to(device)
    optimizer = optimizer_type(model.parameters(), lr=0.0005, betas=(0.9, 0.99), weight_decay=0.02, eps=1e-7)
    inputs = _inputs(device)
    # Use nonempty, nontrivial Adam moments. Clipping is active in both the
    # warmup and the measured full optimizer step.
    optimizer.zero_grad(set_to_none=True)
    output = _output(model, inputs)
    (output.square().mean() - output[:, 4].mean()).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output = _output(model, inputs)
    height_loss = output[:, 4].mean()
    height_flat = _flat(torch.autograd.grad(height_loss, tuple(model.parameters()), retain_graph=True))
    (-2.0 * height_loss + 0.01 * output[:, 0].mean()).backward()
    total_flat = _flat(parameter.grad for parameter in model.parameters())
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
    original_total, original_height = total_flat.clone(), height_flat.clone()
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device == "cuda" else None
    probe = ActionUpdateProbe(
        model,
        optimizer,
        total_flat=total_flat,
        height_flat=height_flat,
        height_weight=1.0,
        max_grad_norm=0.01,
        probe=inputs,
    )
    _assert_equal_tree(model.state_dict(), expected_model)
    _assert_equal_tree(optimizer.state_dict(), expected_optimizer)
    _assert_equal_tree([parameter.grad for parameter in model.parameters()], expected_gradients)
    _assert_equal_tree(torch.get_rng_state(), rng)
    if cuda_rng is not None:
        _assert_equal_tree(torch.cuda.get_rng_state(device), cuda_rng)
    _assert_equal_tree(total_flat, original_total)
    _assert_equal_tree(height_flat, original_height)
    assert model.training
    # Environment-owned observation buffers may change after capture. The
    # completed-step check must still use the immutable old probe inputs.
    inputs["features"].add_(100.0)
    inputs["hidden"].zero_()
    if applied_projection == "height_priority":
        selected, _ = height_priority_gradient(total_flat, height_flat, 1.0)
        offset = 0
        for parameter in model.parameters():
            end = offset + parameter.numel()
            parameter.grad.copy_(selected[offset:end].view_as(parameter))
            offset = end
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, error_if_nonfinite=True)
    assert norm > 0.01
    optimizer.step()
    after_model = copy.deepcopy(model.state_dict())
    after_optimizer = copy.deepcopy(optimizer.state_dict())
    after_gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
    record = probe.finish(model, applied_projection=applied_projection)
    _assert_equal_tree(model.state_dict(), after_model)
    _assert_equal_tree(optimizer.state_dict(), after_optimizer)
    _assert_equal_tree([parameter.grad for parameter in model.parameters()], after_gradients)
    _assert_equal_tree(torch.get_rng_state(), rng)
    if cuda_rng is not None:
        _assert_equal_tree(torch.cuda.get_rng_state(device), cuda_rng)
    assert model.training and record["num_probes"] == 2
    assert record["projection"]["applied"]
    assert record["branches"]["original"]["clip_scale"] < 1
    assert record["branches"]["height_priority"]["clip_scale"] < 1
    tolerance = 1e-6 if device == "cuda" else 0.0
    assert record["actual_step"]["parameter_max_abs_error"] <= tolerance
    assert record["actual_step"]["command_match_error"]["max_abs"] <= tolerance
    branch = "original" if applied_projection == "none" else "height_priority"
    assert record["actual_step"]["height_gradient_dot_parameter_delta"] == pytest.approx(
        record["branches"][branch]["height_gradient_dot_parameter_delta"],
        rel=1e-5,
        abs=tolerance,
    )
    assert record["diagnostic_s"] >= 0
    json.dumps(record, allow_nan=False)


def test_diagnostic_component_gradient_is_an_independent_detached_copy():
    torch.manual_seed(17)
    model = DiffNavPolicy()
    output = _output(model, _inputs("cpu"))
    cfg = DiffNavTrainCfg()
    losses = {name: (output + index * 0.1).square().mean() for index, (name, _) in enumerate(LOSS_COEFFICIENTS)}
    losses["total"] = sum(getattr(cfg, attribute) * losses[name] for name, attribute in LOSS_COEFFICIENTS)
    expected = _flat(torch.autograd.grad(losses["height_boundary"], tuple(model.parameters()), retain_graph=True))
    diagnostic = GradientDiagnostic(model, losses, cfg)
    first = diagnostic.component_gradient("height_boundary")
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    assert not first.requires_grad and all(parameter.grad is None for parameter in model.parameters())
    first.zero_()
    torch.testing.assert_close(diagnostic.component_gradient("height_boundary"), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Unknown gradient component"):
        diagnostic.component_gradient("unknown")
