"""Lossless valid-only encoding, including empty clouds and MaxPool gradient ties."""

import copy

import pytest
import torch

from aerialnav.pointnet.config import CHECKPOINT_METHODS, PointNetTrainCfg
from aerialnav.pointnet.model import PointNetPolicy
from aerialnav.pointnet.points import compact_points, segment_amax


def dense_encode(model, points):
    valid = points[:, -1:] > 0
    features = model.point_encoder(torch.where(valid, points[:, : model.channels], 0))
    pooled = features.masked_fill(~valid, torch.finfo(features.dtype).min).amax(-1)
    return model.perception_projection(torch.where(valid.any(-1), pooled, 0))


@pytest.mark.parametrize("fraction", [0.0, 0.18, 1.0])
@pytest.mark.parametrize("tied", [False, True])
def test_compaction_preserves_all_returns_outputs_and_input_parameter_gradients(fraction, tied):
    torch.manual_seed(11)
    packed_model = PointNetPolicy()
    dense_model = copy.deepcopy(packed_model)
    points = torch.randn(4, 4, 37)
    if tied:
        points[:, :3] = points[:, :3, :1]
    points[:, -1] = (torch.rand(4, 37) < fraction).float()
    points[0, -1] = 0
    points[:, :3] = torch.where(points[:, -1:] > 0, points[:, :3], torch.nan)
    packed_input = points.clone().requires_grad_()
    dense_input = points.clone().requires_grad_()
    compact, counts, offsets = compact_points(packed_input)
    assert compact.shape[0] == int((points[:, -1] > 0).sum()) == offsets[-1]
    torch.testing.assert_close(compact, points[:, :3].transpose(1, 2)[points[:, -1] > 0])
    assert sum(counts.tolist()) == offsets[-1]
    expected, actual = dense_encode(dense_model, dense_input), packed_model.encode(packed_input)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(packed_input.grad, dense_input.grad, atol=2e-6, rtol=2e-4)
    for (name, a), (_, b) in zip(packed_model.named_parameters(), dense_model.named_parameters(), strict=True):
        if a.grad is not None or b.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4, msg=name)


def test_checkpoint_does_not_recompute_recurrent_head_and_keeps_actions_fp32():
    policy = PointNetPolicy()
    policy.configure_execution(
        PointNetTrainCfg(point_microbatch_envs=32, point_precision="bf16", point_encoder_backend="torch")
    )
    points = torch.randn(2, 4, 37)
    points[:, -1] = 1
    state = torch.randn(2, 10, requires_grad=True)
    hidden = torch.zeros(2, 192)
    calls = []
    hook = policy.gru.register_forward_hook(lambda *args: calls.append(1))
    for _ in range(3):
        output, hidden = policy.forward_checkpointed(points, state, hidden)
    from aerialnav.diffnav.dynamics import PointMassState, policy_transition
    from aerialnav.pointnet.losses import pointcloud_transition_costs

    zero = torch.zeros(2, 3)
    cfg = PointNetTrainCfg(point_precision="bf16", point_encoder_backend="torch")
    parameters = {
        "wind": zero,
        "dg": zero,
        "z_drag_coef": torch.ones(2, 1),
        "drag_2": torch.zeros(2, 2),
        "pitch_ctl_delay": torch.full((2, 1), 10.0),
    }
    transition = policy_transition(
        PointMassState(zero, zero, zero, zero), output, torch.eye(3).expand(2, 3, 3), parameters, cfg
    )
    assert all(
        getattr(transition, key).dtype == torch.float32 for key in ("position", "velocity", "acceleration", "actuator")
    )
    history = {
        "velocity": [transition.velocity],
        "position": [transition.position],
        "command": [output],
        "previous_command": [zero],
        "episode_id": [torch.zeros(2, dtype=torch.long)],
        "episode_step": [torch.ones(2, dtype=torch.long)],
        "target_velocity": [torch.ones(2, 3)],
        "obstacle_signed_distance": [torch.ones(2, 10)],
        "margin": [torch.full((2,), 0.3)],
        "height_range": [torch.tensor([[0.5, 2.5]]).expand(2, 2)],
    }
    losses, _, costs = pointcloud_transition_costs(history, cfg)
    assert costs.dtype == torch.float32 and all(x.dtype == torch.float32 for x in losses.values())
    losses["total"].backward()
    hook.remove()
    assert len(calls) == 3
    assert output.dtype == hidden.dtype == state.grad.dtype == torch.float32
    assert all(
        p.dtype == torch.float32 and (p.grad is None or p.grad.dtype == torch.float32) for p in policy.parameters()
    )
    assert policy.execution_metrics()["point_encoder_calls"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Triton kernel")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_affine_forward_and_upstream_parameter_gradients(dtype):
    from aerialnav.pointnet.triton_linear import linear_128_1024

    torch.manual_seed(19)
    x = torch.randn(17, 128, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(1024, 128, device="cuda", requires_grad=True)
    bias = torch.randn(1024, device="cuda", requires_grad=True)
    expected = torch.nn.functional.linear(x, weight.to(dtype), bias.to(dtype))
    actual = linear_128_1024(x, weight, bias)
    tolerance = 0.02 if dtype == torch.bfloat16 else 3e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    upstream = torch.randn_like(actual)
    a = torch.autograd.grad(actual, (x, weight, bias), upstream)
    b = torch.autograd.grad(expected, (x, weight, bias), upstream)
    for actual_grad, expected_grad in zip(a, b, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    "values", [{"point_microbatch_envs": 0}, {"point_precision": "fp16"}, {"point_encoder_backend": "unknown"}]
)
def test_invalid_execution_settings_fail_explicitly(values):
    with pytest.raises(ValueError):
        PointNetTrainCfg(**values).validate()


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))]
)
def test_segment_max_splits_tied_negative_upstream_gradients(device):
    x = torch.ones(4, 2, device=device, requires_grad=True)
    y = segment_amax(x, torch.tensor([4], device=device))
    y.backward(torch.tensor([[-1.0, 2.0]], device=device))
    torch.testing.assert_close(x.grad, torch.tensor([[-0.25, 0.5]], device=device).expand(4, -1))


def test_research_selective_winner_recompute_preserves_ties_and_negative_gradients():
    """Research-only proof; the production path still recomputes the full packed encoder."""
    torch.manual_seed(31)
    dense_model = PointNetPolicy()
    selected_model = copy.deepcopy(dense_model)
    points = torch.randn(2, 9, 3)
    points[:, 1] = points[:, 0]
    dense_input = points.clone().requires_grad_()
    selected_input = points.clone().requires_grad_()

    def features(model, x):
        for layer in model.point_encoder:
            x = (
                torch.nn.functional.linear(x, layer.weight[..., 0], layer.bias)
                if isinstance(layer, torch.nn.Conv1d)
                else layer(x)
            )
        return x

    expected = features(dense_model, dense_input).amax(1)
    with torch.no_grad():
        full = features(selected_model, selected_input)
        maxima = full.amax(1)
        winners = full == maxima[:, None]
        coefficients = winners / winners.sum(1, keepdim=True)
        keep = winners.any(-1)
        owners = torch.arange(2)[:, None].expand(2, 9)[keep]
    selected = features(selected_model, selected_input[keep])
    surrogate = selected.new_zeros(2, 1024).index_add(0, owners, selected * coefficients[keep])
    actual = maxima + (surrogate - surrogate.detach())
    upstream = torch.randn(2, 1024)
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(selected_input.grad, dense_input.grad, atol=2e-5, rtol=2e-4)
    for a, b in zip(selected_model.point_encoder.parameters(), dense_model.point_encoder.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compiler")
def test_compiled_bf16_encoder_handles_empty_dense_and_tied_scans_with_fp32_actions():
    torch.manual_seed(43)
    eager = PointNetPolicy().cuda()
    compiled = copy.deepcopy(eager)
    eager.configure_execution(PointNetTrainCfg(point_precision="bf16", point_encoder_backend="torch"))
    compiled.configure_execution(PointNetTrainCfg(point_precision="bf16", point_encoder_backend="compile"))
    for fraction in (0.0, 0.18, 1.0):
        points = torch.randn(3, 4, 39, device="cuda")
        points[:, :3] = points[:, :3, :1]
        points[:, -1] = (torch.rand(3, 39, device="cuda") < fraction).float()
        points[:, :3] = torch.where(points[:, -1:] > 0, points[:, :3], torch.nan)
        state = torch.randn(3, 10, device="cuda")
        eager.zero_grad(set_to_none=True)
        compiled.zero_grad(set_to_none=True)
        a, _ = eager(points, state)
        b, _ = compiled(points, state)
        assert b.dtype == torch.float32 and torch.isfinite(b).all()
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
        a.square().sum().backward()
        b.square().sum().backward()
        for x, y in zip(eager.parameters(), compiled.parameters(), strict=True):
            torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-3)


def test_compaction_never_truncates_a_fully_valid_5400_ray_scan():
    points = torch.arange(4 * 5400, dtype=torch.float32).reshape(1, 4, 5400)
    points[:, -1] = 1
    packed, counts, offsets = compact_points(points)
    assert packed.shape == (5400, 3) and counts.tolist() == [5400] and offsets == [0, 5400]
    torch.testing.assert_close(packed[-1], points[0, :3, 5399])


@pytest.mark.parametrize("method", CHECKPOINT_METHODS)
def test_legacy_checkpoint_inference_keeps_fp32_without_compilation(tmp_path, method):
    from dataclasses import asdict
    from types import SimpleNamespace

    from aerialnav.diffnav.normalization import PolicyObservationNormalizer
    from aerialnav.pointnet.inference import load_pointnet_policy

    settings = asdict(PointNetTrainCfg())
    for name in PointNetTrainCfg.resume_runtime_fields:
        del settings[name]
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "method": method,
            "format_version": 3,
            "train_cfg": settings,
            "model_state_dict": PointNetPolicy().state_dict(),
            "normalizer_state_dict": PolicyObservationNormalizer(False, 1e-4).state_dict(),
        },
        path,
    )
    import json

    from isaaclab_tasks.utils.hydra import resolve_presets

    from aerialnav.pointnet.checkpoint import VERIFIED_LEGACY_SOURCE, policy_contract
    from aerialnav.tasks.navigation.config.hummingbird.env_cfg import DiffNavEnvCfg

    cfg = DiffNavEnvCfg()
    resolve_presets(cfg, selected=("pointnet", "newton_mjwarp"))
    cfg.configure_scene()
    cfg.sim_to_real.enabled = True
    cfg.sim_to_real.input_delay = True
    env = SimpleNamespace(device="cpu", num_envs=1, cfg=cfg)
    payload = torch.load(path, weights_only=True)
    if method == CHECKPOINT_METHODS[1]:
        source = tmp_path / "source"
        source.mkdir()
        (source / "source_identity.json").write_text(json.dumps({"commit": VERIFIED_LEGACY_SOURCE}))
        payload["stage_log_dir"] = str(tmp_path / "train")
        from aerialnav.pointnet.config import OBSERVATION_CONTRACTS

        payload["environment"] = {"observation_contract": OBSERVATION_CONTRACTS[1]}
    else:
        from aerialnav.pointnet.config import OBSERVATION_CONTRACTS

        payload["environment"] = {"observation_contract": OBSERVATION_CONTRACTS[0]}
        payload["policy_contract"] = policy_contract(env)
    torch.save(payload, path)
    inference = load_pointnet_policy(env, path)
    assert inference.cfg.point_microbatch_envs == 16
    assert inference.cfg.point_precision == "fp32" and inference.model._compiled_pool is None
