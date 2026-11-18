"""Tests for v4 Kinematic Event Field, Zero-Init Dual Stream, Inv-TTC Aux, and BPTT Dynamic Losses."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from isaaclab_navrl.tasks.navigation.context import ObservationContext
from isaaclab_navrl.tasks.navigation.depth_history import (
    KINEMATIC_EVENT_FIELD_OBSERVATION_CONTRACT,
    pooled_event_channels,
    reproject_depth,
    reproject_event_channels,
)
from isaaclab_navrl.tasks.navigation.geometric_depth_events import DepthEventDetector
from isaaclab_navrl.zhang.aux import (
    DEFAULT_V4_AUX_HALF_FOV_RAD,
    dynamic_aux_loss,
    ego_dynamic_sector_targets,
    forward_closing_risk,
)
from isaaclab_navrl.zhang.cli import parse_train_args
from isaaclab_navrl.zhang.config import ZhangTrainCfg
from isaaclab_navrl.zhang.inference import ZhangInference
from isaaclab_navrl.zhang.losses import zhang_transition_costs
from isaaclab_navrl.zhang.model import ZhangPolicy
from isaaclab_navrl.zhang.normalization import PolicyObservationNormalizer


def _camera(n=1, *, x=0.0):
    pose = torch.tensor([x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32).expand(n, -1).clone()
    k = torch.tensor([[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]]).expand(n, -1, -1).clone()
    return pose, k


def test_v4_kinematic_field_encodes_eg_ts_wedges_looming_and_matches_cuda():
    k = torch.tensor([[[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]]]).expand(2, -1, -1).clone()
    det_cpu = DepthEventDetector(
        48, 64, max_depth=24.0, history_frames=4, threshold=0.0, relative_threshold=0.002,
        depth_sigma=0.02, pose_position_sigma=0.01, pose_rotation_sigma=0.002,
        sigma_multiplier=2.5, kinematic_field=True,
    )
    det_gpu = (
        DepthEventDetector(
            48, 64, max_depth=24.0, history_frames=4, threshold=0.0, relative_threshold=0.002,
            depth_sigma=0.02, pose_position_sigma=0.01, pose_rotation_sigma=0.002,
            sigma_multiplier=2.5, kinematic_field=True,
        )
        if torch.cuda.is_available()
        else None
    )
    k_gpu = k.cuda() if det_gpu is not None else None
    for step in range(8):
        pose = torch.tensor([[step * 0.05, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]).expand(2, -1).clone()
        d = torch.full((2, 48, 64), 8.0)
        d[:, 10:25, 15 + 2 * step : 27 + 2 * step] = 4.0
        d[:, 28:40, 40:52] = max(2.0, 6.0 - 0.15 * step)
        rc = det_cpu.update(d, pose, k, step * 0.05)
        kc = rc.kinematic_image(pose, k, 24.0)
        if det_gpu is not None:
            pose_gpu = pose.cuda()
            rg = det_gpu.update(d.cuda(), pose_gpu, k_gpu, step * 0.05)
            kg = rg.kinematic_image(pose_gpu, k_gpu, 24.0).cpu()
            torch.testing.assert_close(kg, kc, atol=1e-4, rtol=1e-4)
    assert float(kc[:, 0].max()) > 0.3
    assert float(kc[:, 1].max()) > 0.3
    assert float(kc[:, 2].max()) > 0.1


def test_v4_zero_init_residual_stream_preserves_depth_only_init_and_decouples_aux():
    torch.manual_seed(11)
    baseline = ZhangPolicy().eval()
    torch.manual_seed(11)
    v4 = ZhangPolicy().eval()
    v4.enable_kinematic_residual_stream(3)
    v4.enable_dynamic_aux_head(num_sectors=6, from_perception=True)

    # 24x32 inv-depth whose 2x2 max-pool equals 12x16 base_depth.
    raw = torch.rand(4, 1, 120, 160) * 10.0 + 1.0
    inv24 = F.adaptive_max_pool2d(3 / raw.clamp(0.3, 24) - 0.6, (24, 32))
    inv12 = F.adaptive_max_pool2d(3 / raw.clamp(0.3, 24) - 0.6, (12, 16))
    kin = torch.randn(4, 3, 24, 32)
    x4 = torch.cat((inv24, kin), dim=1)
    state = torch.randn(4, 10)

    out_base, h_base = baseline(inv12, state)
    out_v4, h_v4, aux_repr = v4.forward_with_aux(x4, state)
    torch.testing.assert_close(out_v4, out_base, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(h_v4, h_base, atol=1e-6, rtol=1e-6)

    # Aux gradient updates kinematic_stem without flowing through GRUCell.
    v4.train()
    _, h2, aux_repr2 = v4.forward_checkpointed_with_aux(x4, state, None)
    pred_aux = v4.predict_dynamic_aux(aux_repr2)
    assert pred_aux.shape == (4, 6, 3)
    loss = dynamic_aux_loss(pred_aux, torch.rand_like(pred_aux))
    loss.backward()
    assert v4.kinematic_stem[0].weight.grad.abs().sum() > 0
    assert v4.gru.weight_ih.grad is None


def test_v4_los_inv_ttc_aux_and_forward_closing_risk():
    dp = torch.zeros(2, 3)
    dv = torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    hd = torch.eye(3)[None].expand(2, -1, -1).contiguous()
    # Obstacle 0 at 4m approaching (visible), Obstacle 1 at 8m faster behind wall (occluded at 4.0m in depth_map).
    op = torch.tensor([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    ov = torch.tensor([[-2.0, 0.0, 0.0], [-8.0, 0.0, 0.0]])
    osz = torch.tensor([[0.5, 0.5, 2.0], [0.5, 0.5, 2.0]])
    dmap = torch.full((2, 1, 24, 32), 4.0)
    targets = ego_dynamic_sector_targets(
        dp, dv, hd, op, ov, osz,
        num_sectors=6, half_fov_rad=DEFAULT_V4_AUX_HALF_FOV_RAD, inv_ttc_mode=True, depth_map=dmap,
    )
    assert targets.shape == (2, 6, 3)
    # Angle=0 lands in sector 3 and must reflect visible Obstacle 0 (~0.4267), not occluded Obstacle 1 (~0.516).
    assert float(targets[0, 3, 0]) == pytest.approx(4.0 / (3.75 * 2.5), rel=1e-3)
    risk = forward_closing_risk(dp, dv, hd, op, ov, osz)
    assert (risk > 0.2).all()
    if torch.cuda.is_available():
        tg = ego_dynamic_sector_targets(
            dp.cuda(), dv.cuda(), hd.cuda(), op.cuda(), ov.cuda(), osz.cuda(),
            num_sectors=6, half_fov_rad=DEFAULT_V4_AUX_HALF_FOV_RAD, inv_ttc_mode=True, depth_map=dmap.cuda(),
        )
        rg = forward_closing_risk(dp.cuda(), dv.cuda(), hd.cuda(), op.cuda(), ov.cuda(), osz.cuda())
        torch.testing.assert_close(tg.cpu(), targets, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(rg.cpu(), risk, atol=1e-5, rtol=1e-5)


def test_v4_smooth_approach_gate_reduces_receding_penalty():
    t, n = 4, 2
    receding_dist = torch.linspace(0.4, 1.4, 10)[None, None, :].expand(t, n, -1).clone()
    history = {
        "velocity": [torch.zeros(n, 3) for _ in range(t)],
        "target_velocity": [torch.zeros(n, 3) for _ in range(t)],
        "episode_id": [torch.zeros(n, dtype=torch.long) for _ in range(t)],
        "margin": [torch.zeros(n) for _ in range(t)],
        "obstacle_signed_distance": [receding_dist[i] for i in range(t)],
        "collision_gap": [receding_dist[i, :, 1:] for i in range(t)],
        "command": [torch.zeros(n, 3) for _ in range(t)],
        "previous_command": [torch.zeros(n, 3) for _ in range(t)],
        "position": [torch.ones(n, 3) for _ in range(t)],
        "height_range": [torch.tensor([[0.5, 2.5]]).expand(n, -1) for _ in range(t)],
        "velocity_prediction": [torch.zeros(n, 3) for _ in range(t)],
        "observed_velocity": [torch.zeros(n, 3) for _ in range(t)],
    }
    cfg_old = ZhangTrainCfg(velocity_loss_mode="per_step", smooth_approach_gate=False)
    cfg_v4 = ZhangTrainCfg(velocity_loss_mode="per_step", smooth_approach_gate=True)
    losses_old, _, _ = zhang_transition_costs(history, cfg_old)
    losses_v4, _, _ = zhang_transition_costs(history, cfg_v4)
    assert float(losses_v4["obstacle_avoidance"]) < float(losses_old["obstacle_avoidance"]) * 0.2


def test_v4_and_stack_cli_and_inference_roundtrip(tmp_path):
    parsed = parse_train_args([
        "--depth_event_channel", "1",
        "--depth_event_kinematic_field", "1",
        "--depth_event_history_frames", "4",
        "--coef_dyn_aux", "0.5",
        "--dyn_aux_sectors", "6",
        "--aux_from_perception", "1",
        "--aux_inv_ttc", "1",
        "--aux_survival_gate_s", "5.0",
        "--smooth_approach_gate", "1",
        "--temporal_yielding_lambda", "1.5",
        "--active_yaw_alignment", "1.0",
    ])
    assert parsed["depth_event_kinematic_field"] is True
    assert parsed["dyn_aux_sectors"] == 6
    assert parsed["temporal_yielding_lambda"] == 1.5

    # Also test v4 kinematic event field contract + ZhangInference load.
    v4_parsed = parse_train_args(
        ["--depth_event_channel", "1", "--depth_event_history_frames", "4", "--coef_dyn_aux", "0.5"]
    )
    assert v4_parsed["depth_event_channel"] is True
    model = ZhangPolicy()
    model.enable_kinematic_residual_stream(3)
    model.enable_dynamic_aux_head(num_sectors=6, from_perception=True)
    cfg = ZhangTrainCfg(
        depth_event_channel=True,
        depth_event_history_frames=4,
        depth_event_kinematic_field=True,
        coef_dyn_aux=0.5,
        dyn_aux_sectors=6,
        aux_from_perception=True,
        aux_inv_ttc=True,
        goal_speed_profile="linear",
    )
    cfg.validate()
    norm = PolicyObservationNormalizer(enabled=True)
    x = torch.randn(2, 4, 24, 32)
    s = torch.randn(2, 10)
    norm.observe(x, s)
    nx, _ = norm.normalize(x, s, norm.begin_rollout())
    torch.testing.assert_close(nx[:, 1:], x[:, 1:])

    ckpt = tmp_path / "v4_kinematic.pt"
    torch.save(
        {
            "format_version": 3,
            "method": "zhang",
            "environment": {"observation_contract": KINEMATIC_EVENT_FIELD_OBSERVATION_CONTRACT},
            "train_cfg": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "normalizer_state_dict": norm.state_dict(),
        },
        ckpt,
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        cfg=SimpleNamespace(
            depth_event_channel=True,
            depth_event_history_frames=4,
            depth_event_kinematic_field=True,
            sim_to_real=SimpleNamespace(enabled=False, input_delay=False),
        ),
        zhang_parameters={
            "max_speed": torch.full((2, 1), 3.0),
            "margin": torch.full((2,), 0.3),
            "thr_est_error": torch.ones((2, 1)),
        },
        get_zhang_observation=lambda: ObservationContext.build(
            policy={
                "position": torch.zeros(2, 3),
                "velocity": torch.zeros(2, 3),
                "quaternion": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, -1),
                "target": torch.tensor([[8.0, 0.0, 1.0]]).expand(2, -1),
                "depth_features": torch.zeros(2, 4, 24, 32),
            },
            privileged={
                "episode_id": torch.zeros(2, dtype=torch.long),
                "position": torch.full((2, 3), float("nan")),
                "target": torch.full((2, 3), float("nan")),
            },
        ),
    )
    inf = ZhangInference(env, ckpt)
    act = inf()
    assert act.shape == (2, 3)
    assert torch.isfinite(act).all()


def test_reproject_and_pool_3ch_kinematic_channels():
    d = torch.full((1, 1, 24, 32), 4.0)
    kin = torch.zeros((1, 3, 24, 32))
    kin[:, 0, 4:10, 8:14] = 0.8
    kin[:, 1, 12:18, 8:14] = 0.6
    kin[:, 2, 4:10, 18:24] = 0.4
    k = torch.tensor([[[16.0, 0.0, 16.0], [0.0, 16.0, 12.0], [0.0, 0.0, 1.0]]])
    rot = torch.eye(3)[None]
    pos0 = torch.zeros(1, 3)
    pos1 = torch.tensor([[0.5, 0.0, 0.0]])
    ad, kn = reproject_depth(d, k, rot, pos0, rot, pos1)
    reproj = reproject_event_channels(d, kin, k, rot, pos0, rot, pos1, ad, kn)
    pooled = pooled_event_channels(ad, reproj, kn, output_size=(24, 32), region_pool=True)
    assert pooled.shape == (1, 3, 24, 32)
    assert pooled[:, 0].sum() > 0 and pooled[:, 1].sum() > 0 and pooled[:, 2].sum() > 0
