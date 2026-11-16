"""The extra input is causal, depth-aligned and initially policy-neutral."""

import math
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from isaaclab_navrl.tasks.navigation.context import ObservationContext
from isaaclab_navrl.tasks.navigation.depth_history import (
    TEMPORAL_DEPTH_EVENT_OBSERVATION_CONTRACT,
    pooled_signed_event_channel,
    reproject_depth,
    reproject_signed_event_channel,
)
from isaaclab_navrl.tasks.navigation.geometric_depth_events import DepthEventDetector
from isaaclab_navrl.tasks.navigation.navrl_env import NavRLEnv
from isaaclab_navrl.tasks.navigation.randomization import PolicyPerturbation, SimToRealCfg
from isaaclab_navrl.zhang.cli import parse_train_args
from isaaclab_navrl.zhang.config import ZhangTrainCfg
from isaaclab_navrl.zhang.inference import ZhangInference
from isaaclab_navrl.zhang.model import ZhangPolicy
from isaaclab_navrl.zhang.normalization import PolicyObservationNormalizer


def _camera(n=1, *, x=0.0, yaw=0.0):
    pose = (
        torch.tensor([x, 0, 0, 0, 0, math.sin(yaw / 2), math.cos(yaw / 2)], dtype=torch.float32)
        .expand(n, -1)
        .clone()
    )
    k = torch.tensor([[60.0, 0, 32.0], [0, 60.0, 24.0], [0, 0, 1.0]]).expand(n, -1, -1).clone()
    return pose, k


def test_depth_event_weights_start_as_identical_single_channel_policy():
    torch.manual_seed(27)
    baseline = ZhangPolicy().eval()
    torch.manual_seed(27)
    with_event = ZhangPolicy().eval()
    with_event.enable_depth_event_channel()
    depth = torch.randn(4, 1, 12, 16)
    state = torch.randn(4, 10)
    assert with_event.stem[0].in_channels == 2
    torch.testing.assert_close(with_event.stem[0].weight[:, :1], baseline.stem[0].weight)
    assert not with_event.stem[0].weight[:, 1].abs().any()
    a, h = baseline(depth, state)
    b, g = with_event(torch.cat((depth, torch.randn_like(depth)), dim=1), state)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(h, g, atol=0, rtol=0)
    with_event.train()
    b, _ = with_event(torch.cat((depth, torch.ones_like(depth)), dim=1), state)
    b.square().sum().backward()
    assert with_event.stem[0].weight.grad[:, 1].abs().sum() > 0


def test_detector_polarities_and_reset_without_dynamic_oracle():
    old_pose, k = _camera()
    image = torch.full((1, 48, 64), 8.0)
    near = image.clone()
    near[:, 15:35, 20:42] = 4.0
    sensor = DepthEventDetector(48, 64, max_depth=24.0)
    first = sensor.update(image, old_pose, k, 0.0)
    assert not first.signed_image(old_pose, k, 24.0).any()
    appearance = sensor.update(near, old_pose, k, 0.05)
    plus = appearance.signed_image(old_pose, k, 24.0)
    assert plus.shape == (1, 1, 48, 64)
    assert (plus > 0).sum() > 250
    assert not (plus < 0).any()
    disappearance = sensor.update(image, old_pose, k, 0.1)
    minus = disappearance.signed_image(old_pose, k, 24.0)
    assert (minus < 0).sum() > 250
    assert not (minus > 0).any()
    sensor.reset()
    reset = sensor.update(near, old_pose, k, 0.15)
    assert not reset.appeared.any()


def test_event_selects_the_nearest_surface_not_foreground_background_mismatch():
    depth = torch.full((1, 1, 120, 160), 5.0)
    signed = torch.zeros_like(depth)
    signed[..., 0, 0] = -1
    depth[..., 0, 1] = 1.0
    chosen = pooled_signed_event_channel(depth, signed)
    assert chosen.shape == (1, 1, 12, 16)
    assert chosen[..., 0, 0] == 0
    signed[..., 0, 1] = 1
    assert pooled_signed_event_channel(depth, signed)[..., 0, 0] == 1


def test_delayed_event_and_depth_are_warped_by_the_same_camera_motion():
    depth = torch.full((1, 1, 12, 16), 4.0)
    signed = torch.zeros_like(depth)
    signed[..., 4:8, 6:10] = 1.0
    intrinsics = torch.tensor([[[8.0, 0.0, 8.0], [0.0, 8.0, 6.0], [0.0, 0.0, 1.0]]])
    source_rotation = torch.eye(3)[None]
    source_position = torch.zeros(1, 3)
    target_position = torch.tensor([[0.5, 0, 0]])
    aligned_depth, known = reproject_depth(
        depth, intrinsics, source_rotation, source_position, source_rotation, target_position
    )
    aligned_event = reproject_signed_event_channel(
        depth, signed, intrinsics, source_rotation, source_position, source_rotation, target_position, aligned_depth, known
    )
    assert aligned_event.sum() > 0
    assert not aligned_event[..., -1].any()
    assert (aligned_event > 0).sum() <= signed.sum()


def test_normalizer_never_normalizes_sparse_event_polarity():
    depth = torch.rand(3, 1, 12, 16)
    evt = torch.zeros_like(depth)
    evt[:, :, 1:4, 2:5] = -0.75
    state = torch.randn(3, 10)
    normalizer = PolicyObservationNormalizer(enabled=True)
    normalizer.observe(torch.cat((depth, evt), 1), state)
    normalized, _ = normalizer.normalize(torch.cat((depth, evt), 1), state, normalizer.begin_rollout())
    torch.testing.assert_close(normalized[:, 1:], evt)


def test_new_cli_optin_does_not_mutate_default():
    assert parse_train_args(["--depth_event_channel", "1"]) == {"depth_event_channel": True}
    assert parse_train_args(["--depth_event_channel", "0"]) == {"depth_event_channel": False}
    assert parse_train_args(["--depth_event_history_frames", "4", "--depth_event_pose_source", "truth"]) == {
        "depth_event_history_frames": 4,
        "depth_event_pose_source": "truth",
    }
    assert ZhangTrainCfg().depth_event_pose_source == "estimated"
    with pytest.raises(ValueError):
        parse_train_args(["--depth_event_history_frames", "0"])
    with pytest.raises(ValueError):
        parse_train_args(["--depth_event_pose_source", "invalid"])


def test_event_checkpoint_loads_and_infers_with_new_observation_context(tmp_path):
    model = ZhangPolicy()
    model.enable_depth_event_channel()
    cfg = ZhangTrainCfg(depth_event_channel=True, depth_event_history_frames=4, goal_speed_profile="linear")
    normalizer = PolicyObservationNormalizer()
    checkpoint = tmp_path / "depth-event.pt"
    torch.save(
        {
            "format_version": 3,
            "method": "zhang",
            "environment": {"observation_contract": TEMPORAL_DEPTH_EVENT_OBSERVATION_CONTRACT},
            "train_cfg": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "normalizer_state_dict": normalizer.state_dict(),
        },
        checkpoint,
    )
    count = 2
    policy_obs = {
        "position": torch.tensor([[0.0, 0.0, 1.0]]).expand(count, -1),
        "velocity": torch.zeros(count, 3),
        "quaternion": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(count, -1),
        "target": torch.tensor([[9.0, 0.0, 1.0]]).expand(count, -1),
        "depth_features": torch.zeros(count, 2, 12, 16),
    }
    privileged_obs = {
        "episode_id": torch.zeros(count, dtype=torch.long),
        "position": torch.full((count, 3), float("nan")),
        "target": torch.full((count, 3), float("nan")),
    }
    context = ObservationContext.build(policy=policy_obs, privileged=privileged_obs)
    env = SimpleNamespace(
        num_envs=count,
        device="cpu",
        cfg=SimpleNamespace(
            depth_event_channel=True,
            depth_event_history_frames=4,
            sim_to_real=SimpleNamespace(enabled=False, input_delay=False),
        ),
        zhang_parameters={
            "max_speed": torch.full((count, 1), 3.0),
            "margin": torch.full((count,), 0.3),
            "thr_est_error": torch.ones((count, 1)),
        },
        get_zhang_observation=lambda: context,
    )
    inference = ZhangInference(env, checkpoint)
    action = inference()
    assert action.shape == (count, 3)
    assert torch.isfinite(action).all()


def _make_mock_env(*, sim_cfg, pose_source="estimated", history_frames=1):
    height, width = 24, 32
    raw = torch.full((1, 1, height, width), 5.0)
    camera_pose, intrinsic = _camera()
    perturbation = PolicyPerturbation(sim_cfg, 1, "cpu", 13, policy_dt=0.05)
    perturbation.reset(torch.tensor([0]))
    if sim_cfg.enabled and sim_cfg.input_delay:
        perturbation.observation_buffer.time_lags.fill_(1)
    camera_output = {
        "policy_depth": SimpleNamespace(
            torch=F.adaptive_max_pool2d(3 / raw.clamp(0.3, 24) - 0.6, (12, 16)).permute(0, 2, 3, 1)
        ),
        "distance_to_image_plane": SimpleNamespace(torch=raw.permute(0, 2, 3, 1)),
    }
    camera = SimpleNamespace(
        data=SimpleNamespace(
            output=camera_output,
            intrinsic_matrices=SimpleNamespace(torch=intrinsic),
            pos_w=SimpleNamespace(torch=camera_pose[:, :3].clone()),
            quat_w_world=SimpleNamespace(torch=camera_pose[:, 3:].clone()),
        ),
        cfg=SimpleNamespace(
            max_distance=24.0,
            offset=SimpleNamespace(rot=(0.0, 0.0, 0.0, 1.0), pos=(0.0, 0.0, 0.0), convention="world"),
        ),
    )
    state = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.05,
        common_step_counter=0,
        root_state=lambda: state,
        scene={"depth_camera": camera},
        perturbation=perturbation,
        cfg=SimpleNamespace(
            sim_to_real=sim_cfg,
            perception_cfg=SimpleNamespace(mode="depth"),
            depth_event_history_frames=history_frames,
            depth_event_pose_source=pose_source,
        ),
        depth_event_detector=DepthEventDetector(height, width, max_depth=24, history_frames=history_frames),
        target_pos=torch.tensor([[10.0, 0.0, 0.0]]),
        episode_id=torch.zeros(1, dtype=torch.long),
        height_range=torch.tensor([[0.0, 2.0]]),
    )
    return env, raw, camera_output, camera


def test_actual_zhang_observation_delays_event_and_depth_together():
    sim_cfg = SimToRealCfg(enabled=True, input_delay=True, observation_noise=False)
    env, raw, camera_output, _ = _make_mock_env(sim_cfg=sim_cfg)
    initial = NavRLEnv.get_zhang_observation(env)
    assert isinstance(initial, ObservationContext)
    assert initial.policy["depth_features"].shape == (1, 2, 12, 16)
    assert initial.policy["depth_motion_pixels"].shape == (1, 1, 24, 32)
    assert not initial.policy["depth_features"][:, 1].any()

    def push(depth_value, tick):
        raw[:, :, 5:19, 8:24] = depth_value
        camera_output["policy_depth"].torch = F.adaptive_max_pool2d(
            3 / raw.clamp(0.3, 24) - 0.6, (12, 16)
        ).permute(0, 2, 3, 1)
        env.common_step_counter = tick
        return NavRLEnv.get_zhang_observation(env)

    delayed = push(3.0, 1)
    assert float(delayed.privileged["depth_age_s"][0]) == pytest.approx(0.05)
    assert not delayed.policy["depth_features"][:, 1].any()
    delivered = push(2.0, 2)
    assert float(delivered.privileged["depth_age_s"][0]) == pytest.approx(0.05)
    assert delivered.policy["depth_features"][:, 1].abs().sum() > 0
    # It must show the DELIVERED depth (3m), not the newer unobserved 2m.
    expected = 3 / 3.0 - 0.6
    assert abs(float(delivered.policy["depth_features"][0, 0, 4, 6]) - expected) < 0.01


def test_detector_defaults_to_estimated_pose_and_respects_truth_ablation():
    sim_cfg = SimToRealCfg(enabled=True, input_delay=False, observation_noise=True, position_std=0.5, attitude_std_deg=0.0)
    est_env, _, _, est_camera = _make_mock_env(sim_cfg=sim_cfg, pose_source="estimated")
    truth_env, _, _, truth_camera = _make_mock_env(sim_cfg=sim_cfg, pose_source="truth")
    # Corrupt simulator true camera pose; estimated mode must ignore it and use noisy odometry + nominal mount.
    est_camera.data.pos_w.torch.fill_(99.0)
    truth_camera.data.pos_w.torch.fill_(99.0)
    NavRLEnv.get_zhang_observation(est_env)
    NavRLEnv.get_zhang_observation(truth_env)
    assert not torch.allclose(est_env.depth_event_detector._pose[0, 0, :3], torch.full((3,), 99.0))
    torch.testing.assert_close(truth_env.depth_event_detector._pose[0, 0, :3], torch.full((3,), 99.0))


def _temporal_detector(height=48, width=64):
    return DepthEventDetector(
        height,
        width,
        max_depth=24,
        history_frames=4,
        threshold=0.0,
        relative_threshold=0.002,
        depth_sigma=0.02,
        pose_position_sigma=0.01,
        pose_rotation_sigma=0.002,
        sigma_multiplier=2.5,
    )


@pytest.mark.parametrize("speed", [0.5, 1.0, 1.5])
def test_temporal_detector_detects_slow_axial_motion(speed):
    pose, k = _camera()
    detector = _temporal_detector()
    last = None
    for step in range(5):
        last = detector.update(torch.full((1, 48, 64), 5.0 - speed * step * 0.05), pose, k, step * 0.05)
        if step < 4:
            assert not last.appeared.any()
    assert last.appeared.any()


def test_temporal_detector_does_not_confuse_known_camera_motion_with_object_motion():
    pose, k = _camera()
    detector = _temporal_detector()
    for step in range(9):
        moved_pose = pose.clone()
        moved_pose[:, 0] = step * 0.025
        result = detector.update(torch.full((1, 48, 64), 5.0 - step * 0.025), moved_pose, k, step * 0.05)
        assert not result.appeared.any()
        assert not result.disappeared.any()


def test_temporal_detector_reset_discards_old_episode_history():
    pose, k = _camera(n=2)
    detector = _temporal_detector()
    for step in range(5):
        result = detector.update(torch.full((2, 48, 64), 5.0 - step * 0.05), pose, k, step * 0.05)
    assert result.appeared.any()
    detector.reset(torch.tensor([0]))
    result = detector.update(torch.full((2, 48, 64), 3.0), pose, k, 0.25)
    assert not result.appeared[0].any()
    assert result.appeared[1].any()


def test_region_pool_keeps_motion_edge_that_nearest_depth_pool_drops():
    depth = torch.full((1, 1, 120, 160), 8.0)
    depth[..., 30:70, 50:80] = 4.0
    signed = torch.zeros_like(depth)
    signed[..., 31:34, 58:60] = 1.0
    old_pool = pooled_signed_event_channel(depth, signed)
    new_pool = pooled_signed_event_channel(depth, signed, region_pool=True)
    assert (new_pool > 0).sum() > (old_pool > 0).sum()
    assert signed.shape == (1, 1, 120, 160)
