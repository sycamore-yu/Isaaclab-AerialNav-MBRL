"""Causal, depth-aligned, initially policy-neutral v4 kinematic event channel tests."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from isaaclab_navrl.tasks.navigation.context import ObservationContext
from isaaclab_navrl.tasks.navigation.geometric_depth_events import DepthEventDetector
from isaaclab_navrl.tasks.navigation.navrl_env import NavRLEnv
from isaaclab_navrl.tasks.navigation.randomization import PolicyPerturbation, SimToRealCfg
from isaaclab_navrl.zhang.cli import parse_train_args
from isaaclab_navrl.zhang.config import ZhangTrainCfg
from isaaclab_navrl.zhang.normalization import PolicyObservationNormalizer


def _camera(n=1, *, x=0.0, yaw=0.0):
    pose = (
        torch.tensor([x, 0, 0, 0, 0, math.sin(yaw / 2), math.cos(yaw / 2)], dtype=torch.float32)
        .expand(n, -1)
        .clone()
    )
    k = torch.tensor([[60.0, 0, 32.0], [0, 60.0, 24.0], [0, 0, 1.0]]).expand(n, -1, -1).clone()
    return pose, k


def test_normalizer_never_normalizes_kinematic_event_channels():
    depth = torch.rand(3, 1, 24, 32)
    kin = torch.rand(3, 3, 24, 32)
    state = torch.randn(3, 10)
    normalizer = PolicyObservationNormalizer(enabled=True)
    normalizer.observe(torch.cat((depth, kin), 1), state)
    normalized, _ = normalizer.normalize(torch.cat((depth, kin), 1), state, normalizer.begin_rollout())
    torch.testing.assert_close(normalized[:, 1:], kin)


def test_cli_optin_does_not_mutate_default():
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


def test_actual_zhang_observation_delays_kinematic_event_and_depth_together():
    sim_cfg = SimToRealCfg(enabled=True, input_delay=True, observation_noise=False)
    env, raw, camera_output, _ = _make_mock_env(sim_cfg=sim_cfg)
    initial = NavRLEnv.get_zhang_observation(env)
    assert isinstance(initial, ObservationContext)
    assert initial.policy["depth_features"].shape == (1, 4, 24, 32)
    assert not initial.policy["depth_features"][:, 1:].any()

    def push(depth_value, tick):
        raw[:, :, 5:19, 8:24] = depth_value
        camera_output["policy_depth"].torch = F.adaptive_max_pool2d(
            3 / raw.clamp(0.3, 24) - 0.6, (12, 16)
        ).permute(0, 2, 3, 1)
        env.common_step_counter = tick
        return NavRLEnv.get_zhang_observation(env)

    delayed = push(3.0, 1)
    assert float(delayed.privileged["depth_age_s"][0]) == pytest.approx(0.05)
    assert not delayed.policy["depth_features"][:, 1:].any()
    delivered = push(2.0, 2)
    assert float(delivered.privileged["depth_age_s"][0]) == pytest.approx(0.05)
    assert delivered.policy["depth_features"][:, 1:].abs().sum() > 0


def test_detector_defaults_to_estimated_pose_and_respects_truth_ablation():
    sim_cfg = SimToRealCfg(enabled=True, input_delay=False, observation_noise=True, position_std=0.5, attitude_std_deg=0.0)
    est_env, _, _, est_camera = _make_mock_env(sim_cfg=sim_cfg, pose_source="estimated")
    truth_env, _, _, truth_camera = _make_mock_env(sim_cfg=sim_cfg, pose_source="truth")
    est_camera.data.pos_w.torch.fill_(99.0)
    truth_camera.data.pos_w.torch.fill_(99.0)
    NavRLEnv.get_zhang_observation(est_env)
    NavRLEnv.get_zhang_observation(truth_env)
    assert not torch.allclose(est_env.depth_event_detector._pose[0, 0, :3], torch.full((3,), 99.0))
    torch.testing.assert_close(truth_env.depth_event_detector._pose[0, 0, :3], torch.full((3,), 99.0))


def test_kinematic_detector_reset_discards_old_episode_history():
    pose, k = _camera(n=2)
    detector = DepthEventDetector(48, 64, max_depth=24, history_frames=4)
    for step in range(5):
        out = detector.update(torch.full((2, 48, 64), 5.0 - step * 0.05), pose, k, step * 0.05)
    assert out.kinematic_image(pose, k, 24.0).abs().sum() > 0
    detector.reset(torch.tensor([0]))
    out = detector.update(torch.full((2, 48, 64), 3.0), pose, k, 0.25)
    assert not out.kinematic_image(pose, k, 24.0)[0].any()
    assert out.kinematic_image(pose, k, 24.0)[1].abs().sum() > 0
