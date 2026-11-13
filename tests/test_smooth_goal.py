"""Approved smooth goal reference, gate and checkpoint contracts."""

import pytest
import torch

from isaaclab_navrl.zhang.cli import parse_train_args
from isaaclab_navrl.zhang.config import ZhangTrainCfg
from isaaclab_navrl.zhang.goal import final_approach_weight, goal_velocity
from isaaclab_navrl.zhang.model import policy_state


def reference(x, speed=6.0):
    return goal_velocity(
        torch.zeros_like(x), torch.cat((x, x * 0, x * 0), dim=-1), torch.full_like(x, speed), profile="smooth"
    )[:, :1]


def test_approved_curve_and_no_zero_speed_switch_at_arrival_boundary():
    x = torch.tensor([[12.0], [10.0], [8.0], [6.0], [4.0], [2.0], [1.0], [0.5], [0.0]])
    actual = reference(x)
    expected = torch.tensor(
        [[6.0], [5.8680556], [5.1851852], [3.9375], [2.4444444], [1.0810185], [0.5119358], [0.2516118], [0.0]]
    )
    torch.testing.assert_close(actual, expected)
    assert actual[-2] > 0
    assert actual[-1] == 0


def test_c2_join_linear_capture_and_reference_is_detached():
    def f(s):
        return s + 4 * s**3 - 7 * s**4 + 3 * s**5

    for point, first in [(0.0, 1.0), (1.0, 0.0)]:
        s = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        y = f(s)
        g = torch.autograd.grad(y, s, create_graph=True)[0]
        h = torch.autograd.grad(g, s)[0]
        assert y.item() == point and g.item() == first and h.item() == 0
    x = torch.tensor([[1e-7]], requires_grad=True)
    out = reference(x)
    assert torch.isfinite(out).all() and not out.requires_grad
    torch.testing.assert_close(out, x.detach() * 0.5)


def test_monotonic_speed_caps_and_ideal_derivative_bounds():
    s = torch.linspace(0, 1, 10001, dtype=torch.float64)
    f = s + 4 * s**3 - 7 * s**4 + 3 * s**5
    fp = 1 + 12 * s * s - 28 * s**3 + 15 * s**4
    fpp = 24 * s - 84 * s * s + 60 * s**3
    assert (f.diff() >= -1e-12).all() and f.min() == 0 and f.max() == 1
    assert (f * fp).max() * 36 / 12 < 3.0
    assert (f * (fp**2 + f * fpp)).abs().max() * 216 / 144 < 5.1
    for speed in [1.0, 3.0, 6.0]:
        actual = reference(torch.linspace(0, 100, 1001)[:, None], speed)
        assert actual.min() >= 0 and actual.max() <= speed


def test_profile_and_cli_validation():
    cfg = ZhangTrainCfg()
    assert cfg.goal_speed_profile == "smooth" and cfg.goal_approach_time_s == 2.0
    cfg.validate()
    assert parse_train_args(["--goal_speed_profile", "linear", "--goal_approach_time_s", "3"]) == dict(
        goal_speed_profile="linear", goal_approach_time_s=3.0
    )
    for bad in [-1.0, 0.0, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="goal_approach_time_s"):
            ZhangTrainCfg(goal_approach_time_s=bad).validate()
    with pytest.raises(ValueError, match="goal_speed_profile"):
        ZhangTrainCfg(goal_speed_profile="unknown").validate()


def test_blocked_final_segment_does_not_trigger_euclidean_slowdown():
    class Wall:
        def query_clearance(self, q, *, body_radius):
            return (q[..., 0] - 3).abs() - 0.2 - body_radius

    class Free:
        def query_clearance(self, q, *, body_radius):
            return torch.full(q.shape[:-1], 10.0) - body_radius

    p = torch.tensor([[0.0, 0.0, 2.0]])
    target = torch.tensor([[6.0, 0.0, 2.0]])
    speed = torch.tensor([[6.0]])
    margin = torch.tensor([0.3])
    blocked = final_approach_weight(Wall(), p, target, speed, margin, 2.0)
    free = final_approach_weight(Free(), p, target, speed, margin, 2.0)
    assert blocked == 0 and free == 1
    torch.testing.assert_close(
        goal_velocity(p, target, speed, profile="smooth", clear_weight=blocked), torch.tensor([[6.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(
        goal_velocity(p, target, speed, profile="smooth", clear_weight=free), torch.tensor([[3.9375, 0.0, 0.0]])
    )
    assert final_approach_weight(Wall(), p, target + torch.tensor([[20.0, 0.0, 0.0]]), speed, margin, 2.0) == 0


def test_policy_features_and_teacher_share_profile_without_changing_dimensions():
    p = torch.tensor([[0.0, 0.0, 2.0]])
    v = p * 0
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    goal = p + torch.tensor([[6.0, 0.0, 0.0]])
    args = (p, v, q, goal, torch.tensor([[6.0]]), torch.tensor([0.3]))
    legacy = policy_state(*args)
    smooth = policy_state(*args, goal_speed_profile="smooth", goal_clear_weight=torch.ones(1))
    assert legacy[0].shape == smooth[0].shape == (1, 10)
    torch.testing.assert_close(legacy[2], torch.tensor([[6.0, 0.0, 0.0]]))
    torch.testing.assert_close(smooth[2], torch.tensor([[3.9375, 0.0, 0.0]]))
    torch.testing.assert_close(smooth[0][:, 3:6], smooth[2])


def test_old_checkpoint_inference_remains_linear_and_new_inference_uses_same_gate(tmp_path):
    from test_zhang_contract import AnalyticEnv

    from isaaclab_navrl.zhang.inference import load_zhang_policy
    from isaaclab_navrl.zhang.runner import ZhangTrainer

    env = AnalyticEnv()
    env.reset()
    cfg = ZhangTrainCfg(variant="step", physics_dt=0.01, policy_dt=0.1, goal_speed_profile="linear")
    trainer = ZhangTrainer(env, cfg, tmp_path)
    ck = tmp_path / "old.pt"
    trainer._checkpoint(ck)
    payload = torch.load(ck, weights_only=True)
    del payload["train_cfg"]["goal_speed_profile"]
    del payload["train_cfg"]["goal_approach_time_s"]
    torch.save(payload, ck)
    old = load_zhang_policy(env, ck)
    assert old.cfg.goal_speed_profile == "linear"
    trainer.cfg = ZhangTrainCfg(variant="step", physics_dt=0.01, policy_dt=0.1)
    ck2 = tmp_path / "new.pt"
    trainer._checkpoint(ck2)
    new = load_zhang_policy(env, ck2)
    inputs = []

    def capture(module, args):
        inputs.append(args[1].clone())

    h1 = old.model.register_forward_pre_hook(capture)
    h2 = new.model.register_forward_pre_hook(capture)
    old()
    new()
    h1.remove()
    h2.remove()
    torch.testing.assert_close(inputs[0][:, 3], torch.full((2,), 6.0))
    assert (inputs[1][:, 3] < 6.0).all()
