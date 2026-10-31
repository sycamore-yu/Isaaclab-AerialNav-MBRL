"""Navigation's episode-band penalty and its actual contribution to BPTT."""

from dataclasses import replace

import pytest
import torch
from test_diffnav_height_loss import _height_only_cfg, _history

from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.diffnav.losses import aerialnav_height_loss, diffnav_transition_costs


def test_aerialnav_band_formula_gradient_and_c1_boundaries():
    z = torch.tensor([0.3, 0.8 - 1e-7, 0.8, 0.8 + 1e-7, 1.5, 2.2 - 1e-7, 2.2, 2.2 + 1e-7, 2.7], dtype=torch.float64)
    p = torch.stack((z * 0, z * 0, z), -1).requires_grad_()
    band = torch.tensor([1.0, 2.0], dtype=p.dtype).expand(len(z), -1).clone().requires_grad_()
    value = aerialnav_height_loss(p, band)
    expected = (0.8 - z).relu().square() + (z - 2.2).relu().square()
    torch.testing.assert_close(value, expected, atol=1e-15, rtol=0)
    grad = torch.autograd.grad(value.sum(), p)[0][:, 2]
    target = torch.where(z < 0.8, 2 * (z - 0.8), torch.where(z > 2.2, 2 * (z - 2.2), 0))
    torch.testing.assert_close(grad, target, atol=1e-15, rtol=0)
    assert band.grad is None and max(abs(float(grad[2])), abs(float(grad[6]))) < 1e-14
    assert torch.autograd.gradcheck(lambda q: aerialnav_height_loss(q, band), (p,))


def test_aerialnav_height_uses_cloned_pre_reset_band_in_total_bptt_cost():
    position = torch.tensor([[[0.0, 0.0, 0.3]], [[0.0, 0.0, 2.7]]], requires_grad=True)
    history = _history(position, position.new_full((2, 1), 1.5))
    band = torch.tensor([[1.0, 2.0]])
    history["height_range"] = [band.clone(), band.clone()]
    band.fill_(9.0)  # A later environment reset must not mutate this transition's interval.
    cfg = _height_only_cfg(height_loss_mode="goal_height_band", coef_height_boundary=1.0)
    losses, _, cost = diffnav_transition_costs(history, cfg)
    torch.testing.assert_close(cost, torch.full((2, 1), 0.25))
    grad = torch.autograd.grad(losses["total"], position)[0]
    torch.testing.assert_close(grad[..., 2], torch.tensor([[-0.5], [0.5]]))
    assert float(losses["height_boundary"]) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="pre-step episode height_range"):
        diffnav_transition_costs({k: v for k, v in history.items() if k != "height_range"}, cfg)


def test_new_defaults_match_the_retained_twenty_hz_configuration():
    cfg = DiffNavTrainCfg()
    cfg.validate()
    assert (cfg.variant, cfg.physics_dt, cfg.policy_dt) == ("dcol", 0.05, 0.05)
    assert cfg.velocity_loss_mode == "full_window" and cfg.velocity_window_steps == 40
    assert cfg.height_loss_mode == "goal_height_band" and cfg.coef_height_boundary == 1
    hz50 = replace(cfg, physics_dt=0.02, policy_dt=0.02)
    hz50.validate()
    assert hz50.velocity_window_steps == 100


def test_shac_default_velocity_supervision_is_per_step():
    cfg = DiffNavTrainCfg(algorithm="shac")
    cfg.validate()
    assert cfg.velocity_loss_mode == cfg.resolved_velocity_loss_mode == "per_step"
