"""The public default, packaged policies and README describe one baseline."""

from pathlib import Path

import pytest
import torch

from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.diffnav.inference import load_diffnav_policy
from aerialnav.diffnav.model import policy_state
from scripts.audit_release import audit

ROOT = Path(__file__).parents[1]


def test_legacy_sampling_label_is_canonicalized_without_enabling_dcol():
    cfg = DiffNavTrainCfg(variant="dcol")
    cfg.validate()
    assert cfg.variant == "terminal" and cfg.collision_metric == "sdf"
    assert cfg.position_integration == "trapezoid" and cfg.loss_timing == "pre_reset"
    with pytest.raises(ValueError, match="SDF|SDF|sdf"):
        DiffNavTrainCfg(collision_metric="dcol").validate()


def test_goal_reference_is_the_approved_linear_formula():
    p = torch.zeros(1, 3)
    _, _, target, _ = policy_state(
        p,
        p,
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.tensor([[2.0, 0.0, 0.0]]),
        torch.tensor([[6.0]]),
        torch.tensor([0.3]),
    )
    torch.testing.assert_close(target, torch.tensor([[2.0, 0.0, 0.0]]))


@pytest.mark.parametrize("name,backend", [("baseline_direct.pt", "curobo_v2"), ("best_static_s0.pt", "native")])
def test_packaged_new_and_legacy_policy_can_load_and_act(name, backend):
    from test_diffnav_contract import AnalyticEnv

    env = AnalyticEnv()
    policy = load_diffnav_policy(env, ROOT / "weights" / name)
    action = policy()
    assert action.shape == (env.num_envs, 3) and torch.isfinite(action).all()
    assert policy.cfg.geometry_backend == backend


def test_weight_audit_finds_private_metadata(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    torch.save({"metadata": "zh" + "ang"}, weights / "policy.pt")
    assert any("private checkpoint metadata" in x for x in audit(tmp_path, weights=True))


def test_readme_is_short_and_has_only_supported_sections():
    text = (ROOT / "README.md").read_text()
    assert len(text.splitlines()) < 100
    assert "Deployment" not in text
    assert "weights/baseline_direct.pt" in text
    assert all(x in text for x in ["## Installation", "## Test the Policy", "## Train the Policy"])
