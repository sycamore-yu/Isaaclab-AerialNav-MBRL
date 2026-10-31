"""Check the standard RSL-RL Beta action distribution against Navigation's affine mapping.

Run in the Isaac Lab 3.0 virtual environment:
    python tests/test_rsl_beta_contract.py

This checks action distribution semantics only, not network checkpoint mapping.
"""

import math

import torch
import torch.nn.functional as F
from rsl_rl.modules.distribution import BetaDistribution


def main() -> None:
    torch.manual_seed(20261028)
    raw = torch.randn(32, 2, 3)

    # Upstream PPO uses IndependentBeta(1 + softplus(raw) + 1e-6)
    # and applies a subsequent affine [0,1] -> [-2,2] velocity mapping.
    old_alpha = 1.0 + F.softplus(raw[:, 0]) + 1e-6
    old_beta = 1.0 + F.softplus(raw[:, 1]) + 1e-6
    old_expected_vel = old_alpha / (old_alpha + old_beta) * 4.0 - 2.0

    rsl_distribution = BetaDistribution(output_dim=3, action_range=(-2.0, 2.0))
    rsl_distribution.update(raw)
    new_expected_vel = rsl_distribution.mean
    mean_error = (old_expected_vel - new_expected_vel).abs().max().item()

    sampled_vel = rsl_distribution.sample()
    assert sampled_vel.shape == (32, 3)
    assert (sampled_vel >= -2).all() and (sampled_vel <= 2).all()
    old_log_prob = (
        torch.distributions.Beta(old_alpha, old_beta)
        .log_prob(((sampled_vel + 2.0) / 4.0).clamp(1e-6, 1.0 - 1e-6))
        .sum(dim=-1)
    )
    new_log_prob = rsl_distribution.log_prob(sampled_vel)
    # A constant log-Jacobian factor changes log density under rescaling.
    log_prob_error = ((old_log_prob - 3.0 * math.log(4.0)) - new_log_prob).abs().max().item()

    print(f"beta_mean_max_abs_velocity_error_mps={mean_error:.9g}")
    print(f"beta_logprob_max_abs_error={log_prob_error:.9g}")
    assert mean_error < 1e-5
    assert log_prob_error < 1e-4


if __name__ == "__main__":
    main()
