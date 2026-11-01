"""The original Navigation shared encoder, adapted to RSL-RL 5.5.1 models.

Network structure is derived from Zhefan-Xu/Navigation (MIT; see LICENSE-Navigation).
PPO, rollout storage, distributions, and checkpoint management remain upstream.
"""

from __future__ import annotations

import copy

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import BetaDistribution
from rsl_rl.utils import resolve_class
from torch import nn


def _mlp(input_dim: int, widths: tuple[int, ...]) -> nn.Sequential:
    layers = []
    for width in widths:
        layers.extend((nn.Linear(input_dim, width), nn.LeakyReLU(), nn.LayerNorm(width)))
        input_dim = width
    return nn.Sequential(*layers)


class NavigationEncoder(nn.Module):
    """One shared LiDAR, dynamic-obstacle, and fusion feature extractor."""

    def __init__(self):
        super().__init__()
        self.lidar = nn.Sequential(
            nn.Conv2d(1, 4, (5, 3), padding=(2, 1)),
            nn.ELU(),
            nn.Conv2d(4, 16, (5, 3), stride=(2, 1), padding=(2, 1)),
            nn.ELU(),
            nn.Conv2d(16, 16, (5, 3), stride=(2, 2), padding=(2, 1)),
            nn.ELU(),
            nn.Flatten(1),
            nn.Linear(16 * 9 * 2, 128),
            nn.LayerNorm(128),
        )
        self.dynamic = _mlp(50, (128, 64))
        self.fusion = _mlp(128 + 8 + 64, (256, 256))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        state = observations[:, :8]
        lidar = observations[:, 8:152].reshape(-1, 1, 36, 4)
        dynamic = observations[:, 152:202]
        # Vendored CatTensors sorts its keys by default: CNN, dynamic, state.
        return self.fusion(torch.cat((self.lidar(lidar), self.dynamic(dynamic), state), dim=-1))


class BetaHeads(nn.Module):
    """Separate original alpha/beta linear heads with orthogonal initialization."""

    def __init__(self):
        super().__init__()
        self.alpha_layer = nn.Linear(256, 3)
        self.beta_layer = nn.Linear(256, 3)
        for layer in (self.alpha_layer, self.beta_layer):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack((self.alpha_layer(features), self.beta_layer(features)), dim=-2)


class NavigationActor(MLPModel):
    """Beta velocity policy owning the complete shared feature extractor."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, distribution_cfg):
        nn.Module.__init__(self)
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        if self.obs_groups != ["policy"] or self.obs_dim != 202 or output_dim != 3:
            raise ValueError("Navigation expects only policy[202] observations and 3 velocity actions.")
        self.obs_normalization = False
        self.obs_normalizer = nn.Identity()
        # Official PPO.construct_algorithm passes actor.cnns to the critic when
        # share_cnn_encoders=True. This object includes all shared feature layers.
        self.cnns = NavigationEncoder()
        self.mlp = BetaHeads()
        distribution_type, distribution_args = resolve_class(distribution_cfg)
        if distribution_type is not BetaDistribution:
            raise ValueError("Navigation's two heads require a BetaDistribution")
        if tuple(distribution_args.get("action_range", ())) != (-2.0, 2.0):
            raise ValueError("Navigation training and exports require the fixed [-2, 2] action range")
        self.distribution = distribution_type(3, **distribution_args)

    def get_latent(self, obs, masks=None, hidden_state=None):
        return self.cnns(obs["policy"])

    def as_jit(self):
        return NavigationExport(self)

    def as_onnx(self, verbose):
        return NavigationExport(self, verbose=verbose)


class NavigationCritic(MLPModel):
    """Value head using actor-owned features without duplicate optimizer entries."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, cnns, distribution_cfg=None):
        nn.Module.__init__(self)
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        if self.obs_groups != ["policy"] or self.obs_dim != 202 or output_dim != 1:
            raise ValueError("Navigation critic expects policy[202] observations and one value.")
        if distribution_cfg is not None:
            raise ValueError("Navigation critic is deterministic")
        if not isinstance(cnns, NavigationEncoder):
            raise TypeError("Navigation actor and critic must share one complete NavigationEncoder.")
        self.obs_normalization = False
        self.obs_normalizer = nn.Identity()
        self.distribution = None
        # Register shared weights exactly once: actor state_dict owns them, while
        # this reference preserves autograd from value loss into the same tensors.
        object.__setattr__(self, "encoder", cnns)
        self.mlp = nn.Linear(256, 1)
        nn.init.orthogonal_(self.mlp.weight, gain=0.01)
        nn.init.zeros_(self.mlp.bias)

    def get_latent(self, obs, masks=None, hidden_state=None):
        return self.encoder(obs["policy"])


class NavigationExport(nn.Module):
    """Deterministic standalone policy for the official runner's export methods."""

    is_recurrent = False

    def __init__(self, actor: NavigationActor, verbose: bool = False):
        super().__init__()
        self.encoder = copy.deepcopy(actor.cnns)
        self.heads = copy.deepcopy(actor.mlp)
        self.verbose = verbose

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.heads(self.encoder(obs))
        alpha, beta = torch.nn.functional.softplus(raw).unbind(-2)
        alpha, beta = alpha + 1.0, beta + 1.0
        return 4.0 * alpha / (alpha + beta) - 2.0

    @torch.jit.export
    def reset(self):
        pass

    def get_dummy_inputs(self):
        return (torch.zeros(1, 202),)

    @property
    def input_names(self):
        return ["policy"]

    @property
    def output_names(self):
        return ["velocity_goal_frame"]
