"""Construct full-window DiffNav losses before reset; normalize once per rollout."""

import torch
from torch.nn import functional as F


class PreResetLoss:
    def __init__(self, cfg, transition_costs, prefix=None):
        self.cfg, self.transition_costs = cfg, transition_costs
        self.window = {
            name: list(prefix[name].detach().unbind()) if prefix is not None else []
            for name in ("velocity", "episode_id", "target_velocity")
        }
        self.steps = 0
        self.losses, self.costs, self.velocity_costs = [], [], []
        self.valid_samples = None

    def append(self, sample):
        window = self.cfg.velocity_window_steps
        for name, values in self.window.items():
            values.append(sample[name])
        velocity = torch.stack(self.window["velocity"][-window:])
        episode = torch.stack(self.window["episode_id"][-window:])
        value = sample["velocity"].new_zeros(sample["velocity"].shape[0])
        valid = torch.zeros_like(sample["episode_id"], dtype=torch.bool)
        # Preserve the source's omitted first sample and window-start target.
        if len(self.window["velocity"]) > window:
            valid = (episode == episode[-1:]).all(0)
            error = (velocity.mean(0) - self.window["target_velocity"][-window]).norm(dim=-1)
            value = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none") * valid
        self.valid_samples = valid.sum() if self.valid_samples is None else self.valid_samples + valid.sum()
        losses, _, cost = self.transition_costs({name: [value] for name, value in sample.items()}, self.cfg)
        self.losses.append(losses)
        self.costs.append(cost[0])
        self.velocity_costs.append(value)
        self.steps += 1
        for name in self.window:
            self.window[name] = self.window[name][-window:]

    def finish(self):
        if not self.steps:
            raise ValueError("Cannot finish an empty loss rollout")
        losses = {
            name: torch.stack([row[name] for row in self.losses]).mean()
            for name in self.losses[0]
            if name not in ("total", "velocity_valid_samples", "velocity_sample_fraction")
        }
        cost = torch.stack(self.costs)
        velocity = torch.stack(self.velocity_costs) * (cost.numel() / self.valid_samples.clamp_min(1))
        losses["velocity"] = velocity.mean()
        cost = cost + self.cfg.coef_v * velocity
        losses.update(
            total=cost.mean(),
            velocity_valid_samples=self.valid_samples.detach(),
            velocity_sample_fraction=self.valid_samples / cost.numel(),
        )
        return losses, {name: torch.stack(values).detach() for name, values in self.window.items()}, cost
