"""Navigation's success-driven dynamic-obstacle curriculum, using training episodes."""

from __future__ import annotations

import torch

from aerialnav.training import STAGE_CONFIG

from .task_logic import obstacle_categories


def active_obstacle_indices(count: int, capacity: int) -> list[int]:
    """Select the original cuboid/cylinder width distribution from a fixed pool."""
    selected, offset = [], 0
    for used, available in zip(obstacle_categories(count), obstacle_categories(capacity), strict=True):
        if used > available:
            raise ValueError("Active obstacle categories exceed the allocated scene")
        selected.extend(range(offset, offset + used))
        offset += available
    return selected


class TrainingCurriculum:
    """Aggregate completed episodes at the upstream EpisodeStats reporting cadence.

    Arrival does not reset the original task. Its first safe outcome is latched
    until natural termination, so a quick success cannot be counted ahead of
    unfinished failures. Stage changes discard unfinished episodes explicitly.
    """

    stages = (60, 80, 100, 120)
    threshold = 0.8
    labels = ("success", "collision", "bounds", "invalid", "timeout")

    def __init__(self, num_envs: int, device: str, stage: str | None = None, *, stage_config=None):
        if stage is not None and stage not in STAGE_CONFIG:
            raise ValueError(f"Unknown training stage: {stage}")
        self.stage_name = stage
        self.stage_config = (STAGE_CONFIG[stage] if stage_config is None else stage_config).copy() if stage else None
        if self.stage_config is not None and self.stage_config["dynamic_obstacles"] < 0:
            raise ValueError("dynamic obstacle count must be nonnegative")
        self.stage_passed = False
        self.min_episodes = num_envs
        self.stage_index = 0
        self.outcome = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.arrival_time = torch.zeros(num_envs, device=device)
        self.counts = torch.zeros(6, dtype=torch.long, device=device)
        self.time_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.return_sum = torch.zeros_like(self.time_sum)
        self.length_sum = torch.zeros_like(self.time_sum)
        self.ready_to_advance = False
        self.acceptance_passed = False
        self.best_by_stage = {}
        self.last_report = None

    @property
    def active_obstacles(self) -> int:
        return self.stage_config["dynamic_obstacles"] if self.stage_name else self.stages[self.stage_index]

    @property
    def stage_key(self):
        return self.stage_name or self.active_obstacles

    @torch.no_grad()
    def observe(self, transition: dict) -> None:
        if "first_event_code" in transition:
            code = transition["first_event_code"].long()
            event_time = transition["first_event_time_s"]
        else:
            code = torch.zeros_like(self.outcome)
            code[transition["timeout"]] = 5
            code[transition["reached"]] = 1
            code[transition["bounds"]] = 3
            code[transition["collision"]] = 2
            code[transition["invalid"]] = 4
            event_time = transition["time_s"]
        first = (self.outcome == 0) & (code != 0)
        self.outcome[first] = code[first]
        arrived = first & (code == 1)
        self.arrival_time[arrived] = event_time[arrived]
        done = transition["terminated"] | transition["timeout"]
        self.counts += torch.bincount(self.outcome[done], minlength=6)
        self.time_sum += self.arrival_time[done & (self.outcome == 1)].double().sum()
        self.return_sum += transition["episode_return"][done].double().sum()
        self.length_sum += transition["step"][done].double().sum()
        self.outcome[done] = 0
        self.arrival_time[done] = 0

    def reset_episodes(self) -> None:
        self.outcome.zero_()
        self.arrival_time.zero_()

    def pop_report(self, update: int | None = None) -> dict | None:
        counts = self.counts.cpu().tolist()
        episodes = sum(counts)
        if episodes < self.min_episodes:
            return None
        if counts[0]:
            raise RuntimeError("A completed training episode has no terminal outcome")
        report = {
            "update": update,
            "dynamic_obstacles": self.active_obstacles,
            "episodes": episodes,
            "counts": dict(zip(self.labels, counts[1:], strict=True)),
            **{f"{name}_rate": count / episodes for name, count in zip(self.labels, counts[1:], strict=True)},
            "success_time_mean_s": float(self.time_sum) / counts[1] if counts[1] else None,
            "episode_return_mean": float(self.return_sum) / episodes,
            "episode_length_mean": float(self.length_sum) / episodes,
            "reward_per_step_mean": float(self.return_sum / self.length_sum),
        }
        if self.stage_name:
            report.update(stage=self.stage_name, **self.stage_config)
        previous = self.best_by_stage.get(self.stage_key)
        report["new_best"] = previous is None or report["success_rate"] > previous["success_rate"]
        if report["new_best"]:
            self.best_by_stage[self.stage_key] = report.copy()
        self.stage_passed |= report["success_rate"] >= self.threshold
        self.acceptance_passed |= (
            self.stage_name == "finetune_s2" if self.stage_name else self.active_obstacles == 100
        ) and self.stage_passed
        self.ready_to_advance = (
            self.stage_name is None
            and self.stage_index < len(self.stages) - 1
            and report["success_rate"] > self.threshold
        )
        self.last_report = report
        self.counts.zero_()
        self.time_sum.zero_()
        self.return_sum.zero_()
        self.length_sum.zero_()
        return report

    def advance(self) -> int:
        if not self.ready_to_advance:
            raise RuntimeError("The current training success rate has not passed the next-stage threshold")
        self.stage_index += 1
        self.stage_passed = False
        self.ready_to_advance = False
        self.reset_episodes()
        self.counts.zero_()
        self.time_sum.zero_()
        self.return_sum.zero_()
        self.length_sum.zero_()
        self.last_report = None
        return self.active_obstacles

    def state_dict(self) -> dict:
        return {
            "stages": list(STAGE_CONFIG) if self.stage_name else list(self.stages),
            "active_dynamic_obstacles": self.active_obstacles,
            "stage": self.stage_name,
            "stage_passed": self.stage_passed,
            "stage_config": self.stage_config if self.stage_name else None,
            "promotion_success_rate": self.threshold,
            "episodes_per_report_min": self.min_episodes,
            "acceptance_passed": self.acceptance_passed,
            "best_by_stage": self.best_by_stage,
            "latest_training_report": self.last_report,
        }
