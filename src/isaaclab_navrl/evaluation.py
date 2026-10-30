"""First-outcome episode accounting shared by both training methods."""

from __future__ import annotations

import statistics


def first_outcome(*, reached, collision, bounds, invalid, timeout):
    if invalid:
        return "invalid"
    if collision:
        return "collision"
    if bounds:
        return "bounds"
    if reached:
        return "success"
    if timeout:
        return "timeout"
    return None


class EpisodeEvaluation:
    """Latch exactly one result per initial drone episode, before autoreset."""

    def __init__(self, starts, goals, seed):
        self.rows = [None] * len(starts)
        self.starts, self.goals, self.seed = starts, goals, seed

    def update(self, transition):
        values = {
            key: transition[key].detach().cpu().tolist()
            for key in ("reached", "collision", "bounds", "invalid", "timeout", "time_s", "step")
        }
        substep_codes = transition.get("first_event_code")
        if substep_codes is not None:
            substep_codes = substep_codes.detach().cpu().tolist()
            values["time_s"] = transition["first_event_time_s"].detach().cpu().tolist()
        for index, row in enumerate(self.rows):
            if row is not None:
                continue
            if substep_codes is None:
                result = first_outcome(
                    **{key: values[key][index] for key in ("reached", "collision", "bounds", "invalid", "timeout")}
                )
            else:
                result = (None, "success", "collision", "bounds", "invalid", "timeout")[substep_codes[index]]
            if result is not None:
                self.rows[index] = {
                    "episode": index,
                    "seed": self.seed,
                    "outcome": result,
                    "time_s": values["time_s"][index],
                    "step": values["step"][index],
                    "start": self.starts[index],
                    "goal": self.goals[index],
                }

    @property
    def complete(self):
        return all(row is not None for row in self.rows)

    def summary(self):
        if not self.complete:
            raise ValueError("Evaluation contains unfinished episodes.")
        labels = ("success", "collision", "timeout", "bounds", "invalid")
        counts = {key: sum(row["outcome"] == key for row in self.rows) for key in labels}
        times = [row["time_s"] for row in self.rows if row["outcome"] == "success"]
        count = len(self.rows)
        return {
            "episodes": count,
            "counts": counts,
            **{f"{label}_rate": counts[label] / count for label in labels},
            "success_time_mean_s": statistics.mean(times) if times else None,
            "success_time_median_s": statistics.median(times) if times else None,
            "success_time_std_s": statistics.stdev(times) if len(times) > 1 else None,
        }
