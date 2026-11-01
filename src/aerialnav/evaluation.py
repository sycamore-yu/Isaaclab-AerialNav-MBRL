"""Final-goal episode accounting with historical first-event support."""

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
    """Record one finished result per initial episode, before autoreset."""

    def __init__(self, starts, goals, seed):
        self.rows = [None] * len(starts)
        self.starts, self.goals, self.seed = starts, goals, seed

    def update(self, transition):
        values = {
            key: transition[key].detach().cpu().tolist()
            for key in ("reached", "collision", "bounds", "invalid", "timeout", "time_s", "step")
        }
        final_codes = transition.get("terminal_event_code")
        substep_codes = final_codes if final_codes is not None else transition.get("first_event_code")
        if substep_codes is not None:
            substep_codes = substep_codes.detach().cpu().tolist()
            if final_codes is None:
                values["time_s"] = transition["first_event_time_s"].detach().cpu().tolist()
        arrivals = None
        if "arrived" in transition:
            arrivals = transition["arrived"].detach().cpu().tolist()
            arrival_times = transition["arrive_time"].detach().cpu().tolist()
        first_codes = transition.get("first_event_code")
        if first_codes is not None:
            first_codes = first_codes.detach().cpu().tolist()
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
                if arrivals is not None:
                    self.rows[index].update(
                        arrived=arrivals[index], arrive_time=arrival_times[index] if arrivals[index] else None
                    )
                if first_codes is not None:
                    self.rows[index]["first_event_code"] = first_codes[index]

    @property
    def complete(self):
        return all(row is not None for row in self.rows)

    def summary(self):
        if not self.complete:
            raise ValueError("Evaluation contains unfinished episodes.")
        labels = ("success", "collision", "timeout", "bounds", "invalid")
        counts = {key: sum(row["outcome"] == key for row in self.rows) for key in labels}
        times = [row.get("arrive_time", row["time_s"]) for row in self.rows if row["outcome"] == "success"]
        count = len(self.rows)
        return {
            "episodes": count,
            "counts": counts,
            **{f"{label}_rate": counts[label] / count for label in labels},
            "success_time_mean_s": statistics.mean(times) if times else None,
            "success_time_median_s": statistics.median(times) if times else None,
            "success_time_std_s": statistics.stdev(times) if len(times) > 1 else None,
        }
