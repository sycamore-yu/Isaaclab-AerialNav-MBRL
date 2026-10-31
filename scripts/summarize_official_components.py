"""Summarize completed ABBA evidence and compare final checkpoints on CPU."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, f"{prefix}/{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from flatten(child, f"{prefix}/{index}")
    else:
        yield prefix, value


def tensor_comparison(left, right):
    lhs = {key: value for key, value in flatten(left) if torch.is_tensor(value)}
    rhs = {key: value for key, value in flatten(right) if torch.is_tensor(value)}
    assert lhs.keys() == rhs.keys()
    squared_error, squared_reference, elements, different = 0.0, 0.0, 0, 0
    largest, largest_key = 0.0, None
    for key, value in lhs.items():
        other = rhs[key]
        assert value.shape == other.shape and value.dtype == other.dtype, key
        assert torch.isfinite(value).all() and torch.isfinite(other).all(), key
        delta = value.double() - other.double()
        maximum = float(delta.abs().max()) if delta.numel() else 0.0
        if maximum > largest:
            largest, largest_key = maximum, key
        squared_error += float(delta.square().sum())
        squared_reference += float(value.double().square().sum())
        elements += delta.numel()
        different += int(torch.count_nonzero(delta))
    return {
        "tensor_count": len(lhs),
        "elements": elements,
        "nonidentical_elements": different,
        "max_abs_error": largest,
        "max_error_key": largest_key,
        "rmse": math.sqrt(squared_error / elements) if elements else 0.0,
        "relative_l2_error": math.sqrt(squared_error / squared_reference) if squared_reference else None,
    }


def summarize(path, final_candidate=None):
    data = json.loads(path.read_text())
    assert data["complete"] and len(data["runs"]) == 4
    if final_candidate is not None:
        final = json.loads(final_candidate.read_text())
        assert final["complete"] and len(final["runs"]) == 1
        for key in ("num_envs", "horizon", "warmup_updates"):
            assert final[key] == data[key], key
        final["runs"][0]["label"] = "final_candidate"
        data["runs"].extend(final["runs"])
    warmup = data["warmup_updates"]
    baseline_config = dict(flatten(data["runs"][0]["config.json"]))
    runs, checkpoints, groups = [], [], {"baseline": [], "candidate": []}
    if final_candidate is not None:
        groups["final_candidate"] = []
    for index, run in enumerate(data["runs"]):
        result, updates = run["training_result.json"], run["updates"]
        assert run["returncode"] == 0 and result["status"] == "completed"
        assert result["nonfinite_events"] == result["nonfinite_grad_events"] == 0
        assert all(item["executed_action_max_error"] == 0 for item in updates)
        measured = [item["update_s"] for item in updates[warmup:]]
        groups[run["label"]].extend(measured)
        config = dict(flatten(run["config.json"]))
        differences = {
            key: [baseline_config.get(key), config.get(key)]
            for key in baseline_config.keys() | config.keys()
            if baseline_config.get(key) != config.get(key)
        }
        runs.append(
            {
                "index": index,
                "label": run["label"],
                "measured_update_s": measured,
                "median_update_s": statistics.median(measured),
                "environment_steps_per_second": data["num_envs"] * data["horizon"] / statistics.mean(measured),
                "torch_peak_allocated_mib": result["torch_peak_allocated_mb"],
                "torch_peak_reserved_mib": max(item["Memory/torch_peak_reserved_mb"] for item in updates),
                "sampled_process_peak_mib": run["peak_process_memory_mib"],
                "losses": [item["total"] for item in updates],
                "gradient_norms": [item["grad_norm"] for item in updates],
                "completed_episodes": [item["completed_episodes"] for item in updates],
                "config_differences_from_run_0": differences,
            }
        )
        checkpoints.append(torch.load(Path(result["checkpoint"]), map_location="cpu", weights_only=False))
    comparisons = []
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (3, 1), (3, 2)]
    if final_candidate is not None:
        pairs.extend([(0, 4), (3, 4)])
    for left, right in pairs:
        a, b = checkpoints[left], checkpoints[right]
        comparisons.append(
            {
                "runs": [left, right],
                "model": tensor_comparison(a["model_state_dict"], b["model_state_dict"]),
                "optimizer": tensor_comparison(a["optimizer_state_dict"], b["optimizer_state_dict"]),
                "torch_rng_equal": torch.equal(a["torch_rng_state"], b["torch_rng_state"]),
                "cuda_rng_equal": torch.equal(a["cuda_rng_state"], b["cuda_rng_state"]),
                "max_loss_difference": max(
                    abs(x - y) for x, y in zip(runs[left]["losses"], runs[right]["losses"], strict=True)
                ),
            }
        )
    return {
        "protocol": data["protocol"],
        "final_candidate_included": final_candidate is not None,
        "nominal_throughput_scope": "initial ABBA only",
        "warmup_updates_per_run": warmup,
        "total_training_environment_steps": sum(
            run["training_result.json"]["environment_steps"] for run in data["runs"]
        ),
        "runs": runs,
        "groups": {
            label: {
                "median_update_s": statistics.median(values),
                "mean_update_s": statistics.mean(values),
                "environment_steps_per_second": data["num_envs"] * data["horizon"] / statistics.mean(values),
            }
            for label, values in groups.items()
        },
        "nominal_throughput_change_percent": (
            statistics.mean(groups["baseline"]) / statistics.mean(groups["candidate"]) - 1
        )
        * 100,
        "checkpoint_comparisons": comparisons,
        "interpretation": "Short shared-GPU experiment; small timing differences are not proof of a speedup. Numerical closeness is not bitwise or convergence equivalence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-candidate", type=Path)
    args = parser.parse_args()
    report = summarize(args.input, args.final_candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
