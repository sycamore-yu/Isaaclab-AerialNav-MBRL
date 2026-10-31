"""Public checkpoint export and timing calculations are independent of workstation state."""

import json

import pytest
import torch

from aerialnav.diffnav.model import DiffNavPolicy


def test_inference_export_strips_training_private_fields(tmp_path):
    from scripts.export_weights import export_weights

    model = DiffNavPolicy()
    source = {
        "format_version": 3,
        "method": "diffnav",
        "model_state_dict": model.state_dict(),
        "normalizer_state_dict": {},
        "train_cfg": {"recurrent_model": "gru"},
        "stage_log_dir": "PRIVATE_SENTINEL",
        "optimizer_state_dict": {"secret": torch.ones(1)},
        "environment": {"observation_contract": "test", "other": "PRIVATE_SENTINEL_2"},
    }
    path, output = tmp_path / "input.pt", tmp_path / "output.pt"
    torch.save(source, path)
    export_weights(path, output)
    got = torch.load(output, map_location="cpu", weights_only=True)
    assert set(got) == {
        "format_version", "method", "model_state_dict", "normalizer_state_dict", "train_cfg", "environment"
    }
    assert got["environment"] == {"observation_contract": "test"}
    with pytest.raises(FileExistsError):
        export_weights(path, output)


def test_speed_report_uses_process_wall_clock_and_completed_episodes(tmp_path):
    from scripts.benchmark_training import summarize

    stats, updates = tmp_path / "stats.jsonl", tmp_path / "updates.jsonl"
    stats.write_text("\n".join(json.dumps({
        "update": i, "static_obstacles": 300, "dynamic_obstacles": 0, "success_rate": rate,
        "episodes": 100, "counts": {"success": int(100 * rate)}
    }) for i, rate in ((1, .3), (2, .81), (3, .89))))
    updates.write_text("\n".join(json.dumps({
        "update": i, "process_elapsed_s": seconds, "environment_steps": i * 38400
    }) for i, seconds in ((1, 20), (2, 40), (3, 60))))
    result = summarize(stats, updates)
    assert result["time_to_threshold"]["80%"]["process_elapsed_s"] == 40
    assert result["best"]["success_rate"] == .89
    assert result["best"]["average_environment_steps_per_s"] == 1920.0
