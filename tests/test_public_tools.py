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
        "format_version",
        "method",
        "model_state_dict",
        "normalizer_state_dict",
        "train_cfg",
        "environment",
    }
    assert got["environment"] == {"observation_contract": "test"}
    with pytest.raises(FileExistsError):
        export_weights(path, output)


def test_speed_report_uses_process_wall_clock_and_completed_episodes(tmp_path):
    from scripts.benchmark_training import summarize

    stats, updates = tmp_path / "stats.jsonl", tmp_path / "updates.jsonl"
    stats.write_text(
        "\n".join(
            json.dumps(
                {
                    "update": i,
                    "static_obstacles": 300,
                    "dynamic_obstacles": 0,
                    "success_rate": rate,
                    "episodes": 100,
                    "counts": {"success": int(100 * rate)},
                }
            )
            for i, rate in ((1, 0.3), (2, 0.81), (3, 0.89))
        )
    )
    updates.write_text(
        "\n".join(
            json.dumps({"update": i, "process_elapsed_s": seconds, "environment_steps": i * 38400})
            for i, seconds in ((1, 20), (2, 40), (3, 60))
        )
    )
    result = summarize(stats, updates)
    assert result["time_to_threshold"]["80%"]["process_elapsed_s"] == 40
    assert result["best"]["success_rate"] == 0.89
    assert result["best"]["average_environment_steps_per_s"] == 1920.0


def test_release_audit_detects_legacy_names_in_publishing_docs(tmp_path):
    from scripts.audit_release import audit

    docs = tmp_path / "docs"
    docs.mkdir()
    publishing = docs / "publishing.md"
    old_repo = "isaaclab" + "-" + "".join(("nav", "rl"))
    publishing.write_text(f"git fetch /path/to/private/{old_repo}\n")
    findings = audit(tmp_path)
    assert any("docs/publishing.md" in item for item in findings)

    publishing.write_text("Use this private host: /home/" + "localuser" + "/private/repo\n")
    findings = audit(tmp_path)
    assert any("machine-specific path" in item for item in findings)

    publishing.unlink()
    upstream = "Nav" + "RL"
    (tmp_path / "README.md").write_text(
        f"- [{upstream}](https://github.com/Zhefan-Xu/NavRL) "
        f"([MIT](licenses/{upstream}.txt))\n"
    )
    assert audit(tmp_path) == []


def test_export_rejects_unrecognised_configuration_metadata(tmp_path):
    from scripts.export_weights import export_weights

    source = {
        "format_version": 3,
        "method": "diffnav",
        "model_state_dict": {},
        "normalizer_state_dict": {},
        "train_cfg": {"private_path": "PRIVATE_SENTINEL"},
    }
    path = tmp_path / "source.pt"
    torch.save(source, path)
    with pytest.raises(ValueError, match="Unsupported public training configuration"):
        export_weights(path, tmp_path / "export.pt")


def test_license_layout_is_packaged_and_keeps_credits_separate_from_code(tmp_path):
    import tomllib
    from pathlib import Path

    from scripts.audit_release import audit

    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    included = {path for pattern in config["project"]["license-files"] for path in root.glob(pattern)}
    assert root / "LICENSE" in included
    assert set((root / "licenses").glob("*.txt")) <= included
    assert len(list((root / "licenses").glob("*.txt"))) == 4
    readme = (root / "README.md").read_text()
    assert all(f"(licenses/{path.name})" in readme for path in (root / "licenses").glob("*.txt"))

    name = "Nav" + "RL"
    licenses = tmp_path / "licenses"
    licenses.mkdir()
    (licenses / f"{name}.txt").write_text(f"{name} MIT license\n")
    source = tmp_path / "src"
    source.mkdir()
    path = source / "policy.py"
    path.write_text(f"# See licenses/{name}.txt\n")
    assert audit(tmp_path) == []
    path.write_text(f"class {name}Policy: pass\n")
    assert any("unexpected upstream project name" in item for item in audit(tmp_path))
