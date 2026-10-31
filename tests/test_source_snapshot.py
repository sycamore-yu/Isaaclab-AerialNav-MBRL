"""Source snapshots survive stage handoff and reject source drift."""

import subprocess
import sys
from pathlib import Path

import pytest

from aerialnav.source import freeze_source, verify_source


def test_snapshot_survives_origin_changes_and_checks_every_stage(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    source = repo / "src" / "example.py"
    source.parent.mkdir()
    source.write_text("ANSWER = 42\n")
    package = source.parent / "aerialnav"
    package.mkdir()
    (package / "__init__.py").write_text("")
    import aerialnav.source as source_module

    original = Path(source_module.__file__).parent
    for name in ("source.py", "training.py"):
        (package / name).write_bytes((original / name).read_bytes())
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )
    frozen = tmp_path / "snapshot"
    identity = freeze_source(repo, frozen)
    import os

    environment = {
        **os.environ,
        "PYTHONPATH": str(frozen / "src"),
        "AERIALNAV_SOURCE_ROOT": str(frozen),
        "AERIALNAV_SOURCE_COMMIT": identity["commit"],
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    imported = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from aerialnav.source import verify_training_import; print(verify_training_import()['commit'])",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
    )
    assert imported.strip() == identity["commit"]
    source.write_text("ANSWER = 0\n")
    assert verify_source(frozen, identity["commit"]) == identity
    with pytest.raises(RuntimeError, match="Commit runtime"):
        freeze_source(repo, tmp_path / "dirty")
    with pytest.raises(RuntimeError, match="identity"):
        verify_source(frozen, "wrong")
    (frozen / "src" / "unexpected.py").write_text("pass\n")
    with pytest.raises(RuntimeError, match="Unrecorded"):
        verify_source(frozen, identity["commit"])
    (frozen / "src" / "unexpected.py").unlink()
    (frozen / "src" / "example.py").write_text("ANSWER = 0\n")
    with pytest.raises(RuntimeError, match="source changed"):
        verify_source(frozen, identity["commit"])
