"""Freeze committed training code and verify stage imports against that snapshot."""

import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

from .training import write_json

RUNTIME_PATHS = ("src", "scripts", "configs", "pyproject.toml", "uv.lock")


def freeze_source(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *RUNTIME_PATHS],
        cwd=source,
        text=True,
    )
    if status:
        raise RuntimeError("Commit runtime source changes before starting a pipeline")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    archive = subprocess.check_output(["git", "archive", commit], cwd=source)
    target.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(target, filter="data")
    hashes = {
        str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob("*") if p.is_file()
    }
    identity = {"commit": commit, "root": str(target), "files": hashes}
    write_json(target / "source_identity.json", identity)
    return identity


def verify_source(root, expected_commit):
    root = Path(root).resolve()
    identity = json.loads((root / "source_identity.json").read_text())
    if identity["commit"] != expected_commit or identity["root"] != str(root):
        raise RuntimeError("Training source identity differs from the pipeline snapshot")
    for name, expected in identity["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Training source changed: {name}")
    for directory in ("src", "scripts"):
        if any(str(p.relative_to(root)) not in identity["files"] for p in (root / directory).rglob("*.py")):
            raise RuntimeError(f"Unrecorded Python source in {directory}")
    return identity


def verify_training_import():
    """Queued stages must import this package from their immutable source root."""
    root = os.environ.get("AERIALNAV_SOURCE_ROOT")
    if root is None:
        return None
    root = Path(root).resolve()
    if Path(__file__).resolve().parent != root / "src" / "aerialnav":
        raise RuntimeError("Imported aerialnav outside the pipeline source snapshot")
    return verify_source(root, os.environ["AERIALNAV_SOURCE_COMMIT"])
