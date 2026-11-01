"""The public evaluation entrypoint accepts the S2 pretraining difficulty."""

import os
import subprocess
import sys
from pathlib import Path


def test_pretrain_s2_reaches_checkpoint_validation_without_launching_simulation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/evaluate.py"),
            "--stage",
            "pretrain_s2",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--output",
            str(tmp_path / "evaluation.json"),
        ],
        env=os.environ | {"PYTHONPATH": str(root / "src"), "OMNI_KIT_ACCEPT_EULA": "YES"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "checkpoint does not exist" in result.stderr
    assert "invalid choice" not in result.stderr
