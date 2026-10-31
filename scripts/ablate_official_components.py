"""Run one-update ablations to locate the observed numerical drift."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate-commit", default="e911e964dc47b1a8d096bca952e52a07976a6696")
parser.add_argument("--output", type=Path, default=root / "logs/official_components_20261031")
options = parser.parse_args()
output = options.output.resolve()
baseline = output / "baseline"
spec = importlib.util.spec_from_file_location("_backend_benchmark", root / "scripts/benchmark_backends.py")
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
helper.NUM_ENVS = 32
runs = []
for label, filename in (
    ("delay_only", "randomization.py"),
    ("rays_only", "sensor_mixins.py"),
    ("mesh_only", "newton_sensors.py"),
):
    directory = output / "ablations" / label
    source = directory / "source"
    shutil.copytree(baseline, source)
    relative = Path("src/isaaclab_navrl/tasks/navigation") / filename
    (source / relative).write_bytes(
        subprocess.check_output(["git", "show", f"{options.candidate_commit}:{relative.as_posix()}"], cwd=root)
    )
    train = directory / "training"
    args = helper.training_arguments(train, "newton_mjwarp", "warp", 1, "cuda:0")
    args[args.index("--num_dynamic_obstacles") + 1] = "40"
    command = [sys.executable, str(source / "scripts/benchmark_backends.py"), "--_train_child", *args]
    env = os.environ.copy()
    env["NAVRL_STAGE_STARTED_MONOTONIC"] = str(time.monotonic())
    env["PYTHONPATH"] = str(source / "src")
    env["WARP_CACHE_PATH"] = str(output / "warp_cache")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(output / "inductor_cache")
    print("START", label, flush=True)
    with (directory / "console.log").open("w") as log:
        result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
    run = {
        "label": label,
        "candidate_commit": options.candidate_commit,
        "source": str(source),
        "changed_file": str(relative),
        "command": command,
        "exit": result.returncode,
    }
    if (train / "updates.jsonl").exists():
        run["updates"] = [json.loads(line) for line in (train / "updates.jsonl").read_text().splitlines() if line]
    if (train / "training_result.json").exists():
        run["result"] = json.loads((train / "training_result.json").read_text())
    runs.append(run)
    (output / "ablations.json").write_text(json.dumps(runs, indent=2) + "\n")
    print("FINISH", label, "exit", result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError(f"Inspect {directory}")
