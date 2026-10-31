"""Measure the hardest stage, choose one common env count, then run all four arms."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_comparison_queue import free_memory_mib

from isaaclab_navrl.source import freeze_source
from isaaclab_navrl.training import write_json


def four_fit(peak_mib, free_mib):
    return 4 * (peak_mib + 256) + 2048 <= free_mib


def group_gpu_memory(group):
    result = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True, timeout=10
    )
    total = 0
    for row in result.splitlines():
        pid, used = (int(value.strip()) for value in row.split(","))
        try:
            if os.getpgid(pid) == group:
                total += used
        except ProcessLookupError:
            pass
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = freeze_source(root, output / "probe_source")
    source = Path(identity["root"])
    plan = json.loads((root / "configs/experiments/zhang_rectangular_gradient_2x2.json").read_text())
    env = os.environ.copy()
    env.update(
        OMNI_KIT_ACCEPT_EULA="YES",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(source / "src"),
        NAVRL_SOURCE_ROOT=str(source),
        NAVRL_SOURCE_COMMIT=identity["commit"],
        PYTORCH_ALLOC_CONF="expandable_segments:True",
    )
    attempts = []
    count = plan["num_envs"]
    while count >= 64:
        path = output / f"capacity_n{count}"
        free_before = free_memory_mib("0")
        command = [
            sys.executable,
            str(source / "scripts/train_zhang.py"),
            "--log_dir",
            str(path),
            "--stage",
            "finetune_s2",
            "--fresh_stage",
            "--num_envs",
            str(count),
            "--max_iterations",
            "2",
            "--report_episodes",
            "1024",
            "--avoidance_gradient",
            "reshaped",
            "--seed",
            "0",
            "physics=newton_mjwarp",
            "--frontend",
            "warp",
        ]
        peak = 0
        with (output / f"capacity_n{count}.stdout.log").open("xb") as stream:
            process = subprocess.Popen(
                command, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
            while process.poll() is None:
                peak = max(peak, group_gpu_memory(process.pid))
                time.sleep(1)
        free = min(free_before, free_memory_mib("0"))
        result = (
            json.loads((path / "training_result.json").read_text()) if (path / "training_result.json").exists() else {}
        )
        log = (output / f"capacity_n{count}.stdout.log").read_text()
        oom = any(text in log.lower() for text in ("out of memory", "cuda_error_out_of_memory"))
        attempt = dict(
            num_envs=count,
            peak_process_group_gpu_mib=peak,
            free_mib=free,
            exit_code=process.returncode,
            completed_updates=result.get("completed_updates"),
            oom=oom,
            command=command,
        )
        attempts.append(attempt)
        write_json(output / "capacity.json", dict(source_commit=identity["commit"], attempts=attempts))
        if process.returncode != 0 and not oom:
            raise RuntimeError(f"Capacity probe failed for a reason other than OOM: {path}")
        if process.returncode == 0:
            if result.get("completed_updates") != 2 or peak <= 0 or result.get("nonfinite_grad_events", 0):
                raise RuntimeError(f"Incomplete or nonfinite capacity probe: {path}")
            if four_fit(peak, free):
                break
        count //= 2
    else:
        raise RuntimeError("Insufficient VRAM for four arms even at N64; all failed probes are retained")
    plan["num_envs"] = count
    for run in plan["runs"]:
        run["estimated_mib"] = peak + 256
    plan["capacity_evidence"] = str(output / "capacity.json")
    write_json(output / "selected_plan.json", plan)
    # Each pipeline freezes its own committed main source; probe weights are never reused.
    env.pop("NAVRL_SOURCE_ROOT")
    env.pop("NAVRL_SOURCE_COMMIT")
    env["PYTHONPATH"] = str(root / "src")
    command = [
        sys.executable,
        str(root / "scripts/run_comparison_queue.py"),
        "--plan",
        str(output / "selected_plan.json"),
        "--output",
        str(output / "runs"),
        "--commit",
        identity["commit"],
        "--slots",
        "4",
        "--reserve_mib",
        "2048",
    ]
    write_json(
        output / "launch.json",
        dict(
            source_commit=identity["commit"],
            num_envs=count,
            report_episodes=1024,
            maximum_training_updates=None,
            slots=4,
            command=command,
        ),
    )
    raise SystemExit(subprocess.call(command, cwd=root, env=env))


if __name__ == "__main__":
    main()
