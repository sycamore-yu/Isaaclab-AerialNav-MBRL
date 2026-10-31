"""Run four fresh original Zhang depth-policy timing/height-reset controls."""

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from analyze_liu_loss_timing import write_report
from run_comparison_queue import admissible, free_memory_mib, process_start_ticks

from isaaclab_navrl.source import freeze_source
from isaaclab_navrl.training import write_json


def matrix(plan):
    return (
        list(plan["runs"]) if "runs" in plan else list(itertools.product(plan["timings"], plan["height_bounds_reset"]))
    )


def verify_pair(left, right):
    configs = [json.loads((path / "config.json").read_text()) for path in (left, right)]
    train = [{k: v for k, v in cfg["train"].items() if k != "loss_timing"} for cfg in configs]
    if train[0] != train[1] or configs[0]["environment"] != configs[1]["environment"]:
        raise RuntimeError("Timing pair changed settings beyond loss_timing")
    hashes = [json.loads((path / "initialization.json").read_text())["model_sha256"] for path in (left, right)]
    if hashes[0] != hashes[1]:
        raise RuntimeError("Timing pair initialized different policy weights")
    policies = [json.loads((path / "initialization.json").read_text()) for path in (left, right)]
    if any(p["policy_class"] != "ZhangPolicy" or p["perception"] != "depth" or not p["fresh"] for p in policies):
        raise RuntimeError("Requested original depth Zhang policy with fresh initialization")
    return {
        "initial_model_sha256": hashes[0],
        "only_training_difference": "loss_timing",
        "policy_class": "ZhangPolicy",
        "fresh": True,
    }


def verify_factorial(root, plan):
    model_hashes, world_hashes, configs, environments = set(), set(), [], []
    for run in plan["runs"]:
        directory = root / run["name"]
        config = json.loads((directory / "config.json").read_text())
        initial = json.loads((directory / "initialization.json").read_text())
        world = json.loads((directory / "initial_rollout.json").read_text())
        if (
            initial["policy_class"] != "ZhangPolicy"
            or initial["perception"] != "depth"
            or not initial["fresh"]
            or config["parent_checkpoint"] is not None
        ):
            raise ValueError("Four-cell controls require fresh original Zhang depth policies")
        if any(config["train"][name] != run[name] for name in ("collision_metric", "position_integration")):
            raise ValueError("Actual training factors differ from the plan")
        configs.append(
            {k: v for k, v in config["train"].items() if k not in ("collision_metric", "position_integration")}
        )
        environments.append(config["environment"])
        model_hashes.add(initial["model_sha256"])
        world_hashes.add(world["world_sha256"])
    if (
        len(model_hashes) != 1
        or len(world_hashes) != 1
        or any(c != configs[0] for c in configs)
        or any(e != environments[0] for e in environments)
    ):
        raise ValueError("Four-cell initial models, worlds or other settings do not match")
    return {
        "same_initial_model_sha256": model_hashes.pop(),
        "same_initial_world_sha256": world_hashes.pop(),
        "only_train_differences": ["collision_metric", "position_integration"],
        "cells": len(plan["runs"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tuning_controller_pid", type=int)
    parser.add_argument(
        "--plan", type=Path, default=Path("configs/experiments/zhang_collision_integration_512_20261031.json")
    )
    args = parser.parse_args()
    root = args.output.resolve()
    checkout = Path(__file__).resolve().parents[1]
    plan = json.loads((checkout / args.plan).read_text())
    if plan["budget_interpretation"] == "awaiting_user_512_updates_or_horizon":
        raise ValueError("Confirm whether 512 denotes optimizer updates or rollout horizon before launch")
    root.mkdir(parents=True, exist_ok=False)
    identity = freeze_source(checkout, root / "source")
    source = Path(identity["root"])
    write_json(root / "plan.json", plan | {"source": identity["commit"]})
    environment = os.environ | {
        "PYTHONPATH": str(source / "src"),
        "NAVRL_SOURCE_ROOT": str(source),
        "NAVRL_SOURCE_COMMIT": identity["commit"],
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    state = {"pid": os.getpid(), "source_commit": identity["commit"], "status": "controls", "runs": {}}
    tuning = None
    if args.tuning_controller_pid:
        pid = args.tuning_controller_pid
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        if "run_weight_agent_loop.py" not in command[1]:
            raise ValueError("Priority gate requires the adaptive tuning controller, never a training process")
        tuning = {"pid": pid, "command": command, "start_ticks": process_start_ticks(pid, command), "held": True}
        os.kill(pid, signal.SIGSTOP)
        state["tuning_gate"] = tuning
    active = {}

    def persist():
        write_json(root / "study.json", state)

    def release_tuning():
        if tuning and tuning["held"]:
            if process_start_ticks(tuning["pid"], tuning["command"]) != tuning["start_ticks"]:
                raise RuntimeError("Tuning controller identity changed")
            os.kill(tuning["pid"], signal.SIGCONT)
            tuning.update(held=False, resumed_epoch=time.time())
            persist()

    def launch(name, timing, height_reset, control=None):
        output = root / name
        entry = "train_dcol_reference.py" if control and control["collision_metric"] == "dcol" else "train_zhang.py"
        command = [
            sys.executable,
            str(source / "scripts" / entry),
            "--log_dir",
            str(output),
            "--loss_timing",
            timing,
            "--height_bounds_reset",
            str(int(height_reset)),
            "--num_envs",
            str(plan["num_envs"]),
            "--timesteps",
            str(plan["timesteps"]),
            "--max_iterations",
            str(plan["updates"]),
            "--stage",
            plan["stage"],
            "--fresh_stage",
            "--seed",
            str(plan["seed"]),
            "--terrain_seed",
            str(plan["terrain_seed"]),
            "--variant",
            "dcol",
            "--algorithm",
            "bptt",
            "--recurrent_model",
            "gru",
            "--lr",
            str(plan["lr"]),
            "--lr_schedule",
            plan["lr_schedule"],
            "--cosine_period_updates",
            str(plan["cosine_period_updates"]),
            "--device",
            "cuda:0",
            f"physics={plan['physics']}",
            "--frontend",
            plan["frontend"],
        ]
        if control:
            command += ["--position_integration", control["position_integration"]]
        command += ["--geometry_backend", "native"]
        if plan.get("gradient_diagnostic_updates"):
            command += ["--gradient_diagnostic_updates", ",".join(map(str, plan["gradient_diagnostic_updates"]))]
        with (root / (name + ".log")).open("x") as log:
            process = subprocess.Popen(
                command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        state["runs"][name] = {
            "status": "training",
            "pid": process.pid,
            "command": command,
            "output": str(output),
            "timing": timing,
            "height_reset": height_reset,
            "started_epoch": time.time(),
            "control": control,
        }
        active[name] = process
        persist()

    def start_evaluation(name):
        job = state["runs"][name]
        command = [
            sys.executable,
            str(source / "scripts/evaluate.py"),
            "--geometry_backend",
            "native",
            "--method",
            "zhang",
            "--checkpoint",
            str(Path(job["output"]) / "checkpoint_final.pt"),
            "--output",
            str(root / (name + "_evaluation.json")),
            "--stage",
            plan["stage"],
            "--episodes",
            str(plan["evaluation_episodes"]),
            "--seed",
            str(plan["evaluation_seed"]),
            "--terrain_seed",
            str(plan["evaluation_terrain_seed"]),
            "--device",
            "cuda:0",
            f"physics={plan['physics']}",
            "--frontend",
            plan["frontend"],
        ]
        with (root / (name + "_evaluation.log")).open("x") as log:
            process = subprocess.Popen(
                command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        job.update(status="evaluating", evaluation_pid=process.pid, evaluation_command=command)
        active[name] = process

    def poll():
        for name, process in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            del active[name]
            job = state["runs"][name]
            if job["status"] == "training":
                path = Path(job["output"]) / "training_result.json"
                result = json.loads(path.read_text()) if path.exists() else {}
                job.update(training_returncode=code, training_result=result)
                if code == 0 and result.get("status") == "completed" and result.get("stage_updates") == plan["updates"]:
                    start_evaluation(name)
                else:
                    job["status"] = "failed"
            else:
                job.update(evaluation_returncode=code, status="completed" if code == 0 else "failed")
            persist()

    def ready():
        return all(
            job["status"] != "training"
            or (Path(job["output"]) / "updates.jsonl").exists()
            and (Path(job["output"]) / "updates.jsonl").stat().st_size > 0
            for name, job in state["runs"].items()
            if name in active
        )

    try:
        last_report = 0
        pending = matrix(plan)
        while pending or active:
            poll()
            free = free_memory_mib(0)
            if pending and admissible(
                len(active), ready(), free, plan["estimated_training_mib"], plan["reserve_mib"], plan["slots"]
            ):
                control = pending.pop(0)
                if isinstance(control, dict):
                    launch(control["name"], plan["loss_timing"], plan["height_bounds_reset"], control)
                else:
                    timing, height = control
                    name = f"train_zhang_{timing}_height{'reset' if height else 'continue'}"
                    launch(name, timing, height)
            if not pending and ready() and free >= plan["estimated_training_mib"] + plan["reserve_mib"]:
                release_tuning()
            state.update(free_gpu_mib=free, active=list(active), pending=len(pending), heartbeat_epoch=time.time())
            persist()
            if time.monotonic() - last_report > 30:
                write_report(root)
                last_report = time.monotonic()
            time.sleep(2)
        if "runs" in plan:
            write_json(root / "factorial_verification.json", verify_factorial(root, plan))
        else:
            for height in plan["height_bounds_reset"]:
                pair = [
                    root / f"train_zhang_{timing}_height{'reset' if height else 'continue'}"
                    for timing in plan["timings"]
                ]
                write_json(root / f"zhang_height{'reset' if height else 'continue'}_pair.json", verify_pair(*pair))
        state["status"] = (
            "completed" if all(job["status"] == "completed" for job in state["runs"].values()) else "failed"
        )
        persist()
        write_report(root)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        persist()
        write_report(root)
        raise
    finally:
        release_tuning()


if __name__ == "__main__":
    main()
