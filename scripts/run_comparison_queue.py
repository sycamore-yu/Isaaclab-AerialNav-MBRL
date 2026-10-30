#!/usr/bin/env python3
"""Fresh comparisons with capacity admission; signals stop dispatch, not children."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from isaaclab_navrl.physics import DEFAULT_FRONTEND, DEFAULT_PHYSICS, parse_physics_args, validate_runtime
from isaaclab_navrl.source import RUNTIME_PATHS
from isaaclab_navrl.training import write_json


def now():
    return datetime.now(UTC).isoformat()


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def train_tokens(run):
    tokens = []
    for key, value in run.get("train_args", {}).items():
        tokens += ["--" + key, str(int(value)) if type(value) is bool else str(value)]
    return tokens


def command_for(source, output, plan, run, device, physics=DEFAULT_PHYSICS, frontend=None):
    if frontend is None:
        frontend = "torch" if physics == "isaacsim_physx" else DEFAULT_FRONTEND
    validate_runtime(physics, frontend)
    command = [
        sys.executable,
        str(source / "scripts/run_acceptance.py"),
        "--method",
        run["method"],
        "--output",
        str(output),
        "--screening",
        "--num_envs",
        str(plan["num_envs"]),
        "--train_seed",
        str(plan["seed"]),
        "--device",
        device,
        f"physics={physics}",
        "--frontend",
        frontend,
        "--finetune_lr",
        str(plan.get("finetune_lr", 5e-5)),
    ]
    if "report_episodes" in plan:
        command += ["--report_episodes", str(plan["report_episodes"])]
    command += ["--start_stage", run.get("start_stage", plan.get("start_stage", "finetune_s1"))]
    if plan.get("fixed_stage", True):
        command += ["--fixed_stage"]
    tokens = train_tokens(run)
    if tokens:
        command += ["--train_args", *tokens]
    return command


def estimated_mib(run):
    if run["method"] == "navrl":
        return 3584
    return 5120 if run.get("train_args", {}).get("algorithm") == "ahac" else 3584


def admissible(active_count, all_ready, free_mib, estimate_mib, reserve_mib=2048, slots=None):
    return (slots is None or active_count < slots) and all_ready and free_mib >= estimate_mib + reserve_mib


def process_start_ticks(pid, command=None):
    """Verify PID ownership before adopting an existing pipeline."""
    path = Path("/proc") / str(pid)
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        if command is not None:
            actual = (path / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            if actual != command:
                raise ValueError(f"Pipeline PID {pid} no longer matches its recorded command")
        return int(fields[19])
    except FileNotFoundError:
        return None


def adopt_running_jobs(previous, jobs):
    if set(previous["runs"]) != set(jobs):
        raise ValueError("Resume requires the same experiment matrix")
    if process_start_ticks(previous["queue_pid"]) is not None:
        raise ValueError("Previous dispatcher must exit before takeover")
    active = {}
    for name, job in jobs.items():
        old = previous["runs"][name]
        if old["command"] != job["command"]:
            raise ValueError(f"Resume cannot change training arguments: {name}")
        estimate = job["estimated_mib"]
        job.update(old, estimated_mib=estimate)
        job.setdefault("source_commit", previous["source_commit"])
        if old["status"] == "running":
            ticks = process_start_ticks(old["pid"], old["command"])
            if ticks is None:
                raise ValueError(f"Cannot adopt missing pipeline: {name}")
            job.update(adopted=True, process_start_ticks=ticks)
            active[name] = old["pid"]
    return active


def check_source(source, expected_commit):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=source, text=True, timeout=10).strip()

    if git("rev-parse", "HEAD") != expected_commit:
        raise RuntimeError("Source HEAD differs from pinned commit")
    if git("status", "--porcelain", "--untracked-files=all", "--", *RUNTIME_PATHS):
        raise RuntimeError("Runtime source worktree is not clean")


def free_memory_mib(gpu):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    lines = output.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError("GPU selector must identify exactly one GPU")
    return int(lines[0].strip())


def current_stage_ready(job):
    output = Path(job["output"])
    parent = read_json(output / "training_result.json")
    if parent.get("status") != "running":
        return False
    stages = parent.get("methods", {}).get(job["method"], {}).get("stages", [])
    if not stages or stages[-1].get("status") != "running":
        return False
    stage = stages[-1].get("stage")
    if stage not in ("pretrain_s0", "pretrain_s1", "pretrain_s2", "finetune_s1", "finetune_s2"):
        return False
    result = read_json(output / job["method"] / stage / "training_result.json")
    updates = result.get("stage_updates")
    return result.get("status") == "running" and result.get("stage") == stage and type(updates) is int and updates >= 1


def launched_zhang_manifest(jobs, plan):
    """Emit only launched Zhang identities; queued jobs and PPO are not evidence."""
    entries = []
    fields = (
        "variant",
        "algorithm",
        "recurrent_model",
        "lr_schedule",
        "velocity_loss_mode",
        "collision_metric",
        "approach_mode",
        "lateral_avoidance",
        "risk_speed_floor",
        "height_loss_mode",
        "optimizer",
        "weight_decay",
        "gradient_projection",
        "plateau_patience",
        "ahac_max_horizon",
        "timing",
        "num_dynamic_obstacles",
        "target_speed_min",
        "target_speed_max",
    )
    for job in jobs.values():
        if job["method"] != "zhang" or job.get("pid") is None:
            continue
        config = job["train_args"]
        entries.append(
            {
                "name": job["name"],
                "path": job["path"],
                "seed": plan["seed"],
                "normalization": config.get("observation_normalization", False),
                "num_envs": plan["num_envs"],
                "start_stage": job.get("start_stage", plan.get("start_stage", "finetune_s1")),
                **{key: config[key] for key in fields if key in config},
            }
        )
    return {"purpose": "launched_zhang_only", "runs": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/experiments/zhang_baseline_fs1_dyn40.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/curriculum") / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_comparison"),
    )
    parser.add_argument("--commit", required=True, help="Full frozen commit SHA")
    parser.add_argument("--source_root", type=Path, help="Frozen training worktree; defaults to dispatcher tree")
    parser.add_argument("--gpu", default="0", help="Physical nvidia-smi GPU selector")
    parser.add_argument("--device", default="cuda:0", help="Trainer CUDA device")
    parser.add_argument("--slots", type=int, default=None, help="Optional limit; default admits by VRAM only")
    parser.add_argument("--resume", action="store_true", help="Take over a stopped dispatcher's live pipelines")
    parser.add_argument("--reserve_mib", type=int, default=2048)
    args = parse_physics_args(parser)
    dispatcher = Path(__file__).resolve().parents[1]
    dispatcher_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=dispatcher,
        text=True,
        timeout=10,
    ).strip()
    check_source(dispatcher, dispatcher_commit)
    source = args.source_root.expanduser().resolve() if args.source_root else dispatcher
    check_source(source, args.commit)
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    if (
        plan.get("num_envs") != 256
        or plan.get("screening") is not True
        or "max_updates" not in plan
        or plan["max_updates"] is not None
        or args.reserve_mib < 512
        or (args.slots is not None and args.slots < 1)
    ):
        parser.error("Requires N256 screening, unlimited updates, reserve >= 512 MiB and positive optional slots")

    protocol = runpy.run_path(str(source / "scripts/run_acceptance.py"))
    if tuple(plan["stage_order"]) != protocol["STAGES"]:
        parser.error("Plan must preserve the complete five-stage order")
    runs = plan["runs"]
    names = [run["name"] for run in runs]
    if not runs or len(names) != len(set(names)):
        parser.error("Run names must be nonempty and unique")
    order = plan.get("launch_order", ["dcol_truncated", "navrl_ppo"])
    if len(order) != len(set(order)) or any(name not in names for name in order):
        parser.error("Invalid launch_order")
    order = [*order, *(name for name in names if name not in order)]
    by_name = {run["name"]: run for run in runs}
    output = args.output.expanduser().resolve()
    jobs, targets = {}, set()
    for name in order:
        run = by_name[name]
        if Path(name).name != name or name in ("", ".", ".."):
            parser.error("Run name must be a simple filename")
        if run["method"] not in ("zhang", "navrl"):
            parser.error("Unsupported method")
        protocol["parse_train_args"](train_tokens(run))
        run["start_stage"] = run.get("start_stage", plan.get("start_stage", "finetune_s1"))
        protocol["selected_stages"](
            run["start_stage"],
            screening=plan["screening"],
            method=run["method"],
            fixed_stage=plan.get("fixed_stage", True),
        )
        if run["method"] == "navrl" and run.get("train_args"):
            parser.error("PPO must not receive Zhang train_args")
        target = (output / run["path"]).resolve()
        if target == output or not target.is_relative_to(output) or target in targets:
            parser.error("Run outputs must be distinct descendants of queue output")
        targets.add(target)
        jobs[name] = {
            **run,
            "output": str(target),
            "status": "queued",
            "pid": None,
            "estimated_mib": estimated_mib(run),
            "physics": args.physics_preset,
            "frontend": args.frontend,
            "command": command_for(source, target, plan, run, args.device, args.physics_preset, args.frontend),
        }

    previous = read_json(output / "queue.json") if args.resume else None
    if args.resume and not previous:
        parser.error("Resume requires an existing queue.json")
    active = adopt_running_jobs(previous, jobs) if previous else {}
    output.mkdir(parents=True, exist_ok=args.resume)
    if previous:
        write_json(output / f"queue_before_takeover_{os.getpid()}.json", previous)
        (output / f"plan_before_takeover_{os.getpid()}.json").write_bytes((output / "plan.json").read_bytes())
    write_json(output / "plan.json", plan)
    environment = os.environ.copy()
    environment.update(
        NAVRL_EXPECTED_COMMIT=args.commit,
        PYTHONPATH=str(source / "src"),
        PYTHONUNBUFFERED="1",
        OMNI_KIT_ACCEPT_EULA="YES",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
    )
    environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    state = {
        "status": "running",
        "created_at": now(),
        "queue_pid": os.getpid(),
        "source": str(source),
        "source_commit": args.commit,
        "source_clean_at_start": True,
        "dispatcher_source": str(dispatcher),
        "dispatcher_commit": dispatcher_commit,
        "dispatcher_clean_at_start": True,
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "gpu": args.gpu,
        "device": args.device,
        "physics": args.physics_preset,
        "frontend": args.frontend,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slots": args.slots,
        "reserve_mib": args.reserve_mib,
        "launch_order": order,
        "runs": jobs,
        "stop_dispatch_reason": None,
    }
    state_path = output / "queue.json"
    if previous:
        state.update(
            previous_dispatcher_commit=previous["dispatcher_commit"],
            resumed_at=now(),
            previous_queue_pid=previous["queue_pid"],
        )
    failed = False

    def stop_dispatch(signum, _frame):
        state["stop_dispatch_reason"] = f"signal:{signum}"
        state["stop_requested_at"] = now()

    signal.signal(signal.SIGTERM, stop_dispatch)
    signal.signal(signal.SIGINT, stop_dispatch)
    write_json(state_path, state)

    while True:
        for name, process in list(active.items()):
            job = jobs[name]
            if isinstance(process, int):
                ticks = process_start_ticks(process)
                if ticks == job["process_start_ticks"]:
                    continue
                code = None  # An adopted process is not our child; its exit code is unavailable.
            else:
                code = process.poll()
                if code is None:
                    continue
            result = read_json(Path(job["output"]) / "training_result.json")
            status = result.get("status")
            valid = (
                (code == 0 or isinstance(process, int))
                and status in ("completed", "stopped")
                and result.get("method") == job["method"]
                and result.get("screening") is True
                and result.get("num_envs") == 256
                and result.get("acceptance_passed") is False
                and result.get("error") is None
                and (status != "completed" or result.get("stage_thresholds_passed") is True)
            )
            job.update(
                status=status if valid else "failed",
                exit_code=code,
                finished_at=now(),
                result_status=status,
                error=result.get("error"),
            )
            del active[name]
            if not valid:
                failed = True
                state["stop_dispatch_reason"] = f"child_failure:{name}"

        pending = [name for name in order if jobs[name]["status"] == "queued"]
        if not active and (not pending or state["stop_dispatch_reason"]):
            state["status"] = "dispatch_stopped" if state["stop_dispatch_reason"] else "finished"
            state["finished_at"] = now()
            write_json(state_path, state)
            return 1 if failed else 0

        if pending and not state["stop_dispatch_reason"] and (args.slots is None or len(active) < args.slots):
            try:
                ready = all(current_stage_ready(jobs[name]) for name in active)
                if ready:
                    name = pending[0]
                    job = jobs[name]
                    free = free_memory_mib(args.gpu)
                    state["last_capacity"] = {"at": now(), "free_mib": free, "candidate": name}
                    if admissible(len(active), ready, free, job["estimated_mib"], args.reserve_mib, args.slots):
                        check_source(source, args.commit)
                        check_source(dispatcher, dispatcher_commit)
                        if Path(job["output"]).exists():
                            raise RuntimeError(f"Existing run output: {job['output']}")
                        if not state["stop_dispatch_reason"] and all(
                            current_stage_ready(jobs[item]) for item in active
                        ):
                            job.update(
                                status="starting",
                                started_at=now(),
                                free_before_mib=free,
                                concurrent_pipelines_at_launch=len(active) + 1,
                            )
                            write_json(state_path, state)
                            log_path = output / f"{name}.stdout.log"
                            job["stdout_log"] = str(log_path)
                            with log_path.open("xb") as stream:
                                process = subprocess.Popen(
                                    job["command"],
                                    cwd=source,
                                    env=environment,
                                    stdout=stream,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=True,
                                )
                            active[name] = process
                            job.update(status="running", pid=process.pid, process_group=process.pid)
                            job["source_commit"] = args.commit
                            write_json(output / "zhang_manifest.json", launched_zhang_manifest(jobs, plan))
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                failed = True
                state["stop_dispatch_reason"] = f"dispatch_error:{error}"
                for job in jobs.values():
                    if job["status"] == "starting":
                        job.update(status="failed", error=str(error))

        state["status"] = "draining" if state["stop_dispatch_reason"] else "running"
        state["updated_at"] = now()
        write_json(state_path, state)
        time.sleep(15)


if __name__ == "__main__":
    raise SystemExit(main())
