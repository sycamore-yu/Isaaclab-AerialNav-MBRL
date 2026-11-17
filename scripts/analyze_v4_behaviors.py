#!/usr/bin/env python3
"""Paper-grade behavioral & ablation visualizer for v4 experiments.

Produces:
  1. summary_table.md / summary.json across evaluated baselines & v4 runs
  2. collision_modes_breakdown.png (Head-on / Cross-traffic / Rear-tail / Static)
  3. noise_robustness_curve.png (Success rate vs odometry/depth noise multiplier)
  4. temporal_yielding_analysis.png (Dual-axis Speed/TTC/Risk + top-down trajectory slice)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_eval(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_collision_modes(entries: list[tuple[str, Path]], out_path: Path) -> list[dict]:
    modes = ("head_on", "cross_traffic", "rear_tail", "static")
    labels_cn = {
        "head_on": "Head-on",
        "cross_traffic": "Cross-traffic",
        "rear_tail": "Rear / Tail",
        "static": "Static",
    }
    colors = {
        "head_on": "#d62728",
        "cross_traffic": "#ff7f0e",
        "rear_tail": "#9467bd",
        "static": "#7f7f7f",
    }

    rows = []
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    x = list(range(len(entries)))
    bottoms = [0.0] * len(entries)
    names = []
    for _idx, (name, path) in enumerate(entries):
        data = load_eval(path)
        s = data["summary"]
        cm = s.get("collision_modes", {})
        if not cm:
            cm = {
                "head_on": 0,
                "cross_traffic": s.get("dynamic_collision_count", 0),
                "rear_tail": 0,
                "static": s.get("static_collision_count", 0),
            }
        names.append(f"{name}\n({s['success_rate']*100:.1f}%)")
        rows.append({"name": name, "success_rate": s["success_rate"], **cm, "total_episodes": s["episodes"]})

    for m in modes:
        vals = [r.get(m, 0) for r in rows]
        ax.bar(x, vals, bottom=bottoms, label=labels_cn[m], color=colors[m], width=0.55)
        bottoms = [b + v for b, v in zip(bottoms, vals, strict=True)]

    for idx, total in enumerate(bottoms):
        ax.text(idx, total + 1.0, f"{int(total)}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Collision Count (256 episodes)")
    ax.set_title("Collision Failure Mode Decomposition")
    ax.legend(frameon=False, ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return rows


def plot_noise_curve(series: dict[str, list[tuple[float, Path]]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    for name, pts in series.items():
        pts_sorted = sorted(pts, key=lambda item: item[0])
        xs = [p[0] for p in pts_sorted]
        ys = [load_eval(p[1])["summary"]["success_rate"] * 100.0 for p in pts_sorted]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=name)
    ax.set_xlabel("Odometry & Depth Noise Scale (x SimToRealCfg)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate Degradation Under Odometry Drift")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_temporal_yielding(trace_json: Path, out_path: Path, episode_idx: int | None = None) -> dict:
    payload = json.loads(trace_json.read_text())
    episodes = payload["episodes"]
    outcomes = payload.get("outcomes", [])
    # Auto-pick the dense cross-traffic episode with highest dynamic braking modulation while succeeding.
    if episode_idx is None:
        best_score = -1.0
        episode_idx = 0
        for idx, ep in enumerate(episodes):
            if not ep:
                continue
            outcome_ok = idx >= len(outcomes) or outcomes[idx]["outcome"] == "success"
            has_risk = sum(1 for step in ep if step["r_closing"] > 0.35 and step["min_fov_ttc_s"] < 4.0)
            speed_drop = max(step["speed"] for step in ep) - min(step["speed"] for step in ep[min(20, len(ep)-1):] or ep)
            score = has_risk * speed_drop + (1000.0 if outcome_ok else 0.0)
            if score > best_score:
                best_score = score
                episode_idx = idx
    ep = episodes[episode_idx]
    ts = [s["t"] for s in ep]
    speeds = [s["speed"] for s in ep]
    r_close = [s["r_closing"] for s in ep]
    ttcs = [min(s["min_fov_ttc_s"], 10.0) for s in ep]
    xs = [s["pos"][0] for s in ep]
    ys = [s["pos"][1] for s in ep]

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12.2, 4.4), dpi=160)
    l1 = ax1.plot(ts, speeds, color="#1f77b4", linewidth=2.0, label="Drone Speed ||v|| (m/s)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Speed (m/s)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    l2 = ax2.plot(ts, r_close, color="#d62728", linestyle="--", linewidth=1.7, label="Forward Closing Risk R")
    l3 = ax2.plot(ts, ttcs, color="#2ca02c", linestyle=":", linewidth=1.8, label="Min FOV TTC (s, cap 10)")
    ax2.set_ylabel("Closing Risk / TTC (s)", color="#d62728")
    lines = l1 + l2 + l3
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=8, frameon=False)
    ax1.set_title(f"Temporal Yielding Timeseries (Ep {episode_idx})")

    # Top-down trajectory colored by speed + dynamic obstacle snapshot around peak risk step.
    sc = ax3.scatter(xs, ys, c=speeds, cmap="viridis", s=10, label="Drone Trajectory")
    peak_i = max(range(len(ep)), key=lambda k: ep[k]["r_closing"])
    for ox, oy in ep[peak_i].get("nearby_obs_xy", []):
        ax3.scatter([ox], [oy], c="#d62728", marker="o", s=55, alpha=0.6)
    ax3.scatter([xs[peak_i]], [ys[peak_i]], c="#ff7f0e", marker="*", s=130, zorder=5, label=f"Peak Risk t={ep[peak_i]['t']:.1f}s")
    ax3.set_xlabel("World X (m)")
    ax3.set_ylabel("World Y (m)")
    ax3.set_title("Top-Down Trajectory & Dynamic Traffic Snapshot")
    ax3.axis("equal")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best", fontsize=8, frameon=False)
    fig.colorbar(sc, ax=ax3, label="Speed (m/s)")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return {"episode_idx": episode_idx, "steps": len(ep), "peak_risk_t": ep[peak_i]["t"], "peak_risk": ep[peak_i]["r_closing"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--eval", action="append", default=[], help="name=path/to/eval.json")
    parser.add_argument("--noise", action="append", default=[], help="name:scale=path/to/eval.json")
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--episode_idx", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    if args.eval:
        entries = []
        for item in args.eval:
            name, _, path_str = item.partition("=")
            entries.append((name, Path(path_str)))
        rows = plot_collision_modes(entries, args.out_dir / "collision_modes_breakdown.png")
        summary["ablation_rows"] = rows
        md_lines = [
            "| Method | Success Rate | Head-on | Cross-traffic | Rear/Tail | Static |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in rows:
            md_lines.append(
                f"| {r['name']} | {r['success_rate']*100:.2f}% | {r['head_on']} | {r['cross_traffic']} | {r['rear_tail']} | {r['static']} |"
            )
        (args.out_dir / "summary_table.md").write_text("\n".join(md_lines) + "\n")

    if args.noise:
        series: dict[str, list[tuple[float, Path]]] = {}
        for item in args.noise:
            left, _, path_str = item.partition("=")
            name, _, scale_str = left.partition(":")
            series.setdefault(name, []).append((float(scale_str), Path(path_str)))
        plot_noise_curve(series, args.out_dir / "noise_robustness_curve.png")

    if args.trace is not None and args.trace.is_file():
        summary["yielding_episode"] = plot_temporal_yielding(
            args.trace, args.out_dir / "temporal_yielding_analysis.png", episode_idx=args.episode_idx
        )

    (args.out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
