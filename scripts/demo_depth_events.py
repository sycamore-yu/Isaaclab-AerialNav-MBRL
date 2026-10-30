"""CPU demo: measured depth + a binary motion channel, using native M-detector.

No scene/object state is passed to either detector. Render labels are retained
only for scoring. Native library and parameter provenance are recorded.
"""

import argparse
import hashlib
import importlib.util
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from isaaclab_navrl.tasks.navigation.depth_history import (
    depth_event_observation,
    depth_motion_events,
    reproject_depth,
)

WIDTH, HEIGHT, DT = 160, 120, 0.05
K = np.array([[138.56367, 0, 80.125], [0, 138.56367, 60.125], [0, 0, 1.0]])
OPTICAL_TO_BODY = np.array([[0.0, 0, 1], [-1.0, 0, 0], [0, -1.0, 0]])


def render_frame(t, scenario):
    """Analytic pinhole sensor: walls, ground and two vertical cylinders."""
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    rays_optical = np.stack((x + 0.5, y + 0.5, np.ones_like(x)), axis=-1) @ np.linalg.inv(K).T
    yaw = 0.08 * np.sin(0.6 * t)
    c, s = np.cos(yaw), np.sin(yaw)
    body_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    position = np.array([0.18 * t, 0.3 * np.sin(t), 1.5])
    optical_rotation = body_rotation @ OPTICAL_TO_BODY
    directions = rays_optical @ optical_rotation.T
    depths = [np.full((HEIGHT, WIDTH), 24.0)]
    labels = [False]
    for axis, plane in ((0, 12), (1, -6), (1, 6), (2, 0)):
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = (plane - position[axis]) / directions[..., axis]
        depths.append(np.where(distance > 0, distance, np.inf))
        labels.append(False)
    active_time = max(t - 1.0, 0.0)
    moving = scenario != "ego_only" and t > 1.0
    dynamic_center = {
        "ego_only": (6, 0),
        "crossing": (6, 2.0 - active_time),
        "approaching": (8 - active_time, 0),
        "receding": (3 + 0.8 * active_time, 0),
    }[scenario]
    for (cx, cy), radius, is_moving in (((4, -2), 0.45, False), (dynamic_center, 0.5, moving)):
        offset = position[:2] - [cx, cy]
        a = (directions[..., :2] ** 2).sum(-1)
        b = 2 * (directions[..., :2] * offset).sum(-1)
        discriminant = b * b - 4 * a * (offset @ offset - radius * radius)
        distance = (-b - np.sqrt(discriminant.clip(0))) / (2 * a)
        z = position[2] + directions[..., 2] * distance
        depths.append(np.where((discriminant >= 0) & (distance > 0) & (z >= 0) & (z <= 5), distance, np.inf))
        labels.append(is_moving)
    stack = np.stack(depths)
    owner = stack.argmin(0)
    depth = stack.min(0)
    truth = np.asarray(labels)[owner]
    return depth, optical_rotation, body_rotation, position, truth, rays_optical


def panel(depth, mask, title):
    gray = (255 * (1 - depth.clip(0, 12) / 12)).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    if mask is not None:
        rgb[mask] = [240, 55, 45]
    image = Image.fromarray(rgb).resize((320, 240), Image.Resampling.NEAREST)
    output = Image.new("RGB", (320, 266), "white")
    output.paste(image, (0, 26))
    ImageDraw.Draw(output).text((6, 6), title, fill="black")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-module", type=Path, required=True)
    parser.add_argument("--native-config", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("_m_detector", args.native_module.resolve())
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    parameters = yaml.safe_load(args.native_config.read_text())["dyn_obj"]
    parameters.update(
        points_num_perframe=WIDTH * HEIGHT,
        buffer_size=WIDTH * HEIGHT * 6,
        frame_dur=DT,
        buffer_delay=DT,
        depth_map_dur=0.1,
        max_depth_map_num=5,
        debug_publish=False,
        fov_up=89.0,
        fov_down=-89.0,
        fov_cut=-89.0,
        fov_left=180.0,
        fov_right=-180.0,
    )
    summary = {
        "kind": "synthetic sensor demo, no training or RL success claim",
        "detector_inputs": ["visible depth points", "camera calibration", "ego pose estimate", "frame timestamp"],
        "pose_assumption": "noise-free onboard pose estimate in this demo; no obstacle state inputs",
        "camera": {"width": WIDTH, "height": HEIGHT, "dt_s": DT},
        "native_module": str(args.native_module.resolve()),
        "native_sha256": hashlib.sha256(args.native_module.read_bytes()).hexdigest(),
        "config_path": str(args.native_config.resolve()),
        "config_sha256": hashlib.sha256(args.native_config.read_bytes()).hexdigest(),
        "parameters": parameters,
        "scenarios": {},
    }
    for scenario in ("ego_only", "crossing", "approaching", "receding"):
        detector = native.Detector(parameters)
        history = deque(maxlen=4)
        images, trace, selected_inputs = [], [], []
        scores = {name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for name in ("naive", "compensated", "event", "moving")}
        for frame in range(100):
            t = frame * DT
            depth, rotation, body_rotation, position, truth, rays = render_frame(t, scenario)
            measured = torch.tensor(depth, dtype=torch.float32)[None, None]
            history.append(
                (
                    measured,
                    torch.tensor(rotation, dtype=torch.float32)[None],
                    torch.tensor(position, dtype=torch.float32)[None],
                )
            )
            old, old_rotation, old_position = history[0]
            before = time.perf_counter()
            warped, known = reproject_depth(
                old,
                torch.tensor(K, dtype=torch.float32)[None],
                old_rotation,
                old_position,
                history[-1][1],
                history[-1][2],
            )
            compensated = depth_motion_events(measured, warped, known)[0, 0].numpy()
            cheap_s = time.perf_counter() - before
            naive = (measured - old).abs()[0, 0].numpy() > (0.08 + 0.01 * depth)
            visible = depth < 24
            body_points = (rays * depth[..., None]) @ OPTICAL_TO_BODY.T
            before = time.perf_counter()
            detected = detector.process(np.ascontiguousarray(body_points[visible]), body_rotation, position, t + DT)
            native_s = time.perf_counter() - before
            event, moving = np.zeros_like(truth), np.zeros_like(truth)
            event[visible] = np.asarray(detected["event_labels"]) == 1
            moving[visible] = np.asarray(detected["frame_labels"]) == 1
            masks = {"naive": naive, "compensated": compensated, "event": event, "moving": moving}
            actor = depth_event_observation(measured, torch.from_numpy(moving)[None, None])
            assert actor.shape == (1, 2, 12, 16) and torch.isfinite(actor).all()
            assert set(actor[:, 1].unique().tolist()) <= {0, 1}
            if frame >= 20:
                for name, mask in masks.items():
                    scores[name]["tp"] += int((mask & truth & visible).sum())
                    scores[name]["fp"] += int((mask & ~truth & visible).sum())
                    scores[name]["fn"] += int((~mask & truth & visible).sum())
                    scores[name]["tn"] += int((~mask & ~truth & visible).sum())
            trace.append(
                {
                    "time_s": t,
                    "native_s": native_s,
                    "cheap_s": cheap_s,
                    "known_fraction": float(known.float().mean()),
                    "moving_pixels": int(moving.sum()),
                }
            )
            canvas = Image.new("RGB", (960, 554), "#edf1f4")
            ImageDraw.Draw(canvas).text(
                (8, 5),
                f"{scenario} | t={t:.2f}s | red=event/moving | pose-compensated, no obstacle oracle",
                fill="black",
            )
            tiles = [
                panel(depth, None, "Received depth"),
                panel(depth, naive, "Raw depth change (ego-motion included)"),
                panel(depth, compensated, "Cheap ego-compensated event"),
                panel(depth, event, "M-detector point-event label"),
                panel(depth, moving, "M-detector frame-out moving label"),
                panel(
                    (3 / (actor[0, 0] + 0.6)).numpy(),
                    actor[0, 1].bool().numpy(),
                    "Policy input: 12x16 depth + binary mask",
                ),
            ]
            for index, tile in enumerate(tiles):
                canvas.paste(tile, ((index % 3) * 320, 22 + (index // 3) * 266))
            images.append(canvas)
            if frame in (0, 40, 70, 99):
                selected_inputs.append(actor.numpy())
            if frame == 70:
                canvas.save(args.output / f"{scenario}.png")
        images[0].save(args.output / f"{scenario}.gif", save_all=True, append_images=images[1:], duration=50, loop=0)
        np.savez_compressed(args.output / f"{scenario}_inputs.npz", depth_event=np.concatenate(selected_inputs))
        for score in scores.values():
            score["precision"] = score["tp"] / max(score["tp"] + score["fp"], 1)
            score["recall"] = score["tp"] / max(score["tp"] + score["fn"], 1)
            score["false_positive_rate"] = score["fp"] / max(score["fp"] + score["tn"], 1)
        summary["scenarios"][scenario] = {
            "scores_after_1s_warmup": scores,
            "native_median_ms": float(np.median([r["native_s"] for r in trace[20:]]) * 1000),
            "cheap_median_ms": float(np.median([r["cheap_s"] for r in trace[20:]]) * 1000),
            "trace": trace,
        }
        (args.output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(
            scenario,
            {
                name: {k: round(v, 4) for k, v in score.items() if k in ("precision", "recall", "false_positive_rate")}
                for name, score in scores.items()
            },
            flush=True,
        )
        del detector
    pages = "".join(
        f'<h2>{name}</h2><img src="{name}.gif" width="960"><p><a href="{name}_inputs.npz">Two-channel policy tensors</a></p>'
        for name in summary["scenarios"]
    )
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Depth motion event demo</title><style>body{font:16px system-ui;max-width:1000px;margin:32px auto;background:#edf1f4}img{max-width:100%;height:auto}</style><h1>Depth + binary motion events</h1><p>Native M-detector, synthetic 20 Hz camera. A zero bit means no confirmed motion, not proof of a static point. Ground truth is used only for metrics. This is not an RL benchmark.</p><p><a href="metrics.json">Inputs, parameters, timing and detection metrics</a></p>'
        + pages
    )


if __name__ == "__main__":
    main()
