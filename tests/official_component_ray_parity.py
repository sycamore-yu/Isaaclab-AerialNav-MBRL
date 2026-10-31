"""Compare the actual old/new ray methods at the training batch shape.

This isolates ray arithmetic from renderer geometry and training nondeterminism.
The baseline camera's cached pixel constants are reconstructed as in its initializer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from types import SimpleNamespace as Namespace

import torch
from isaaclab.sensors.ray_caster import patterns

from isaaclab_navrl.tasks.navigation.sensor_mixins import _BatchedDepthCameraMixin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from pathlib import Path

    path = Path(args.baseline) / "src/isaaclab_navrl/tasks/navigation/sensor_mixins.py"
    spec = importlib.util.spec_from_file_location("isaaclab_navrl.tasks.navigation._baseline_sensor_mixins", path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    device = "cuda:0"
    cfg = patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
        [554.254691191187 / 4, 0, 320.5 / 4, 0, 554.254691191187 / 4, 240.5 / 4, 0, 0, 1],
        width=160,
        height=120,
    )
    grid = torch.meshgrid(
        torch.arange(cfg.width, dtype=torch.int32, device=device),
        torch.arange(cfg.height, dtype=torch.int32, device=device),
        indexing="xy",
    )
    pixels = torch.vstack(list(map(torch.ravel, grid))).T
    pixels = torch.hstack([pixels, torch.ones((len(pixels), 1), device=device)])
    pixels += torch.tensor([[0.5, 0.5, 0]], device=device)
    results = []
    for count, batch in ((4, 3), (32, 64), (65, 64)):
        calibration = torch.zeros(count, 3, 3, device=device)
        fx = cfg.width * cfg.focal_length / cfg.horizontal_aperture
        fy = cfg.height * cfg.focal_length / cfg.vertical_aperture
        calibration[:, 0, 0], calibration[:, 1, 1] = fx, fy
        calibration[:, 0, 2] = cfg.horizontal_aperture_offset * fx + cfg.width / 2
        calibration[:, 1, 2] = cfg.vertical_aperture_offset * fy + cfg.height / 2
        calibration[:, 2, 2] = 1
        camera = Namespace(
            _device=device,
            cfg=Namespace(pattern_cfg=cfg),
            _data=Namespace(intrinsic_matrices=Namespace(torch=calibration)),
            _ray_inverse_intrinsics=torch.inverse(calibration),
            _ray_pixels=pixels.T,
            _ray_axis_sign=torch.tensor([1, -1, -1], device=device).unsqueeze(0).unsqueeze(2),
        )
        for begin in range(0, count, batch):
            end = min(count, begin + batch)
            reference = old._BatchedDepthCameraMixin._compute_local_rays(camera, begin, end)
            actual = _BatchedDepthCameraMixin._compute_local_rays(camera, begin, end)
            maximum = max(float((a - b).abs().max()) for a, b in zip(reference, actual, strict=True))
            equal = all(torch.equal(a, b) for a, b in zip(reference, actual, strict=True))
            results.append(
                {"num_envs": count, "begin": begin, "end": end, "max_abs_error": maximum, "bitwise_equal": equal}
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))
    assert all(item["bitwise_equal"] for item in results)


if __name__ == "__main__":
    main()
