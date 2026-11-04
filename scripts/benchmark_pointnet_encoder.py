"""Compare lossless PointNet execution on captured full-resolution LiDAR scans."""

import argparse
import gc
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from aerialnav.pointnet.config import PointNetTrainCfg
from aerialnav.pointnet.model import PointNetPolicy


class DenseReference(PointNetPolicy):
    checkpoint_batch_size = 16

    def forward(self, points, state, hidden=None):
        valid = points[:, -1:] > 0
        features = self.point_encoder(torch.where(valid, points[:, : self.channels], 0))
        pooled = features.masked_fill(~valid, torch.finfo(features.dtype).min).amax(-1)
        pooled = torch.where(valid.any(-1), pooled, 0)
        return self._recurrent(pooled, state, hidden)

    def forward_checkpointed(self, points, state, hidden):
        values = [
            checkpoint(self, points[i : i + 16], state[i : i + 16], hidden[i : i + 16], use_reentrant=False)
            for i in range(0, points.shape[0], 16)
        ]
        return tuple(torch.cat(part) for part in zip(*values, strict=True))


CASES = {
    "dense_fp32_16": None,
    "packed_fp32_16": (16, "fp32", "torch"),
    "packed_fp32_32": (32, "fp32", "torch"),
    "packed_fp32_64": (64, "fp32", "torch"),
    "packed_bf16_32": (32, "bf16", "torch"),
    "packed_bf16_64": (64, "bf16", "torch"),
    "compile_fp32_64": (64, "fp32", "compile"),
    "compile_bf16_64": (64, "bf16", "compile"),
    "triton_fp32_64": (64, "fp32", "triton"),
    "triton_bf16_64": (64, "bf16", "triton"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = [{k: v.cuda() for k, v in frame.items()} for frame in torch.load(args.points, weights_only=True)]
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "shared_gpu": True,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "frames": len(frames),
        "num_envs": len(frames[0]["points"]),
        "timesteps": args.timesteps,
        "valid_fractions": [float((f["points"][:, -1] > 0).float().mean()) for f in frames],
        "points_path": str(args.points.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "gates": {
            "fp32_output_relative_l2": 1e-4,
            "fp32_parameter_gradient_relative_l2": 0.005,
            "bf16_output_relative_l2": 0.02,
            "bf16_parameter_gradient_relative_l2": 0.1,
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "candidate.patch").write_text(subprocess.check_output(["git", "diff", "HEAD"], text=True))
    baseline_path = args.output_dir / "baseline.pt"
    for name in args.cases:
        model = DenseReference().cuda() if CASES[name] is None else PointNetPolicy().cuda()
        model.load_state_dict(payload["model_state_dict"])
        if CASES[name] is not None:
            chunk, precision, backend = CASES[name]
            model.configure_execution(
                PointNetTrainCfg(point_microbatch_envs=chunk, point_precision=precision, point_encoder_backend=backend)
            )
        measurements = []
        result = {"case": name}
        try:
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(args.repeats + 1):
                model.zero_grad(set_to_none=True)
                hidden = torch.zeros(len(frames[0]["points"]), 192, device="cuda")
                loss = 0
                torch.cuda.synchronize()
                started = time.perf_counter()
                for step in range(args.timesteps):
                    frame = frames[step % len(frames)]
                    output, hidden = model.forward_checkpointed(frame["points"], frame["state"], hidden)
                    loss = loss + output.square().mean() + 0.01 * hidden.square().mean()
                assert output.dtype == hidden.dtype == loss.dtype == torch.float32
                loss.backward()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if repeat:
                    measurements.append(elapsed)
                else:
                    result["cold_s"] = elapsed
            gradients = torch.cat([p.grad.flatten() for p in model.parameters()]).detach().cpu()
            output = output.detach().cpu()
            if name == "dense_fp32_16":
                torch.save({"output": output, "gradients": gradients}, baseline_path)
            reference = torch.load(baseline_path, weights_only=True)
            output_error = float((output - reference["output"]).norm() / reference["output"].norm().clamp_min(1e-12))
            gradient_error = float(
                (gradients - reference["gradients"]).norm() / reference["gradients"].norm().clamp_min(1e-12)
            )
            approximate = CASES[name] is not None and CASES[name][1] == "bf16"
            passed = output_error <= (0.02 if approximate else 1e-4) and gradient_error <= (
                0.1 if approximate else 0.005
            )
            result.update(
                seconds=measurements,
                median_s=statistics.median(measurements),
                peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                output_relative_l2=output_error,
                gradient_relative_l2=gradient_error,
                numerical_gate_passed=passed,
            )
        except Exception as exc:
            result.update(error=f"{type(exc).__name__}: {exc}")
        (args.output_dir / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
