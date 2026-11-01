"""Same-state historical DCOL/SDF diagnostics; never used by the trainer."""

import time

import torch


def compare_dynamic_metrics(query, positions, sizes, radius=0.3):
    """Same frozen poses and points; no terminal, policy or loss-weight changes."""
    from aerialnav.tasks.navigation.geometry import dynamic_signed_distance

    from .dcol import dynamic_min_alpha

    points = query.detach().reshape(-1, 3).requires_grad_(True)
    poses = positions.detach().clone()
    result = {"points": len(points), "obstacles": len(poses), "radius_m": radius, "metrics": {}}
    if not len(poses):
        return result
    gradients = []
    for name, measure in (
        ("dcol", lambda p: radius * (dynamic_min_alpha(p, radius, poses, sizes) - 1)),
        ("sdf", lambda p: dynamic_signed_distance(p, poses, sizes) - radius),
    ):
        if points.device.type == "cuda":
            torch.cuda.synchronize(points.device)
        started = time.perf_counter()
        value = measure(points)
        gradient = torch.autograd.grad(value.sum(), points)[0]
        if points.device.type == "cuda":
            torch.cuda.synchronize(points.device)
        measured_s = time.perf_counter() - started
        gradients.append(gradient)
        numerical = []
        with torch.no_grad():
            for axis in range(3):
                delta = torch.zeros_like(points)
                delta[:, axis] = 1e-3
                numerical.append((measure(points + delta) - measure(points - delta)) / 2e-3)
        numerical = torch.stack(numerical, dim=-1)
        norms = gradient.norm(dim=-1)
        result["metrics"][name] = {
            "gap_mean_m": float(value.detach().mean()),
            "gradient_norm_mean": float(norms.mean()),
            "single_forward_backward_s": measured_s,
            "fd_error_max": float((gradient - numerical).abs().max()),
            "fd_error_p95": float((gradient - numerical).abs().flatten().quantile(0.95)),
        }
    cosine = torch.nn.functional.cosine_similarity(*gradients, dim=-1)
    result["gradient_cosine_mean"] = float(cosine.mean())
    result["note"] = (
        "finite differences may cross closest-obstacle or surface ties; proxy units do not imply equal gradient scales"
    )
    return result
