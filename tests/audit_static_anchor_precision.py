"""Independent float64 check of recorded real-mesh anchor precision."""

import argparse
import json
from pathlib import Path

import numpy as np


def closest64(point, triangle):
    p = np.asarray(point, dtype=np.float64)
    a, b, c = np.asarray(triangle, dtype=np.float64)
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    normal_squared = np.dot(normal, normal)
    candidates = []
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        length_squared = np.dot(edge, edge)
        fraction = np.clip(np.dot(p - start, edge) / length_squared, 0, 1) if length_squared else 0.0
        candidates.append(start + fraction * edge)
    if normal_squared > 0:
        projected = p - normal * (np.dot(p - a, normal) / normal_squared)
        i, j = [index for index in range(3) if index != int(np.argmax(np.abs(normal)))]
        relative = projected - a
        determinant = ab[i] * ac[j] - ab[j] * ac[i]
        beta = (relative[i] * ac[j] - relative[j] * ac[i]) / determinant
        gamma = (ab[i] * relative[j] - ab[j] * relative[i]) / determinant
        if beta >= 0 and gamma >= 0 and beta + gamma <= 1:
            candidates.append(projected)
    return min(candidates, key=lambda candidate: float(np.dot(candidate - p, candidate - p)))


def nearest64(point, triangles):
    p = np.asarray(point, dtype=np.float64)
    candidates = [closest64(p, triangle) for triangle in triangles]
    return min(candidates, key=lambda candidate: float(np.dot(candidate - p, candidate - p)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        parser.error("Output must be new and must not overwrite the recorded probes.")
    data = json.loads(args.input.read_text())
    examples = data["exit_checks"]["largest_anchor_residual_examples"]
    results = []
    for example in examples:
        assert not example["nearby_triangles_truncated"]
        triangles = [row["vertices_m"] for row in example["nearby_triangles"]]
        point = np.asarray(example["query_position_m"], dtype=np.float64)
        anchor = np.asarray(example["selected_anchor_m"], dtype=np.float64)
        requery = np.asarray(example["anchor_requery_closest_m"], dtype=np.float64)
        closest = nearest64(point, triangles)
        distance = np.linalg.norm(closest - point)
        direction = (closest - point) / distance
        old_direction = np.asarray(example["escape_direction"], dtype=np.float64)
        cosine = np.dot(direction, old_direction) / np.linalg.norm(old_direction)
        results.append(
            {
                "query_index": example["origin_query_index"],
                "anchor_actual_mesh_residual_m": float(np.linalg.norm(anchor - nearest64(anchor, triangles))),
                "requery_actual_mesh_residual_m": float(np.linalg.norm(requery - nearest64(requery, triangles))),
                "true_closest_point_m": closest.tolist(),
                "true_distance_m": float(distance),
                "distance_relative_inflation": abs(example["signed_proxy_distance_m"]) / float(distance) - 1,
                "anchor_tangent_error_m": (anchor - closest).tolist(),
                "true_escape_direction": direction.tolist(),
                "direction_angle_error_deg": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
                "probe": example,
            }
        )
    result = {
        "method": "NumPy float64 plane projection, dominant-axis barycentrics and edge projections",
        "source": str(args.input.resolve()),
        "training_acceptance": False,
        "scope": "The exported nearby actual triangles; this script does not prove global BVH face selection.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(results),
                "max_anchor_residual_m": max(row["anchor_actual_mesh_residual_m"] for row in results),
                "max_direction_error_deg": max(row["direction_angle_error_deg"] for row in results),
            }
        )
    )


if __name__ == "__main__":
    main()
