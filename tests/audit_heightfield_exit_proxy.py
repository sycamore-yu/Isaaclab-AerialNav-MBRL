"""Read-only NumPy reconstruction from the exact installed converter and project sampler."""

import argparse
import ast
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLER = ROOT / 'src/isaaclab_navrl/terrain.py'


def load_functions(path, names):
    source = ast.parse(path.read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes],
                        type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {'np': np}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, choices=(300, 400), default=300)
    parser.add_argument('--converter', type=Path, required=True,
                        help='Path to the exact installed Isaac Lab height_field/utils.py; read only.')
    parser.add_argument('--terrain_seed', type=int, default=0)
    parser.add_argument('--save_mesh', action='store_true',
                        help='Also save a potentially large NPZ beside the JSON; keep it out of git.')
    parser.add_argument('--offset', type=float, default=20.0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    functions = load_functions(SAMPLER, ('obstacle_height_field', 'discrete_obstacles_terrain'))
    convert = load_functions(args.converter, ('convert_height_field_to_mesh',))['convert_height_field_to_mesh']
    # Exact installed wrapper: 40 m / 0.1 + 1 = 401, even border_width=0
    # reserves one border pixel, so the sampler receives size 399*0.1.
    cfg = SimpleNamespace(size=(399 * 0.1, 399 * 0.1), horizontal_scale=0.1, vertical_scale=0.1,
                          obstacle_width_range=(0.4, 1.1), platform_width=0.0, num_obstacles=args.count,
                          obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],
                          obstacle_height_probability=[0.1, 0.15, 0.20, 0.55], seed=args.terrain_seed)
    inner, origin = functions['discrete_obstacles_terrain'](0.0, cfg)
    assert inner.shape == (399, 399), inner.shape
    heights = np.zeros((401, 401), dtype=np.int16)
    heights[1:-1, 1:-1] = inner
    vertices, raw_faces = convert(heights, 0.1, 0.1, 0.75)
    # Converter returns float32; trimesh stores doubles before the generator's
    # centering transform. USD/world query vertices are float32 again.
    vertices = vertices.astype(np.float64)
    vertices[:, :2] -= args.offset
    vertices = vertices.astype(np.float32)
    raw_triangles = vertices[raw_faces]
    raw_normals = np.cross(raw_triangles[:, 1] - raw_triangles[:, 0], raw_triangles[:, 2] - raw_triangles[:, 0])
    raw_length = np.linalg.norm(raw_normals, axis=-1)
    good = raw_length > 1e-12
    raw_ids = np.flatnonzero(good)
    triangles = raw_triangles[good]
    normals = raw_normals[good] / raw_length[good, None]
    centers = triangles.mean(axis=1)
    projected = np.abs(normals[:, 2]) > 1e-8
    projected_ids = np.flatnonzero(projected)
    projected_triangles = triangles[projected].astype(np.float64)
    eps = max(1e-4, float(np.abs(vertices).max()) * 8 * np.finfo(np.float32).eps)
    above_ids = np.flatnonzero(centers[:, 2] > 0.01 + 2 * eps)
    print(json.dumps({'phase': 'generated', 'count': args.count, 'faces_above': len(above_ids),
                      'nondegenerate': len(triangles), 'projected': len(projected_ids),
                      'negative_nz_above': int((normals[above_ids, 2] < -1e-8).sum()),
                      'origin_unpadded': origin.tolist(), 'seconds': time.perf_counter() - started}), flush=True)

    bins = defaultdict(list)
    cell = 0.1
    lo = np.floor(projected_triangles[:, :, :2].min(axis=1) / cell).astype(int)
    hi = np.floor(projected_triangles[:, :, :2].max(axis=1) / cell).astype(int)
    for index in range(len(projected_triangles)):
        for ix in range(lo[index, 0], hi[index, 0] + 1):
            for iy in range(lo[index, 1], hi[index, 1] + 1):
                bins[ix, iy].append(index)
    bins = {key: np.asarray(indices, dtype=np.int32) for key, indices in bins.items()}
    ab = projected_triangles[:, 1, :2] - projected_triangles[:, 0, :2]
    ac = projected_triangles[:, 2, :2] - projected_triangles[:, 0, :2]
    denom = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]

    def intersections(point):
        ids = bins.get((math.floor(float(point[0]) / cell), math.floor(float(point[1]) / cell)))
        if ids is None:
            return []
        relative = point[:2].astype(np.float64) - projected_triangles[ids, 0, :2]
        u = (relative[:, 0] * ac[ids, 1] - relative[:, 1] * ac[ids, 0]) / denom[ids]
        v = (ab[ids, 0] * relative[:, 1] - ab[ids, 1] * relative[:, 0]) / denom[ids]
        valid = (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10)
        ids, u, v = ids[valid], u[valid], v[valid]
        z = (projected_triangles[ids, 0, 2] * (1-u-v) + projected_triangles[ids, 1, 2] * u
             + projected_triangles[ids, 2, 2] * v)
        entries = [(float(zi), int(raw_ids[projected_ids[index]]), 1 if denom[index] > 0 else -1,
                    float(ui), float(vi)) for zi, index, ui, vi in zip(z, ids, u, v, strict=True)]
        return sorted(entries, reverse=True)

    def classify(point, hits):
        highest_inside = bool(hits and float(point[2]) < hits[0][0])
        above = [entry for entry in hits if entry[0] > float(point[2])]
        # This raw sum is intentionally NOT called an exact winding number:
        # boundary duplicate hits are visible in the detailed hit records.
        raw_crossings = sum(entry[2] for entry in above)
        groups = []
        for entry in above:
            if groups and abs(entry[0] - groups[-1][0]) < 1e-7:
                groups[-1][1].add(entry[2])
            else:
                groups.append([entry[0], {entry[2]}])
        grouped_crossings = sum(sum(signs) for _, signs in groups)
        return highest_inside, raw_crossings, grouped_crossings

    counts = defaultdict(int)
    anomalous = []
    negative_examples = []
    for j, index in enumerate(above_ids):
        center, normal = centers[index], normals[index]
        plus = (center + np.float32(eps) * normal).astype(np.float32)
        minus = (center - np.float32(eps) * normal).astype(np.float32)
        hp, hm = intersections(plus), intersections(minus)
        cp, cm = classify(plus, hp), classify(minus, hm)
        both_inside = cp[0] and cm[0]
        both_outside = not cp[0] and not cm[0]
        counts['both_inside'] += int(both_inside)
        counts['both_outside'] += int(both_outside)
        counts[f'raw_crossing_pair_{cp[1]}_{cm[1]}'] += 1
        counts[f'grouped_crossing_pair_{cp[2]}_{cm[2]}'] += 1
        if both_inside or both_outside:
            item = {'nondegenerate_face': int(index), 'raw_face': int(raw_ids[index]),
                    'vertices': triangles[index].tolist(), 'center': center.tolist(), 'normal': normal.tolist(),
                    'both_inside': bool(both_inside), 'both_outside': bool(both_outside),
                    'plus': plus.tolist(), 'minus': minus.tolist(),
                    'plus_highest_raw_grouped': cp, 'minus_highest_raw_grouped': cm,
                    'plus_all_hits_z_raw_face_sign_uv': hp, 'minus_all_hits_z_raw_face_sign_uv': hm}
            anomalous.append(item)
        if normal[2] < -1e-8 and len(negative_examples) < 10:
            negative_examples.append({'raw_face': int(raw_ids[index]), 'vertices': triangles[index].tolist(),
                                      'normal': normal.tolist(), 'plus': cp, 'minus': cm})
        if j and j % 10000 == 0:
            print(json.dumps({'phase': 'probed', 'done': j, 'seconds': time.perf_counter()-started}), flush=True)
    result = {
        'source_converter_path': str(args.converter.resolve()), 'source_sampler_path': str(SAMPLER.relative_to(ROOT)),
        'converter_sha256': hashlib.sha256(args.converter.read_bytes()).hexdigest(),
        'sampler_sha256': hashlib.sha256(SAMPLER.read_bytes()).hexdigest(),
        'scope': 'Local exact-function top-mesh reconstruction; requires world-frame/face matching to live PhysX export.',
        'height_samples_sha256': hashlib.sha256(heights.tobytes()).hexdigest(),
        'num_obstacles': args.count, 'seed': args.terrain_seed, 'origin_offset_xy': args.offset,
        'above_ground_faces': len(above_ids), 'nondegenerate_faces': len(triangles),
        'probe_offset': eps, 'counts': dict(counts), 'anomalous_faces': anomalous,
        'negative_normal_examples': negative_examples, 'elapsed_s': time.perf_counter()-started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    if args.save_mesh:
        np.savez_compressed(args.output.with_suffix('.npz'), vertices=vertices, faces=raw_faces, heights=heights)
    print(json.dumps({'phase': 'complete', 'path': str(args.output.with_suffix('.json')),
                      'above_ground_faces': len(above_ids), 'counts': dict(counts),
                      'seconds': time.perf_counter()-started}), flush=True)


if __name__ == '__main__':
    main()
