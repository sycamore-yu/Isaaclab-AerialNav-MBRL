"""Verify signed static geometry on the actual Isaac Lab terrain and PhysX scene.

Manual geometry/pose probes are not learned policy successes.  A persisted
``status=passed`` JSON and marker are required; Kit process exit zero alone is
not a passing result.  Run once for each static map/count used by the restart.
"""

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def write_result(path, result, *, marker=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if marker:
        print("SIGNED_STATIC_SCENE " + json.dumps(result, allow_nan=False), flush=True)


def surface_exposure_check(terrain):
    """Retain the real mesh's folded/internal/fin audit without deleting faces.

    The first scene failed with 74 both-inside and 43 both-outside faces at
    300 obstacles. Those facts remain in the report. The corrected gate is
    whether inside queries select valid original-mesh exits despite them.
    """
    import torch
    import warp as wp

    centers = wp.to_torch(terrain._face_centers)
    normals = wp.to_torch(terrain._face_normals)
    mask = centers[:, 2] > 0.01 + 2 * terrain._normal_probe
    indices = mask.nonzero().flatten()
    centers, normals = centers[indices], normals[indices]
    plus = terrain.signed_query(centers + terrain._normal_probe * normals)
    minus = terrain.signed_query(centers - terrain._normal_probe * normals)
    both_inside = plus.inside & minus.inside
    both_outside = ~plus.inside & ~minus.inside
    record = {
        "above_ground_faces_checked": len(indices), "all_above_ground_face_centroids_checked": True,
        "probe_offset_m": terrain._normal_probe,
        "both_sides_inside_count": int(both_inside.sum()),
        "both_sides_outside_count": int(both_outside.sum()),
        "incompatible_face_indices": indices[both_inside | both_outside][:20].tolist(),
        "incompatible_face_centers_m": centers[both_inside | both_outside][:20].tolist(),
        "counts_are_retained_map_facts": True,
        "scope": "Actual face-centroid probes; folded/internal/free sheets are not removed or treated as solid exits.",
    }
    outward = torch.where(plus.inside[:, None], -normals, normals)
    return record, centers, outward, ~(both_inside | both_outside)


def verify_selected_exits(terrain, points, record, source_face_ids=None):
    """Independently re-query selected anchors and both sides of their exits."""
    import torch

    point = points.detach().clone().requires_grad_()
    details = terrain.signed_query(point)
    inside = details.inside
    invalid = inside & ~details.exit_valid
    record.update(
        query_count=len(point), inside_count=int(inside.sum()), outside_count=int((~inside).sum()),
        exit_search_count=int(details.used_exit_search.sum()), invalid_inside_exit_count=int(invalid.sum()),
        invalid_inside_positions_m=point.detach()[invalid][:20].tolist(),
        original_zero_or_near_zero_inside_count=int(
            (inside & ((point.detach() - details.mesh_closest_point).norm(dim=-1) < 2e-6)).sum()
        ),
    )
    assert not bool(invalid.any()), "Inside query returned an unverified exit."
    assert bool(torch.isfinite(details.distance).all()), "Signed proxy distance must be finite."
    assert bool(torch.isfinite(details.outward_normal).all()), "Escape direction must be finite."
    anchor = details.closest_point[inside]
    selected_length = (anchor - point.detach()[inside]).norm(dim=-1)
    direction = (anchor - point.detach()[inside]) / selected_length.clamp_min(1e-12)[:, None]
    direction = torch.where((selected_length == 0)[:, None], details.outward_normal[inside], direction)
    anchor_requery = terrain.closest_points(anchor)
    residual = (anchor - anchor_requery).norm(dim=-1)
    before = terrain.signed_query(anchor - terrain._normal_probe * direction)
    after = terrain.signed_query(anchor + terrain._normal_probe * direction)
    bad_direction = ~before.inside | after.inside
    record.update(
        selected_anchor_original_mesh_residual_max_m=float(residual.max()) if len(residual) else None,
        invalid_direction_probe_count=int(bad_direction.sum()),
        invalid_direction_anchor_examples_m=anchor[bad_direction][:20].tolist(),
        original_mesh_unchanged=True,
        finite_probe_offset_m=terrain._normal_probe,
        exact_signed_distance_claimed=False,
        mathematical_zero_epsilon_boundary_proof_claimed=False,
    )
    if bool(bad_direction.any()):
        inside_indices = inside.nonzero().flatten()
        samples = []
        for slot in bad_direction.nonzero().flatten()[:5].tolist():
            origin = int(inside_indices[slot])
            samples.append({
                "origin_query_index": origin, "query_m": point.detach()[origin].tolist(),
                "anchor_m": anchor[slot].tolist(), "gradient_direction": direction[slot].tolist(),
                "kernel_direction": details.outward_normal[origin].tolist(),
                "before_m": (anchor[slot] - terrain._normal_probe * direction[slot]).tolist(),
                "after_m": (anchor[slot] + terrain._normal_probe * direction[slot]).tolist(),
                "before_inside": bool(before.inside[slot]), "after_inside": bool(after.inside[slot]),
                "before_projected_height": float(before.surface_height[slot]) if bool(before.has_surface[slot]) else None,
                "after_projected_height": float(after.surface_height[slot]) if bool(after.has_surface[slot]) else None,
                "used_exit_search": bool(details.used_exit_search[origin]),
                "exit_valid": bool(details.exit_valid[origin]),
            })
        record["invalid_direction_details"] = samples
    if bool((residual >= 2e-5).any()):
        # Preserve the original gate. Export only the five worst points and
        # their actual nearby triangles for an independent float64 oracle;
        # a second float32 nearest query is not assumed infallible evidence.
        import numpy as np

        triangles = terrain._triangle_points.numpy().reshape(-1, 3, 3)
        lower, upper = triangles.min(axis=1), triangles.max(axis=1)
        inside_indices = inside.nonzero().flatten()
        worst = residual.argsort(descending=True)[:5]
        examples = []
        for slot in worst.tolist():
            origin = int(inside_indices[slot])
            selected = anchor[slot].detach().cpu().numpy()
            separation = np.maximum(np.maximum(lower - selected, selected - upper), 0.0)
            bbox_squared = np.square(separation.astype(np.float64)).sum(axis=1)
            nearby = np.flatnonzero(bbox_squared <= 0.002**2)
            total_nearby = len(nearby)
            if total_nearby > 128:
                nearby = nearby[np.argsort(bbox_squared[nearby])[:128]]
            examples.append({
                "origin_query_index": origin, "inside_batch_index": slot,
                "source_face_index": int(source_face_ids[origin]) if source_face_ids is not None else None,
                "query_position_m": point.detach()[origin].tolist(),
                "signed_proxy_distance_m": float(details.distance[origin]),
                "selected_anchor_m": anchor[slot].tolist(),
                "anchor_requery_closest_m": anchor_requery[slot].tolist(),
                "anchor_requery_residual_m": float(residual[slot]),
                "original_query_mesh_closest_m": details.mesh_closest_point[origin].tolist(),
                "used_exit_search": bool(details.used_exit_search[origin]),
                "exit_valid": bool(details.exit_valid[origin]),
                "selected_anchor_xy_equals_query": bool(torch.equal(anchor[slot, :2], point.detach()[origin, :2])),
                "highest_projected_height_m": float(details.surface_height[origin]),
                "escape_direction": direction[slot].tolist(),
                "nearby_triangle_aabb_radius_m": 0.002,
                "nearby_triangle_count": total_nearby,
                "nearby_triangles_truncated": total_nearby > 128,
                "nearby_triangles": [{"face_index": int(face), "vertices_m": triangles[face].tolist(),
                                      "bbox_distance_m": float(np.sqrt(bbox_squared[face]))}
                                     for face in nearby],
            })
        record["largest_anchor_residual_examples"] = examples
    assert bool((residual < 2e-5).all()), "An exit anchor must lie on the original mesh, not a virtual curtain."
    assert not bool(bad_direction.any()), "Selected direction must cross from training occupancy into free space."
    gradient = torch.autograd.grad(details.distance.sum(), point, retain_graph=True)[0]
    assert bool(torch.isfinite(gradient).all())
    alignment = (gradient[inside] * direction).sum(dim=-1)
    record["escape_gradient_alignment_min"] = float(alignment.min()) if len(alignment) else None
    assert bool((alignment > 0.9999).all())
    # Outside this explicitly defined occupancy, even a zero-thickness fin
    # keeps the full original mesh value and nonzero-distance derivative.
    unsigned = (point - details.mesh_closest_point).norm(dim=-1)
    outside = ~inside
    torch.testing.assert_close(details.distance[outside], unsigned[outside], rtol=0, atol=0)
    canonical_gradient = torch.autograd.grad(unsigned.sum(), point)[0]
    nonzero_outside = outside & (unsigned.detach() > 0)
    torch.testing.assert_close(gradient[nonzero_outside], canonical_gradient[nonzero_outside], rtol=0, atol=0)
    record["outside_reference"] = "unchanged_mesh_with_float64_selected_face_refinement"
    record.update(outside_value_parity=True, outside_nonzero_gradient_parity=True, status="passed")


def loss_probe(terrain, point, expected_outward):
    import torch
    import torch.nn.functional as functional

    from isaaclab_navrl.tasks.navigation.geometry import nearest_obstacle_signed_distance
    from isaaclab_navrl.zhang.dcol import hybrid_collision_gap

    query = point.detach().clone().reshape(1, 3).requires_grad_()
    signed = terrain.signed_query(query)
    empty = query.new_empty(0, 3)
    distance = nearest_obstacle_signed_distance(query, signed.distance, empty, empty)
    gap = hybrid_collision_gap(query, 0.3, signed.distance, empty, empty)
    avoidance = 3.0 * functional.relu(1 - (distance - 0.3)).square().mean()
    collision = 7.5 * functional.softplus(-32 * gap).mean()
    gradients = {}
    for name, loss in (("avoidance", avoidance), ("collision", collision), ("sum", avoidance + collision)):
        gradient = torch.autograd.grad(loss, query, retain_graph=True)[0][0]
        assert bool(torch.isfinite(gradient).all())
        projection = float((-gradient * expected_outward).sum())
        assert projection > 0, (name, query.tolist(), gradient.tolist(), expected_outward.tolist())
        gradients[name] = {"position_gradient": gradient.tolist(), "descent_dot_outward": projection}
    moved = query.detach() + 0.002 * expected_outward
    new_distance = terrain.signed_distance(moved)
    assert float(new_distance[0]) > float(signed.distance[0]) + 0.001
    has_surface = bool(signed.has_surface[0])
    return {
        "position_m": query.detach()[0].tolist(), "inside": bool(signed.inside[0]),
        "distance_m": float(signed.distance[0]), "closest_point_m": signed.closest_point[0].tolist(),
        "projected_height_m": float(signed.surface_height[0]) if has_surface else None,
        "has_projected_surface": has_surface, "clearance_m": float(gap[0]),
        "outward": expected_outward.tolist(), "distance_after_outward_2mm_m": float(new_distance[0]),
        "weighted_loss_gradients": gradients,
    }


def actual_mesh_checks(env, result, progress):
    import torch
    from new_mechanisms_scene import check_perception, write_pose

    terrain = env.terrain_distance
    progress("above_ground_surface_exposure")
    exposure, centers, normals, exposed = surface_exposure_check(terrain)
    result["surface_exposure"] = exposure
    result["geometry_contract"] = {
        "mode": terrain.geometry_mode, "occupancy": "conservative highest projected actual triangle",
        "projection_precision": terrain.projection_precision,
        "inside": "negative original-mesh verified-exit distance proxy",
        "outside": "original full-mesh unsigned distance, including zero-thickness fins",
        "exact_mesh_sdf": False, "physical_air_cavity_equivalence": False,
        "virtual_curtains_added": False, "physx_and_depth_mesh_changed": False,
        "local_exit_search_radius_limit_m": terrain.exit_search_radius_limit_m,
        "near_surface_refinement_radius_m": terrain.near_surface_refinement_radius_m,
        "exit_probe_offset_m": terrain._normal_probe,
        "exit_height_clearance_guard_m": terrain.exit_height_guard_m,
    }
    progress("surface_exposure_result")
    progress("selected_exit_validity")
    # Test all original above-ground face sides and the anomalous face centers
    # themselves, including actual internal crossing/zero-distance points.
    points = torch.cat((centers + terrain._normal_probe * normals,
                        centers - terrain._normal_probe * normals, centers[~exposed]))
    # This uses the same exact filter/order as surface_exposure_check.
    import warp as wp

    all_centers = wp.to_torch(terrain._face_centers)
    face_ids = (all_centers[:, 2] > 0.01 + 2 * terrain._normal_probe).nonzero().flatten()
    source_face_ids = torch.cat((face_ids, face_ids, face_ids[~exposed]))
    result["selected_exit_checks"] = {}
    result["selected_exit_checks"]["query_origin_layout"] = [
        {"kind": "face_plus", "start": 0, "count": len(centers)},
        {"kind": "face_minus", "start": len(centers), "count": len(centers)},
        {"kind": "anomalous_face_center", "start": 2 * len(centers), "count": int((~exposed).sum())},
    ]
    verify_selected_exits(terrain, points, result["selected_exit_checks"], source_face_ids)
    progress("selected_exit_result")
    # Bounded query-tail check, measured under the recorded shared GPU load.
    # These synthetic batches expose anomalous/low-z costs; they do not estimate
    # policy training throughput or claim to follow the training distribution.
    import statistics

    batches = {"anomalous_face_centers": centers[~exposed]}
    low = centers[torch.linspace(0, len(centers) - 1, 2560, device=env.device).long()].clone()
    low[:, 2] = -0.5
    batches["low_z_2560"] = low
    timings = {}
    for name, queries in batches.items():
        for _ in range(2):
            terrain.signed_query(queries)
        torch.cuda.synchronize(env.device)
        elapsed = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            measured = terrain.signed_query(queries)
            end.record()
            end.synchronize()
            elapsed.append(start.elapsed_time(end))
        assert bool(torch.isfinite(measured.distance).all())
        assert not bool((measured.inside & ~measured.exit_valid).any())
        timings[name] = {
            "query_count": len(queries), "cuda_query_ms": elapsed,
            "median_cuda_query_ms": statistics.median(elapsed),
            "inside_count": int(measured.inside.sum()),
            "exit_search_count": int(measured.used_exit_search.sum()),
            "invalid_inside_exit_count": int((measured.inside & ~measured.exit_valid).sum()),
        }
    result["bounded_query_timings"] = timings
    progress("bounded_query_timings_result")
    vertical = exposed & (normals[:, 2].abs() < 1e-5) & (centers[:, 2] > 0.8) & (centers[:, 2] < 3.0)
    top = exposed & (normals[:, 2] > 0.99) & (centers[:, 2] > 1.0)
    assert bool(vertical.any()) and bool(top.any()), "Need real vertical and top terrain triangles."

    progress("actual_side_and_top_queries")
    chosen = {}
    for label, mask in (("side", vertical), ("top", top)):
        indices = mask.nonzero().flatten()
        indices = indices[torch.linspace(0, len(indices) - 1, min(1024, len(indices)), device=env.device).long()]
        candidate, normal = centers[indices], normals[indices]
        before = terrain.signed_query(candidate - 0.02 * normal)
        after = terrain.signed_query(candidate + 0.02 * normal)
        valid = (before.inside & ~after.inside & ((before.distance + 0.02).abs() < 2e-4)
                 & ((after.distance - 0.02).abs() < 2e-4))
        assert bool(valid.any()), f"No unambiguous actual {label} surface found."
        index = int(valid.nonzero()[0, 0])
        center, outward = candidate[index], normal[index]
        chosen[label] = (center, outward)
        result[label] = {
            "surface_center_m": center.tolist(), "outward": outward.tolist(),
            "inside": loss_probe(terrain, center - 0.02 * outward, outward),
            "outside": loss_probe(terrain, center + 0.02 * outward, outward),
        }
        surface = center.detach().clone().reshape(1, 3).requires_grad_()
        distance = terrain.signed_distance(surface)
        gradient = torch.autograd.grad(distance.sum(), surface)[0][0]
        assert bool(torch.isfinite(gradient).all()) and float((gradient * outward).sum()) > 0.9
        assert float(distance.abs().max()) < 2e-5
        result[label]["surface_distance_m"] = float(distance[0])
        result[label]["surface_gradient"] = gradient.tolist()

    progress("outside_extent_and_ground")
    outside = torch.tensor([[float(terrain.bounds[1][0]) + 1, 0, 0.6],
                            [float(terrain.bounds[1][0]) + 1, 0, -0.5]], device=env.device)
    outside_query = terrain.signed_query(outside)
    assert not bool(outside_query.has_surface.any()) and not bool(outside_query.inside.any())
    from isaaclab_navrl.tasks.navigation.geometry import nearest_obstacle_signed_distance

    below = outside[1:2].detach().clone().requires_grad_()
    empty = below.new_empty(0, 3)
    ground = nearest_obstacle_signed_distance(below, terrain.signed_distance(below), empty, empty)
    gradient = torch.autograd.grad(ground.sum(), below)[0]
    torch.testing.assert_close(ground, below[:, 2] - 0.01, rtol=0, atol=0)
    torch.testing.assert_close(gradient, below.new_tensor([[0, 0, 1]]), rtol=0, atol=0)
    result["outside_extent"] = {"positions_m": outside.tolist(), "projected_hits": 0,
                                "ground_distance_m": float(ground[0]), "ground_gradient": gradient[0].tolist()}

    progress("physx_inside_side_pose")
    center, outward = chosen["side"]
    inside_position = (center - 0.02 * outward)[None]
    lane = env.num_envs - 1
    ids = torch.tensor([lane], dtype=torch.int32, device=env.device)
    env._reset_idx(ids)
    write_pose(env, ids, inside_position, 0.01 * outward[None])
    env.target_pos[lane] = center + 3 * outward
    env.target_dir[lane] = env.target_pos[lane] - inside_position[0]
    episode = int(env.episode_id[lane])
    action = torch.zeros(env.num_envs, 3, device=env.device)
    _, _, terminated, timeout, info = env.step(action)
    terminal = info["transition"]["position"][lane]
    assert not bool(terminated[lane] | timeout[lane]) and int(env.episode_id[lane]) == episode
    physics_query = terrain.signed_query(terminal[None])
    assert bool(physics_query.inside[0]), "Test point unexpectedly left the obstacle during one transition."
    result["physx_probe"] = {
        "manual_pose": True, "policy_success_claimed": False, "episode_id_unchanged": True,
        "position_before_m": inside_position[0].tolist(), "position_after_m": terminal.tolist(),
        "collision_detector": bool(info["transition"]["collision"][lane]),
        "loss_gradient": loss_probe(terrain, terminal, outward), "depth": check_perception(env),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_static_obstacles", type=int, choices=(300, 400), default=300)
    parser.add_argument("--terrain_seed", type=int, default=0)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    args = parser.parse_args()

    import gymnasium as gym

    import isaaclab_navrl.tasks  # noqa: F401
    from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import ZhangNavRLEnvCfg

    result = {"test": "signed_static_scene", "status": "running", "num_envs": 64,
              "static_obstacles": args.num_static_obstacles, "dynamic_obstacles": 0,
              "terrain_seed": args.terrain_seed, "training_acceptance": False,
              "manual_geometry_probes": True, "current_check": "configuration"}

    def progress(name):
        result["current_check"] = name
        write_result(args.output, result)
        print("SIGNED_STATIC_CHECK " + name, flush=True)

    try:
        cfg = ZhangNavRLEnvCfg()
        cfg.scene.num_envs = 64
        cfg.num_static_obstacles, cfg.num_dynamic_obstacles = args.num_static_obstacles, 0
        cfg.terrain_seed = args.terrain_seed
        cfg.training_stage, cfg.training_curriculum = "pretrain_s0", True
        cfg.variant, cfg.sim.device, cfg.sim_to_real.enabled = "continuous", args.device, False
        cfg.configure_scene()
        progress("launch_simulation")
        with launch_simulation(cfg, args):
            env = None
            try:
                progress("create_environment")
                env = gym.make("Isaac-NavRL-Zhang-Hummingbird-Direct-v0", cfg=cfg).unwrapped
                env.reset()
                assert env.motion.position.shape == env.motion.size.shape == (0, 3)
                actual_mesh_checks(env, result, progress)
                result.update(status="passed", current_check="complete")
                write_result(args.output, result, marker=True)
            except BaseException as error:
                result.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                write_result(args.output, result, marker=True)
                raise
            finally:
                if env is not None:
                    env.close()
    except BaseException as error:
        if result["status"] != "failed":
            result.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            write_result(args.output, result, marker=True)
        raise


if __name__ == "__main__":
    main()
