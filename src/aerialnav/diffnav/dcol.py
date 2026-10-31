"""Batched DCOL common-scale queries for a spherical drone.

DCOL minimizes a *shared* scale alpha of both primitives about their own
centres.  Alpha is dimensionless: alpha > 1 means separation and alpha < 1
means intersection.  ``radius * (alpha - 1)`` is a metre-scaled proxy, not an
exact signed distance.  These queries provide geometric first derivatives;
they do not differentiate PhysX contact impulses or termination times.

For an axis-aligned box, the defining equation is

    norm(relu(abs(p - c) - alpha * half_extents)) = alpha * radius.

A finite Z-axis cylinder has the same equation in (radial, vertical)
coordinates.  We solve each possible active set analytically, then use the
implicit derivative of this equation.  This is the DCOL scaling problem,
not an SDF renamed as alpha.  The exact metric is only piecewise smooth;
at coincident centres we select a deterministic one-sided escape derivative.

Reference: https://arxiv.org/html/2207.00669v3, Section III-A.
"""

from __future__ import annotations

import torch

from aerialnav.tasks.navigation.geometry import escape_abs, escape_norm_xy


def _scalar_like(value, reference):
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _radius_like(radius, shape, reference):
    if isinstance(radius, (int, float)) and radius <= 0:
        raise ValueError("The drone sphere radius must be positive")
    return torch.broadcast_to(_scalar_like(radius, reference), shape)


def _scale_from_coordinates(coordinates, extents, radius):
    """Solve a 2D/3D box-distance root, retaining only its implicit graph.

    Inputs have broadcast-compatible leading dimensions.  Coordinates and
    extents must be nonnegative, and radius must be strictly positive.
    Zero extents are supported, including a finite planar rectangle.
    This construction supplies first-order gradients, not a Hessian API.
    """
    coordinates, extents = torch.broadcast_tensors(coordinates, extents)
    radius = _radius_like(radius, coordinates.shape[:-1], coordinates)
    tiny = torch.finfo(coordinates.dtype).tiny
    with torch.no_grad():
        # An axis is active precisely while alpha < coordinate / extent.
        ratio = torch.where(
            extents > 0,
            coordinates / extents.clamp_min(tiny),
            torch.where(coordinates > 0, torch.inf, 0.0),
        )
        order = ratio.argsort(dim=-1, descending=True)
        d = coordinates.gather(-1, order)
        e = extents.gather(-1, order)
        upper = ratio.gather(-1, order)
        lower = torch.cat((upper[..., 1:], torch.zeros_like(upper[..., :1])), dim=-1)
        a = e.square().cumsum(-1) - radius[..., None].square()
        b = (d * e).cumsum(-1)
        c = d.square().cumsum(-1)
        discriminant = (b.square() - a * c).clamp_min(0)
        # Rationalized smaller positive root; also stable when a is zero.
        candidates = c / (b + discriminant.sqrt()).clamp_min(tiny)
        tolerance = 16 * torch.finfo(coordinates.dtype).eps * (1 + candidates.abs())
        valid = (candidates >= lower - tolerance) & (candidates <= upper + tolerance)
        alpha = candidates.masked_fill(~valid, torch.inf).amin(-1)
        # At an active-set tie roundoff can reject all intervals.  Select the
        # analytic candidate with the smallest original-equation residual.
        residuals = (
            (coordinates[..., None, :] - candidates[..., :, None] * extents[..., None, :]).clamp_min(0).norm(dim=-1)
            - candidates * radius[..., None]
        ).abs()
        fallback = candidates.gather(-1, residuals.argmin(-1, keepdim=True)).squeeze(-1)
        alpha = torch.where(torch.isfinite(alpha), alpha, fallback)
        alpha = torch.where(coordinates.square().sum(-1) == 0, 0.0, alpha)

    outside = (coordinates - alpha[..., None] * extents).clamp_min(0)
    distance = outside.norm(dim=-1)
    normal = outside.detach() / distance.detach().clamp_min(tiny)[..., None]
    derivative_alpha = -(normal * extents.detach()).sum(-1) - radius.detach()
    residual = distance - alpha * radius
    # The value remains the solved alpha.  The derivative is -F_theta/F_alpha.
    result = alpha - (residual - residual.detach()) / derivative_alpha
    axis = extents.detach().argmin(-1, keepdim=True)
    escape = coordinates.gather(-1, axis).squeeze(-1) / (extents.gather(-1, axis).squeeze(-1) + radius)
    return torch.where(coordinates.detach().square().sum(-1) == 0, escape, result)


def sphere_sphere_alpha(center, radius, other_center, other_radius):
    """Exact shared-scale alpha for two spheres, with ordinary broadcasting."""
    delta = center - _scalar_like(other_center, center)
    radius = _radius_like(radius, delta.shape[:-1], delta)
    other_radius = _radius_like(other_radius, delta.shape[:-1], delta)
    return delta.norm(dim=-1) / (radius + other_radius)


def sphere_box_alpha(center, radius, box_center, half_extents):
    """Sphere versus axis-aligned solid box; half_extents are in metres."""
    delta = center - _scalar_like(box_center, center)
    return _scale_from_coordinates(escape_abs(delta), _scalar_like(half_extents, center), radius)


def sphere_cylinder_alpha(center, radius, cylinder_center, cylinder_radius, half_height):
    """Sphere versus a solid finite cylinder whose axis is world Z."""
    delta = center - _scalar_like(cylinder_center, center)
    coordinates = torch.stack((escape_norm_xy(delta), escape_abs(delta[..., 2])), dim=-1)
    radial_extent, vertical_extent = torch.broadcast_tensors(
        _scalar_like(cylinder_radius, center), _scalar_like(half_height, center)
    )
    extents = torch.stack((radial_extent, vertical_extent), dim=-1)
    return _scale_from_coordinates(coordinates, extents, radius)


def _dynamic_pair_alpha(query, radius, positions, sizes):
    """Navigation's size-Z > 1 convention denotes finite Z-axis cylinders."""
    delta = query - positions
    cylinder = sizes[..., 2] > 1
    cylindrical_coordinates = torch.stack(
        (escape_norm_xy(delta), escape_abs(delta[..., 2]), torch.zeros_like(delta[..., 2])), dim=-1
    )
    cylindrical_extents = torch.stack(
        (sizes[..., 0] * 0.5, sizes[..., 2] * 0.5, torch.zeros_like(sizes[..., 2])), dim=-1
    )
    coordinates = torch.where(cylinder[..., None], cylindrical_coordinates, escape_abs(delta))
    extents = torch.where(cylinder[..., None], cylindrical_extents, sizes * 0.5)
    result = _scale_from_coordinates(coordinates, extents, radius)
    # The padded cylinder coordinate is not an escape axis.
    axis = extents.detach().masked_fill(cylinder[..., None] & (extents == 0), torch.inf).argmin(-1, keepdim=True)
    escape = coordinates.gather(-1, axis).squeeze(-1) / (extents.gather(-1, axis).squeeze(-1) + radius)
    return torch.where(coordinates.detach().square().sum(-1) == 0, escape, result)


def dynamic_min_alpha(query, radius, positions, sizes, *, obstacle_chunk_size=32):
    """Minimum DCOL alpha over all current dynamic primitives.

    ``query`` has shape (..., 3), and ``radius`` broadcasts to query.shape[:-1].
    Positions are shared (obstacles, 3) or have leading dimensions that
    broadcast to the query leading dimensions: (..., obstacles, 3).
    Sizes remain shared (obstacles, 3), with full sizes in metres.  For
    example, per-environment snapshots for query (N, S, 3) use positions
    (N, 1, obstacles, 3), whereas a terminal query (N, 3) uses (N, obstacles, 3).
    All candidates are evaluated in detached chunks.  Only the winning
    primitive is recomputed with a graph, so saved backward tensors do not
    grow with the number of obstacles or retain mutable scene buffers.
    Ties select one valid branch.  An empty obstacle set returns +inf with a
    zero query gradient, ready to combine with terrain/ground using minimum.
    """
    if obstacle_chunk_size < 1:
        raise ValueError("obstacle_chunk_size must be positive")
    if (
        query.ndim < 1
        or query.shape[-1] != 3
        or positions.ndim < 2
        or positions.shape[-1] != 3
        or sizes.ndim != 2
        or sizes.shape[-1] != 3
        or positions.shape[-2] != sizes.shape[0]
    ):
        raise ValueError("Expected query (..., 3), positions (..., obstacles, 3), and shared sizes (obstacles, 3)")
    shape = query.shape[:-1]
    try:
        position_shape = torch.broadcast_shapes(shape, positions.shape[:-2])
    except RuntimeError as error:
        raise ValueError("Positions leading dimensions must broadcast to the query batch") from error
    if position_shape != shape:
        raise ValueError("Positions leading dimensions must not expand the query batch")
    point = query.reshape(-1, 3)
    radii = _radius_like(radius, shape, query).reshape(-1)
    obstacle_count = sizes.shape[0]
    if obstacle_count == 0:
        return (point.sum(-1) * 0 + torch.inf).reshape(shape)

    with torch.no_grad():
        positions = positions.detach().to(device=query.device, dtype=query.dtype)
        sizes = sizes.detach().to(device=query.device, dtype=query.dtype)
        # This is a broadcast view.  Materialize at most one obstacle chunk
        # when flattening a non-contiguous per-environment snapshot batch.
        positions = torch.broadcast_to(positions, shape + (obstacle_count, 3))
        winner = torch.zeros(len(point), device=query.device, dtype=torch.long)
        best = torch.full((len(point),), torch.inf, device=query.device, dtype=query.dtype)
        for start in range(0, obstacle_count, obstacle_chunk_size):
            stop = min(start + obstacle_chunk_size, obstacle_count)
            chunk_positions = positions[..., start:stop, :].reshape(-1, stop - start, 3)
            alpha = _dynamic_pair_alpha(
                point.detach()[:, None, :],
                radii.detach()[:, None],
                chunk_positions,
                sizes[None, start:stop],
            )
            value, index = alpha.min(-1)
            improve = value < best
            winner = torch.where(improve, index + start, winner)
            best = torch.minimum(best, value)
        # Gather only one immutable position per query.  The full broadcast
        # position history and winner-selection intermediates stay out of
        # the backward graph, even when every environment used a different
        # physical-substep snapshot.
        selected_index = winner.reshape(shape + (1, 1)).expand(shape + (1, 3))
        selected_position = positions.gather(-2, selected_index).squeeze(-2).reshape(-1, 3)
        selected_size = sizes[winner]
    return _dynamic_pair_alpha(point, radii, selected_position, selected_size).reshape(shape)


def hybrid_collision_gap(query, radius, static_signed_distance, positions, sizes, ground_z=0.01):
    """Combine the verified-exit static clearance proxy with dynamic-body DCOL.

    ``static_signed_distance`` is the live differentiable metre distance from
    ``TerrainDistance.signed_distance``: positive outside and negative inside.
    Its inside derivative follows the selected original-mesh escape direction;
    it is not claimed to be the exact SDF of a self-intersecting terrain mesh.
    It is deliberately required instead of accepting old closest-point vectors,
    which would silently restore inward penetration gradients.
    Dynamic ``radius * (alpha - 1)`` is a metre-scaled DCOL proxy, not an SDF.
    The static query must exclude dynamic bodies, otherwise their old distance
    would also enter this minimum and mask the intended DCOL comparison.
    """
    if static_signed_distance.shape != query.shape[:-1]:
        raise ValueError("static_signed_distance must have shape query.shape[:-1], not closest-point vectors.")
    radii = _radius_like(radius, query.shape[:-1], query)
    static_gap = static_signed_distance - radii
    ground_gap = query[..., 2] - ground_z - radii
    dynamic_gap = radii * (dynamic_min_alpha(query, radii, positions, sizes) - 1)
    return torch.minimum(torch.minimum(static_gap, ground_gap), dynamic_gap)


def height_bounded_gap(query, obstacle_gap, height_bounds=(0.2, 4.0)):
    """Metre clearance to obstacles and the environment's centre-height bounds."""
    return torch.minimum(
        obstacle_gap, torch.minimum(query[..., 2] - height_bounds[0], height_bounds[1] - query[..., 2])
    )


def metric_navigation_gap(query, static_signed_distance, positions, sizes, radius=0.3):
    """Actual primitive/ground/height clearances for AHAC, independent of DCOL alpha.

    Static terrain retains its documented verified-exit signed proxy. Dynamic
    selection is detached; only the selected primitive keeps a backward graph.
    Positions may be shared or broadcast per-query substep snapshots.
    """
    from aerialnav.tasks.navigation.geometry import nearest_obstacle_signed_distance

    return height_bounded_gap(
        query, nearest_obstacle_signed_distance(query, static_signed_distance, positions, sizes) - radius
    )
