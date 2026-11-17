"""Bounded MuJoCo 3.2.7 box-face vertex contacts, with stock fallback.

The caller guards the engine version and zero inclusion margins/gaps. Only
strictly contained opposite-box projections with one to four penetrating
vertices are supported, including native incident-face and driver pruning.
Geometry, packing and fallback selection stay in JAX;
the original collider is called once and every allocated frame is preserved.
"""

from functools import wraps

import jax.numpy as jp


def _box_position_status(centers, matrices, sizes, positions, pad):
    relative = positions[:, :, None, :] - centers[:, None, :, :]
    local = jp.abs(jp.sum(relative[:, :, :, :, None] * matrices[:, None, :, :, :], axis=3))
    inflated = sizes[:, None, :, :] * 1.01
    deflated = sizes[:, None, :, :] / 1.01
    outside = jp.any(local > inflated, axis=3)
    inside = jp.all(local < deflated, axis=3)
    uncertainty = pad[:, None, None, None]
    outside_lower = jp.any(local > inflated + uncertainty, axis=3)
    outside_upper = jp.any(local > inflated - uncertainty, axis=3)
    inside_lower = jp.all(local < deflated - uncertainty, axis=3)
    inside_upper = jp.all(local < deflated + uncertainty, axis=3)
    removal_lower = (outside_lower[:, :, 0] & ~inside_upper[:, :, 1]) | (
        outside_lower[:, :, 1] & ~inside_upper[:, :, 0]
    )
    removal_upper = (outside_upper[:, :, 0] & ~inside_lower[:, :, 1]) | (
        outside_upper[:, :, 1] & ~inside_lower[:, :, 0]
    )
    stable = removal_lower == removal_upper
    removed = (outside[:, :, 0] & ~inside[:, :, 1]) | (outside[:, :, 1] & ~inside[:, :, 0])
    return removed, stable


def _face_contacts(centers, matrices, sizes, normal, reference):
    opposite = 1 - reference
    outward = normal if reference == 0 else -normal
    local_normal = jp.sum(matrices[:, reference] * outward[:, :, None], axis=1)
    axis = jp.argmax(jp.abs(local_normal), axis=1)
    tangential = jp.arange(3)[None, :] != axis[:, None]
    aligned = (
        (jp.max(jp.where(tangential, jp.abs(local_normal), 0), axis=1) <= 2e-7)
        & (jp.max(jp.abs(local_normal), axis=1) >= 1 - 2e-7)
        & (jp.max(jp.abs(local_normal), axis=1) <= 1 + 2e-7)
    )
    signs = jp.asarray(
        (
            (-1, -1, -1),
            (-1, -1, 1),
            (-1, 1, -1),
            (-1, 1, 1),
            (1, -1, -1),
            (1, -1, 1),
            (1, 1, -1),
            (1, 1, 1),
        ),
        dtype=centers.dtype,
    )
    vertices = centers[:, opposite, None, :] + jp.sum(
        matrices[:, opposite, None, :, :]
        * (signs[None, :, :] * sizes[:, opposite, None, :])[:, :, None, :],
        axis=3,
    )
    relative = vertices - centers[:, reference, None, :]
    local_vertices = jp.sum(relative[:, :, :, None] * matrices[:, reference, None, :, :], axis=2)
    pad = (
        64
        * jp.finfo(centers.dtype).eps
        * (jp.sum(jp.abs(centers), axis=(1, 2)) + jp.sum(sizes, axis=(1, 2)) + 1)
        + 1e-6
    )
    contained = jp.all(
        jp.where(
            tangential[:, None, :],
            jp.abs(local_vertices) < sizes[:, reference, None, :] - pad[:, None, None],
            True,
        ),
        axis=(1, 2),
    )
    support = jp.sum(jp.abs(local_normal) * sizes[:, reference], axis=1)
    depth = jp.sum(relative * outward[:, None, :], axis=2) - support[:, None]
    incident_normal = jp.sum(matrices[:, opposite] * outward[:, :, None], axis=1)
    incident_edges = jp.abs(incident_normal) < 0.5
    corner = jp.where(incident_normal < 0, 1, -1)
    incident = jp.all(
        incident_edges[:, None, :] | (signs[None, :, :] == corner[:, None, :]), axis=2
    )
    candidates = incident & (depth < 0)
    midpoints = vertices - depth[:, :, None] * 0.5 * outward[:, None, :]
    removed, stable_pruning = _box_position_status(centers, matrices, sizes, midpoints, pad)
    penetrating = candidates & ~removed
    count = jp.sum(penetrating, axis=1)
    cumulative = jp.cumsum(penetrating.astype(jp.int32).reshape(-1), dtype=jp.int32).reshape(-1, 8)
    offsets = jp.cumsum(count, dtype=jp.int32) - count
    ranks = cumulative - offsets[:, None] - 1
    indices = jp.min(
        jp.where(
            penetrating[:, None, :] & (ranks[:, None, :] == jp.arange(4)[None, :, None]),
            jp.arange(8)[None, None, :],
            8,
        ),
        axis=2,
    )
    safe_indices = jp.minimum(indices, 7)
    flat_indices = (jp.arange(centers.shape[0])[:, None] * 8 + safe_indices).reshape(-1)
    selected_depth = depth.reshape(-1)[flat_indices].reshape(-1, 4)
    selected_vertex = vertices.reshape(-1, 3)[flat_indices].reshape(-1, 4, 3)
    positions = selected_vertex - selected_depth[:, :, None] * 0.5 * outward[:, None, :]
    distances = jp.where(indices < 8, selected_depth * 0.5, 1)
    certified = (
        aligned
        & contained
        & (count >= 1)
        & (count <= 4)
        & (jp.sum(depth < 0, axis=1) <= 4)
        & (jp.sum(incident_edges, axis=1) <= 2)
        & jp.all(jp.abs(jp.abs(incident_normal) - 0.5) > 2e-7, axis=1)
        & jp.all(~incident | (jp.abs(depth) > pad[:, None]), axis=1)
        & jp.all(~candidates | stable_pruning, axis=1)
        & jp.all(jp.isfinite(vertices), axis=(1, 2))
        & jp.all(jp.isfinite(local_vertices), axis=(1, 2))
        & jp.all(jp.isfinite(depth), axis=1)
        & jp.all(jp.isfinite(positions), axis=(1, 2))
        & jp.isfinite(pad)
    )
    return certified, distances, positions


def wrap_native_contained_box_faces(function):
    """Keep four pair-major slots; unsupported geometry remains unchanged."""
    if getattr(function, "ncon", None) != 4:
        raise ValueError("contained box faces require the stock four-contact collider")

    @wraps(function)
    def execute(model, data, key, geometry):
        distance, position, frame = function(model, data, key, geometry)
        pair_count = geometry.shape[0]
        centers = data.geom_xpos[geometry]
        matrices = data.geom_xmat[geometry]
        sizes = model.geom_size[geometry]
        frames = frame.reshape(pair_count, 4, 3, 3)
        normal = frames[:, 0, 0, :]
        norm_squared = jp.sum(normal * normal, axis=1)
        unit_normal = normal / jp.sqrt(jp.where(norm_squared > 0, norm_squared, 1))[:, None]
        gram = jp.sum(matrices[:, :, :, :, None] * matrices[:, :, :, None, :], axis=2)
        valid = (
            jp.all(
                (geometry >= 0)
                & (geometry < data.geom_xpos.shape[0])
                & (geometry < model.geom_size.shape[0])
                & (geometry < data.geom_xmat.shape[0]),
                axis=1,
            )
            & jp.all(jp.isfinite(centers), axis=(1, 2))
            & jp.all(jp.isfinite(matrices), axis=(1, 2, 3))
            & jp.all(jp.abs(gram - jp.eye(3)) < 2e-6, axis=(1, 2, 3))
            & jp.all(jp.isfinite(sizes) & (sizes > 0), axis=(1, 2))
            & jp.all(jp.isfinite(distance.reshape(pair_count, 4)), axis=1)
            & jp.any(distance.reshape(pair_count, 4) < 0, axis=1)
            & jp.all(jp.isfinite(position.reshape(pair_count, 4, 3)), axis=(1, 2))
            & jp.all(jp.isfinite(frames), axis=(1, 2, 3))
            & jp.all(jp.abs(frames[:, :, 0, :] - normal[:, None, :]) < 2e-7, axis=(1, 2))
            & jp.isfinite(norm_squared)
            & (jp.abs(norm_squared - 1) < 2e-6)
        )
        first, first_distance, first_position = _face_contacts(
            centers, matrices, sizes, unit_normal, 0
        )
        second, second_distance, second_position = _face_contacts(
            centers, matrices, sizes, unit_normal, 1
        )
        certified = valid & (first | second)
        corrected_distance = jp.where(first[:, None], first_distance, second_distance).reshape(-1)
        corrected_position = jp.where(
            first[:, None, None], first_position, second_position
        ).reshape(-1, 3)
        selected = jp.repeat(certified, 4)
        return (
            jp.where(selected, corrected_distance, distance),
            jp.where(selected[:, None], corrected_position, position),
            frame,
        )

    execute.ncon = function.ncon
    return execute
