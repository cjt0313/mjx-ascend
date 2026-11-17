"""解析圆柱与凸顶点集的有界 GJK 分离查询。

圆柱轴为局部 Z，尺寸为半径/半高；顶点已在世界坐标。
只接受严格分离；没有 EPA 或通用接触流形实现。
所有运行时数值计算为 JAX FP32；保留失败状态与 NaN 输出。
"""

import itertools
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jp


class Status(IntEnum):
    SEPARATED = 0
    PENETRATING = 1
    GJK_LIMIT = 2
    EPA_LIMIT = 3
    EPA_OVERFLOW = 4
    DEGENERATE = 5
    INVALID_INPUT = 6
    TOUCHING = 8


class CollisionResult(NamedTuple):
    distance: jax.Array
    normal: jax.Array
    cylinder_witness: jax.Array
    mesh_witness: jax.Array
    contact: jax.Array
    status: jax.Array
    valid: jax.Array
    gjk_iterations: jax.Array
    epa_iterations: jax.Array
    error_bound: jax.Array


def _norm(vector):
    return jp.sqrt(jp.sum(vector * vector, axis=-1))


def cylinder_support(position, rotation, size, direction):
    """Exact support, including cap interior for an axial direction."""
    local = rotation.T @ direction
    radial = _norm(local[:2])
    planar = size[0] * local[:2] / jp.where(radial > 0, radial, 1.0)
    axial = size[1] * jp.sign(local[2])
    return position + rotation @ jp.concatenate((planar, axial[None]))


def mesh_support(world_vertices, direction):
    return world_vertices[jp.argmax(world_vertices @ direction)]


def _triangle_weights(triangle, point):
    edge_first = triangle[..., 1, :] - triangle[..., 0, :]
    edge_second = triangle[..., 2, :] - triangle[..., 0, :]
    offset = point - triangle[..., 0, :]
    normal = jp.cross(edge_first, edge_second)
    determinant = jp.sum(normal * normal, axis=-1)
    safe_determinant = jp.where(determinant > 1e-12, determinant, 1.0)
    offset = offset - normal * (jp.sum(offset * normal, axis=-1) / safe_determinant)[..., None]
    axis = jp.argmax(normal * normal, axis=-1)
    component = jp.take_along_axis(normal, axis[..., None], axis=-1)[..., 0]
    denominator = jp.where(determinant > 1e-12, component, 1.0)
    first_cross = jp.cross(offset, edge_second)
    second_cross = jp.cross(edge_first, offset)
    first = jp.take_along_axis(first_cross, axis[..., None], axis=-1)[..., 0] / denominator
    second = jp.take_along_axis(second_cross, axis[..., None], axis=-1)[..., 0] / denominator
    return jp.stack((1 - first - second, first, second), axis=-1), determinant > 1e-12


def _edge_projection(base, edge, edge_sq):
    denominator = jp.where(edge_sq > 1e-12, edge_sq, 1.0)
    return jp.cross(edge, jp.cross(base, edge)) / denominator[..., None]


def _face_projection(triangle):
    normal = jp.cross(
        triangle[..., 1, :] - triangle[..., 0, :], triangle[..., 2, :] - triangle[..., 0, :]
    )
    normal_sq = jp.sum(normal * normal, axis=-1)
    denominator = jp.where(normal_sq > 1e-12, normal_sq, 1.0)
    return normal * (jp.sum(normal * triangle[..., 0, :], axis=-1) / denominator)[..., None]


def _tetra_weights(tetra):
    base = tetra[..., 0, :]
    edge_first = tetra[..., 1, :] - base
    edge_second = tetra[..., 2, :] - base
    edge_third = tetra[..., 3, :] - base
    determinant = jp.sum(edge_first * jp.cross(edge_second, edge_third), axis=-1)
    denominator = jp.where(jp.abs(determinant) > 1e-9, determinant, 1.0)
    first = jp.sum(-base * jp.cross(edge_second, edge_third), axis=-1) / denominator
    second = jp.sum(edge_first * jp.cross(-base, edge_third), axis=-1) / denominator
    third = jp.sum(edge_first * jp.cross(edge_second, -base), axis=-1) / denominator
    return jp.stack((1 - first - second - third, first, second, third), axis=-1), jp.abs(
        determinant
    ) > 1e-9


def _closest_triangles(triangles):
    pairs = jp.array([[0, 1], [1, 2], [2, 0]])
    edges = triangles[:, pairs[:, 1]] - triangles[:, pairs[:, 0]]
    edge_sq = jp.sum(edges * edges, axis=-1)
    fraction = jp.clip(
        -jp.sum(triangles[:, pairs[:, 0]] * edges, axis=-1)
        / jp.where(edge_sq > 1e-12, edge_sq, 1.0),
        0.0,
        1.0,
    )
    pair_weights = (
        jp.eye(3)[pairs[:, 0]] * (1 - fraction[..., None])
        + jp.eye(3)[pairs[:, 1]] * fraction[..., None]
    )
    face_weights, face_good = _triangle_weights(triangles, jp.zeros(3))
    vertex_weights = jp.broadcast_to(jp.eye(3), (triangles.shape[0], 3, 3))
    weights = jp.concatenate((vertex_weights, pair_weights, face_weights[:, None]), axis=1)
    candidates = jp.sum(weights[..., None] * triangles[:, None], axis=2)
    eligible = jp.concatenate(
        (
            jp.ones((triangles.shape[0], 6), dtype=jp.bool_),
            (face_good & jp.all(face_weights >= 0, axis=-1))[:, None],
        ),
        axis=1,
    )
    scores = jp.where(eligible, jp.sum(candidates * candidates, axis=-1), jp.inf)
    best = jp.argmin(scores, axis=1)
    rows = jp.arange(triangles.shape[0])
    return candidates[rows, best], weights[rows, best]


def _simplex_scores(candidates, edge_base, edges, fraction):
    affine_edges = edge_base + fraction[:, None] * edges
    return jp.concatenate(
        (
            jp.sum(candidates[:4] * candidates[:4], axis=1),
            jp.sum(affine_edges * affine_edges, axis=1),
            jp.sum(candidates[10:] * candidates[10:], axis=1),
        )
    )


def _closest_simplex(points, active):
    pairs = jp.array(list(itertools.combinations(range(4), 2)))
    triples = jp.array(list(itertools.combinations(range(4), 3)))
    edges = points[pairs[:, 1]] - points[pairs[:, 0]]
    edge_sq = jp.sum(edges * edges, axis=-1)
    fraction = -jp.sum(points[pairs[:, 0]] * edges, axis=-1) / jp.where(
        edge_sq > 1e-12, edge_sq, 1.0
    )
    pair_weights = (
        jp.eye(4)[pairs[:, 0]] * (1 - fraction[:, None])
        + jp.eye(4)[pairs[:, 1]] * fraction[:, None]
    )
    triangle_weights, triangle_good = _triangle_weights(points[triples], jp.zeros(3))
    triple_weights = jp.sum(jp.eye(4)[triples] * triangle_weights[..., None], axis=1)
    tetra_weights, tetra_good = _tetra_weights(points)
    weights = jp.concatenate((jp.eye(4), pair_weights, triple_weights, tetra_weights[None]))
    eligible = jp.concatenate(
        (
            active,
            jp.all(active[pairs], axis=1) & (edge_sq > 1e-12),
            jp.all(active[triples], axis=1) & triangle_good,
            (jp.all(active) & tetra_good)[None],
        )
    )
    eligible &= jp.all(weights >= 0.0, axis=1)
    candidates = jp.concatenate(
        (
            points,
            _edge_projection(points[pairs[:, 0]], edges, edge_sq),
            _face_projection(points[triples]),
            (tetra_weights @ points)[None],
        )
    )
    scores = jp.where(
        eligible, _simplex_scores(candidates, points[pairs[:, 0]], edges, fraction), jp.inf
    )
    best = jp.argmin(scores)
    consistent = _norm(weights[best] @ points - candidates[best]) <= 1e-6
    return candidates[best], weights[best], jp.isfinite(scores[best]) & consistent


def collide_cylinder_mesh(
    cylinder_pos,
    cylinder_mat,
    cylinder_size,
    mesh_vertices,
    *,
    max_gjk_iterations=64,
    tolerance=1e-5,
):
    """返回分离距离与 witness；接触、穿透、退化及超限查询明确拒绝。"""
    if max_gjk_iterations < 0 or tolerance <= 0:
        raise ValueError("Iteration bounds must be nonnegative and tolerance positive")
    if mesh_vertices.ndim != 2 or mesh_vertices.shape[1] != 3 or mesh_vertices.shape[0] < 4:
        raise ValueError("mesh_vertices must have shape (N >= 4, 3)")
    if (
        cylinder_pos.shape != (3,)
        or cylinder_mat.shape != (3, 3)
        or cylinder_size.shape not in ((2,), (3,))
    ):
        raise ValueError("Expected cylinder position (3,), rotation (3,3), size (2,) or (3,)")
    cylinder_pos = jp.asarray(cylinder_pos, dtype=jp.float32)
    cylinder_mat = jp.asarray(cylinder_mat, dtype=jp.float32)
    cylinder_size = jp.asarray(cylinder_size[:2], dtype=jp.float32)
    mesh_vertices = jp.asarray(mesh_vertices, dtype=jp.float32)
    extent = jp.max(jp.max(mesh_vertices, axis=0) - jp.min(mesh_vertices, axis=0))
    scale = jp.maximum(jp.maximum(jp.max(cylinder_size), extent), 1e-6)
    local_vertices = (mesh_vertices - cylinder_pos) / scale
    size = cylinder_size / scale

    def support(direction):
        cylinder = cylinder_support(jp.zeros(3), cylinder_mat, size, direction)
        mesh = mesh_support(local_vertices, -direction)
        return cylinder - mesh, cylinder, mesh

    def result(distance, normal, cylinder, mesh, status, gjk_count, epa_count, gap):
        valid = (
            (status == Status.SEPARATED)
            | (status == Status.PENETRATING)
            | (status == Status.TOUCHING)
        )
        cylinder = jp.where(valid, cylinder * scale + cylinder_pos, jp.nan)
        mesh = jp.where(valid, mesh * scale + cylinder_pos, jp.nan)
        return CollisionResult(
            jp.where(valid, distance * scale, jp.nan),
            jp.where(valid, normal, jp.nan),
            cylinder,
            mesh,
            (cylinder + mesh) / 2,
            jp.asarray(status, jp.int32),
            valid,
            jp.asarray(gjk_count, jp.int32),
            jp.asarray(epa_count, jp.int32),
            jp.where(valid, gap * scale, jp.nan),
        )

    def failure(status, gjk_count=0, epa_count=0):
        return result(0.0, jp.zeros(3), jp.zeros(3), jp.zeros(3), status, gjk_count, epa_count, 0.0)

    def gjk(_):
        direction = jp.mean(local_vertices, axis=0)
        direction = jp.where(_norm(direction) > 1e-7, direction, jp.array([1.0, 0.0, 0.0]))
        point, cylinder, mesh = support(direction)
        initial = (
            jp.int32(0),
            jp.zeros((4, 3)).at[0].set(point),
            jp.zeros((4, 3)).at[0].set(cylinder),
            jp.zeros((4, 3)).at[0].set(mesh),
            jp.array([True, False, False, False]),
            jp.int32(-1),
            jp.zeros(4),
            jp.float32(jp.inf),
        )

        def condition(state):
            return (state[0] < max_gjk_iterations) & (state[5] == -1)

        def step(state):
            count, points, cylinders, meshes, active, status, _, _ = state
            closest, weights, good = _closest_simplex(points, active)
            distance = _norm(closest)
            point, cylinder, mesh = support(-closest)
            gap = (jp.sum(closest * closest) - jp.sum(closest * point)) / jp.maximum(
                distance, 1e-12
            )
            hit = distance <= tolerance
            converged = (gap <= tolerance) & ~hit
            keep = active & (weights > 1e-7)
            slot = jp.argmin(keep.astype(jp.int32))
            duplicate = jp.any(active & (_norm(points - point) < 1e-7))
            bad = ~good | (gap < -tolerance) | ((duplicate | jp.all(keep)) & ~hit & ~converged)
            status = jp.where(
                bad, Status.DEGENERATE, jp.where(hit, 7, jp.where(converged, Status.SEPARATED, -1))
            ).astype(jp.int32)
            advance = status == -1
            points = jp.where(advance, points.at[slot].set(point), points)
            cylinders = jp.where(advance, cylinders.at[slot].set(cylinder), cylinders)
            meshes = jp.where(advance, meshes.at[slot].set(mesh), meshes)
            active = jp.where(advance, keep.at[slot].set(True), active)
            return (
                count + 1,
                points,
                cylinders,
                meshes,
                active,
                status,
                weights,
                jp.maximum(gap, 0.0),
            )

        count, points, cylinders, meshes, active, status, weights, gap = jax.lax.while_loop(
            condition, step, initial
        )

        def separation(_):
            cylinder = weights @ cylinders
            mesh = weights @ meshes
            closest, _, good = _closest_simplex(points, active)
            distance = _norm(closest)
            normal = -closest / jp.maximum(distance, 1e-12)
            consistent = good & (_norm(mesh - cylinder - normal * distance) <= tolerance)
            return result(
                distance,
                normal,
                cylinder,
                mesh,
                jp.where(consistent, Status.SEPARATED, Status.DEGENERATE),
                count,
                0,
                gap,
            )

        return jax.lax.cond(
            status == Status.SEPARATED,
            separation,
            lambda _: failure(
                jp.where(
                    status == -1,
                    Status.GJK_LIMIT,
                    jp.where(status == 7, Status.DEGENERATE, status),
                ),
                count,
            ),
            None,
        )

    finite = (
        jp.all(jp.isfinite(cylinder_pos))
        & jp.all(jp.isfinite(cylinder_mat))
        & jp.all(jp.isfinite(cylinder_size))
        & jp.all(jp.isfinite(mesh_vertices))
    )
    rotation_good = jp.max(jp.abs(cylinder_mat.T @ cylinder_mat - jp.eye(3))) < 1e-4
    orientation = jp.sum(cylinder_mat[:, 0] * jp.cross(cylinder_mat[:, 1], cylinder_mat[:, 2]))
    inputs_good = finite & jp.all(cylinder_size > 0) & rotation_good & (orientation > 0)
    centered = local_vertices - jp.mean(local_vertices, axis=0)
    covariance = centered.T @ centered / mesh_vertices.shape[0]
    volume = jp.sum(covariance[0] * jp.cross(covariance[1], covariance[2]))
    return jax.lax.cond(
        inputs_good,
        lambda _: jax.lax.cond(volume > 1e-12, gjk, lambda _: failure(Status.DEGENERATE), None),
        lambda _: failure(Status.INVALID_INPUT),
        None,
    )
