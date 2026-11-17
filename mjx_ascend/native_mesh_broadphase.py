"""Whole-chunk sphere certificates without changing collider slot allocation.

The caller must restrict wrapping to supported bounded convex families (the
staged caller selects BOX-MESH/MESH-MESH and opt-in CYLINDER-BOX/CYLINDER-MESH),
validate that radii
bound those geometries, and choose a conservative inclusion margin before JIT.
That global bound must include negative geom gaps and explicit pair margin-gap,
as well as any supported overrides. A conservatively positive bound is advised.
Uncertain chunks execute the original function unchanged. Certified inactive
slots intentionally do not preserve exact far-pair narrowphase diagnostics.

``lax.cond`` describes conditional execution, not guaranteed fused device control:
the current native NpuConditionalThunk synchronizes a scalar selector to host.
Geometry predicates stay on device; native runtime/performance validation and
trajectory/contact-force integration remain the caller's responsibility.
"""

from functools import wraps
import math
from numbers import Integral, Real

import jax
import jax.numpy as jp


def _validate_margin(margin):
    if isinstance(margin, bool) or not isinstance(margin, Real):
        raise TypeError("margin must be a static real scalar")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")


def sphere_separation_lower_bound(positions, radii, geometry, margin):
    """Return margin-independent per-pair lower bounds and scalar all-certified.

    Inputs have shapes ``(ngeom, 3)``, ``(ngeom,)``, and ``(npairs, 2)``.
    Margin is a static finite nonnegative Python real. Invalid pair inputs or
    overflowing intermediate arithmetic yield ``-inf`` and prohibit skipping.
    The pad is at least 64 dtype eps times coordinate scale plus radii plus one,
    plus 1e-5. Float32 is the minimum arithmetic precision. No data leave JAX.
    """
    _validate_margin(margin)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (ngeom, 3)")
    if radii.shape != (positions.shape[0],):
        raise ValueError("radii must have shape (ngeom,)")
    if geometry.ndim != 2 or geometry.shape[1] != 2:
        raise ValueError("geometry must have shape (npairs, 2)")
    if not jp.issubdtype(geometry.dtype, jp.integer):
        raise TypeError("geometry must contain integer indices")
    if not jp.issubdtype(positions.dtype, jp.floating) or not jp.issubdtype(
        radii.dtype, jp.floating
    ):
        raise TypeError("positions and radii must be real floating arrays")
    dtype = jp.result_type(positions.dtype, radii.dtype, jp.float32)
    positions = positions.astype(dtype)
    radii = radii.astype(dtype)
    if positions.shape[0] == 0:
        lower = jp.full((geometry.shape[0],), -jp.inf, dtype=dtype)
        return lower, jp.array(False)
    centers = positions[geometry]
    pair_radii = radii[geometry]
    difference = centers[:, 1] - centers[:, 0]
    distance = jp.sqrt(jp.sum(difference * difference, axis=1))
    radius_sum = jp.sum(pair_radii, axis=1)
    coordinate_scale = jp.sum(jp.max(jp.abs(centers), axis=2), axis=1)
    pad = 64 * jp.finfo(dtype).eps * (coordinate_scale + radius_sum + 1) + 1e-5
    gap = distance - radius_sum
    lower = gap - pad
    valid = (
        jp.all(jp.isfinite(centers), axis=(1, 2))
        & jp.all(jp.isfinite(pair_radii) & (pair_radii > 0), axis=1)
        & jp.all((geometry >= 0) & (geometry < positions.shape[0]), axis=1)
        & jp.isfinite(distance)
        & jp.isfinite(radius_sum)
        & jp.isfinite(pad)
        & jp.isfinite(gap)
        & jp.isfinite(lower)
    )
    lower = jp.where(valid, lower, -jp.inf)
    return lower, jp.all(valid & (lower > margin))


def wrap_mesh_broadphase(function, ncon, margin):
    """Wrap ``function(model, data, key, geometry)`` with whole-chunk rejection.

    ``ncon`` is the unchanged positive static contact count per pair. Both
    branches return distances, positions and frames in MJX's pair-major order.
    Only margin and slot-count guards are owned here; family/model-option guards
    belong to the integrating caller. No collider registry is modified.
    """
    _validate_margin(margin)
    if isinstance(ncon, bool) or not isinstance(ncon, Integral):
        raise TypeError("ncon must be a static integer")
    if ncon < 1:
        raise ValueError("ncon must be positive")
    if hasattr(function, "ncon") and function.ncon != ncon:
        raise ValueError("ncon must match the original collider")

    @wraps(function)
    def execute(model, data, key, geometry):
        lower, can_skip = sphere_separation_lower_bound(
            data.geom_xpos, model.geom_rbound, geometry, margin
        )

        def inactive(_):
            centers = data.geom_xpos[geometry]
            midpoint = centers[:, 0] * 0.5 + centers[:, 1] * 0.5
            dist = jp.repeat(lower, ncon, axis=0)
            pos = jp.repeat(midpoint, ncon, axis=0)
            frame = jp.broadcast_to(jp.eye(3, dtype=pos.dtype), (dist.shape[0], 3, 3))
            return dist, pos, frame

        return jax.lax.cond(
            can_skip, inactive, lambda _: function(model, data, key, geometry), operand=None
        )

    execute.ncon = ncon
    return execute


def wrap_group_broadphase(function, ncon, margin):
    """Host-schedule a whole-group certificate before the supplied chunk path.

    Only ``bool(can_skip)`` synchronizes to host. The sphere certificate and
    inactive slot construction are separate native JIT calls on live geometry.
    Uncertain groups call ``function`` unchanged, retaining its chunk scheduler.
    Certified slots follow ``wrap_mesh_broadphase`` semantics and pair order.
    This standalone scheduler rejects tracing, including nested JIT execution.
    Bounded-family/radius and conservative-margin contracts remain caller-owned.
    """
    _validate_margin(margin)
    if isinstance(ncon, bool) or not isinstance(ncon, Integral):
        raise TypeError("ncon must be a static integer")
    if ncon < 1:
        raise ValueError("ncon must be positive")
    if hasattr(function, "ncon") and function.ncon != ncon:
        raise ValueError("ncon must match the original collider")

    @jax.jit
    def certificate(positions, radii, geometry):
        return sphere_separation_lower_bound(positions, radii, geometry, margin)

    @jax.jit
    def inactive(positions, geometry, lower):
        centers = positions[geometry]
        midpoint = centers[:, 0] * 0.5 + centers[:, 1] * 0.5
        dist = jp.repeat(lower, ncon, axis=0)
        pos = jp.repeat(midpoint, ncon, axis=0)
        frame = jp.broadcast_to(jp.eye(3, dtype=pos.dtype), (dist.shape[0], 3, 3))
        return dist, pos, frame

    @wraps(function)
    def execute(model, data, key, geometry):
        leaves = jax.tree_util.tree_leaves(
            (model, data, key, geometry, data.geom_xpos, model.geom_rbound)
        )
        if not jax.core.trace_ctx.is_top_level() or any(
            isinstance(value, jax.core.Tracer) for value in leaves
        ):
            raise TypeError(
                "Whole-group broadphase is a host scheduler; tracing/nested jit is unsupported"
            )
        lower, can_skip = certificate(data.geom_xpos, model.geom_rbound, geometry)
        if bool(can_skip):
            return inactive(data.geom_xpos, geometry, lower)
        return function(model, data, key, geometry)

    execute.ncon = ncon
    return execute
