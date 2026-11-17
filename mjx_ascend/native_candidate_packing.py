"""Fixed-shape candidate packing for homogeneous collider groups.

Uncertified sphere bounds, including invalid inputs, retain the original
narrowphase. Certified-far diagnostics intentionally differ from narrowphase
diagnostics. Callers own geometry-family/radius/margin validity and integration.
This module does not establish native execution, speed or scene precision.
Native lax.cond control currently synchronizes its selector with the host;
packing arithmetic and geometry selection remain JAX device operations.
"""

from functools import wraps
import math
from numbers import Integral, Real

import jax
import jax.numpy as jp

from .native_mesh_broadphase import sphere_separation_lower_bound


def wrap_candidate_packing(function, ncon, margin, capacity):
    """Preserve pair-major slots while evaluating a stable fixed-size queue.

    Zero candidates skip narrowphase; overflow calls the original full group.
    Otherwise padding repeats the first candidate and is never restored into
    output slots. No geometry leaves the device and no registry is modified.
    """
    if isinstance(margin, bool) or not isinstance(margin, Real):
        raise TypeError("margin must be a static real scalar")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    for name, value in (("ncon", ncon), ("capacity", capacity)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a static integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if hasattr(function, "ncon") and function.ncon != ncon:
        raise ValueError("ncon must match the original collider")

    @wraps(function)
    def execute(model, data, key, geometry):
        lower, _ = sphere_separation_lower_bound(
            data.geom_xpos, model.geom_rbound, geometry, margin
        )
        candidate = ~(lower > margin)
        ranks = jp.cumsum(candidate.astype(jp.int32), dtype=jp.int32) - 1
        count = jp.sum(candidate.astype(jp.int32), dtype=jp.int32)
        pair_count = geometry.shape[0]

        def inactive(_):
            centers = data.geom_xpos[geometry]
            midpoint = centers[:, 0] * 0.5 + centers[:, 1] * 0.5
            distance = jp.repeat(lower, ncon, axis=0)
            position = jp.repeat(midpoint, ncon, axis=0)
            frame = jp.broadcast_to(jp.eye(3, dtype=position.dtype), (pair_count * ncon, 3, 3))
            return distance, position, frame

        if pair_count == 0:
            return inactive(None)

        def packed(_):
            pair_indices = jp.arange(pair_count, dtype=jp.int32)
            slot_indices = jp.arange(capacity, dtype=jp.int32)
            matches = candidate[None, :] & (ranks[None, :] == slot_indices[:, None])
            selected = jp.min(jp.where(matches, pair_indices[None, :], pair_count), axis=1)
            selected = jp.where(selected < pair_count, selected, selected[0])
            packed_output = function(model, data, key, geometry[selected])
            inactive_output = inactive(None)
            inverse = jp.clip(ranks, 0, capacity - 1)
            restored = []
            for active, far in zip(packed_output, inactive_output):
                tail_shape = active.shape[1:]
                active_pairs = active.reshape((capacity, ncon) + tail_shape)[inverse]
                far_pairs = far.reshape((pair_count, ncon) + tail_shape)
                mask = candidate.reshape((pair_count,) + (1,) * (active_pairs.ndim - 1))
                restored.append(jp.where(mask, active_pairs, far_pairs).reshape(far.shape))
            return tuple(restored)

        def nonzero(_):
            return jax.lax.cond(
                count <= capacity,
                packed,
                lambda _: function(model, data, key, geometry),
                operand=None,
            )

        return jax.lax.cond(count == 0, inactive, nonzero, operand=None)

    execute.ncon = ncon
    return execute
