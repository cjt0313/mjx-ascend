"""Bounded JIT scheduling for the pinned MJX step, not a physics backend."""

from contextlib import contextmanager
from functools import partial, wraps
import math
from numbers import Real
import time

import jax
import jax.numpy as jp
from mujoco.mjx._src import collision_driver, constraint, forward, sensor, smooth, solver
from mujoco.mjx._src.types import GeomType


_STAGES = (
    (smooth, "kinematics"),
    (smooth, "com_pos"),
    (smooth, "camlight"),
    (smooth, "tendon"),
    (smooth, "crb"),
    (smooth, "factor_m"),
    (smooth, "transmission"),
    (constraint, "make_constraint"),
    (forward, "fwd_velocity"),
    (forward, "fwd_actuation"),
    (forward, "fwd_acceleration"),
    (forward, "euler"),
    (sensor, "sensor_pos"),
    (sensor, "sensor_vel"),
    (sensor, "sensor_acc"),
    (solver, "solve"),
)


@partial(jax.jit, static_argnames=("count",))
def _geometry_chunk(geometry, start, *, count):
    return geometry[jp.arange(count, dtype=jp.int32) + start]


def _chunk_collision(compiled, batch_size, box_mesh_batch_overrides=None):
    @wraps(compiled)
    def execute(model, data, key, geometry):
        selected_size = batch_size
        if box_mesh_batch_overrides and key.types == (GeomType.BOX, GeomType.MESH):
            selected_size = box_mesh_batch_overrides.get(key.data_ids[1], batch_size)
        if geometry.shape[0] <= selected_size:
            return compiled(model, data, key, geometry)
        outputs = [
            compiled(
                model,
                data,
                key,
                _geometry_chunk(
                    geometry,
                    jp.array(start, dtype=jp.int32),
                    count=min(selected_size, geometry.shape[0] - start),
                ),
            )
            for start in range(0, geometry.shape[0], selected_size)
        ]
        return tuple(jp.concatenate(values, axis=0) for values in zip(*outputs))

    execute.ncon = compiled.ncon
    return execute


def _observe_timing(compiled, name, observer):
    @wraps(compiled)
    def execute(model, data, *args, **kwargs):
        cache_size = getattr(compiled, "_cache_size", lambda: None)
        before = cache_size()
        started = time.perf_counter()
        tracing = False
        record = {"stage": name, "status": "failed", "cache_before": before}
        if name.startswith("collision:") and len(args) >= 2:
            record.update(group=str(args[0]), pairs=int(args[1].shape[0]))
        try:
            result = compiled(model, data, *args, **kwargs)
            tracing = any(
                isinstance(value, jax.core.Tracer) for value in jax.tree_util.tree_leaves(result)
            )
            if tracing:
                return result
            jax.block_until_ready(result)
            record["status"] = "complete"
            return result
        finally:
            if not tracing:
                after = cache_size()
                record.update(
                    seconds=time.perf_counter() - started,
                    cache_after=after,
                    cache_grew=None if before is None or after is None else after > before,
                )
                observer(record)

    return execute


@contextmanager
def staged_kernels(
    collision_batch_size=None,
    factored_elliptic_hessian=False,
    stage_timing_observer=None,
    box_mesh_batch_overrides=None,
    mesh_broadphase_margin=None,
    cylinder_broadphase=False,
    cylinder_collision_batch_size=None,
    native_parallel_box_faces=False,
    stable_elliptic_linesearch=False,
    cylinder_candidate_capacity=None,
    cache_contact_templates=False,
    group_mesh_broadphase=False,
):
    """Temporarily JIT MJX kernels while keeping step orchestration in Python.

    Call stock ``mjx.step(model, data)`` directly inside this context, without
    an outer JIT. Enter inside native cylinder registration so the currently
    registered adapter is wrapped too. Models, collision filters, arithmetic,
    and array device placement are unchanged; there is no host physics bridge.

    The caller must reject non-Euler integrators: this model-free context only
    stages Euler. Patches are process-global, so do not overlap this context
    with concurrent MJX execution or collision registration in other threads.
    Originals, including table identity, are restored even if setup fails.

    An optional positive mesh collision batch size changes only execution scheduling:
    all pairs and all contacts stay in original order. Slices and concatenations
    remain JAX device operations; there is no pair culling or host conversion.
    Optional BOX-MESH overrides map mesh asset IDs to positive chunk sizes, with
    the default retained for other assets and all other collision families.
    Optional conservative mesh broadphase retains all pair/contact slots but
    skips narrowphase for certified-separated BOX-MESH/MESH-MESH chunks.
    Optional group prechecks certify an entire mesh group before chunk dispatch.
    Only a boolean decision synchronizes to host; mixed groups retain the
    original chunk schedule. Timing observes the entire group without nested
    chunk timers, including the precheck and inactive-output assembly.
    Optional contact templates prepare original immutable model-only parameters
    lazily once. Only dynamic collision fields are rebuilt; model/layout/registry
    mutations reject. This host schedule is not an outer-JIT interface.
    Only inactive positive-distance/position/frame diagnostics may change.
    Cylinder broadphase additionally wraps CYLINDER-BOX/CYLINDER-MESH and
    chunks their outer calls using the explicit cylinder collision batch size.
    A positive candidate capacity opts only CYLINDER-BOX into one group JIT,
    replacing outer chunking with device packing and original-full-group overflow.
    Uncertain chunks retain the original adapter, including unsupported-query
    rejection; no penetrating-cylinder support is introduced by this option.
    Native box faces opt into the pinned native MuJoCo 3.2.7 convention
    for certified parallel or strictly contained active face contacts.
    The caller must reject other versions and nonzero inclusion parameters.
    The caller must validate static inclusion margins and bounding radii.
    The opt-in factored elliptic Hessian reassociates the original block products
    without creating the dense constraint-space matrix; solver settings stay fixed.
    Optional stable elliptic line search clamps negative-roundoff squared norms
    in the pinned point factory, retaining original solver limits and tolerance.
    Opt-in timing synchronizes each kernel and records cache growth. Its timings
    include compilation when a new signature appears and are diagnostic, not
    asynchronous production throughput measurements.
    """
    if collision_batch_size is not None and (
        not isinstance(collision_batch_size, int) or collision_batch_size < 1
    ):
        raise ValueError("Collision batch size must be a positive integer")
    if not isinstance(factored_elliptic_hessian, bool):
        raise ValueError("Factored elliptic Hessian selection must be boolean")
    if not isinstance(cache_contact_templates, bool):
        raise ValueError("Contact template caching selection must be boolean")
    if not isinstance(group_mesh_broadphase, bool):
        raise ValueError("Group mesh broadphase selection must be boolean")
    if group_mesh_broadphase and (mesh_broadphase_margin is None or collision_batch_size is None):
        raise ValueError("Group mesh broadphase requires a margin and explicit mesh batch size")
    if not isinstance(cylinder_broadphase, bool):
        raise ValueError("Cylinder broadphase selection must be boolean")
    if cylinder_candidate_capacity is not None:
        if type(cylinder_candidate_capacity) is not int or cylinder_candidate_capacity < 1:
            raise ValueError("Cylinder candidate capacity must be a positive integer")
        if (
            not cylinder_broadphase
            or mesh_broadphase_margin is None
            or cylinder_collision_batch_size is None
        ):
            raise ValueError(
                "Cylinder candidate packing requires cylinder broadphase, a margin and an explicit outer batch size"
            )
    if not isinstance(native_parallel_box_faces, bool):
        raise ValueError("Native parallel box faces selection must be boolean")
    if not isinstance(stable_elliptic_linesearch, bool):
        raise ValueError("Stable elliptic line search selection must be boolean")
    if stable_elliptic_linesearch:
        import mujoco

        if mujoco.__version__ != "3.2.7":
            raise ValueError("Stable elliptic line search requires pinned MuJoCo 3.2.7")
    if cylinder_collision_batch_size is not None and (
        type(cylinder_collision_batch_size) is not int or cylinder_collision_batch_size < 1
    ):
        raise ValueError("Cylinder outer collision batch size must be a positive integer")
    if cylinder_broadphase and (
        mesh_broadphase_margin is None or cylinder_collision_batch_size is None
    ):
        raise ValueError("Cylinder broadphase requires a margin and an explicit outer batch size")
    if mesh_broadphase_margin is not None and (
        isinstance(mesh_broadphase_margin, bool)
        or not isinstance(mesh_broadphase_margin, Real)
        or not math.isfinite(mesh_broadphase_margin)
        or mesh_broadphase_margin < 0
    ):
        raise ValueError("Mesh broadphase margin must be static finite nonnegative real")
    if box_mesh_batch_overrides is not None and (
        not isinstance(box_mesh_batch_overrides, dict)
        or any(
            type(mesh_id) is not int or mesh_id < 0 or type(size) is not int or size < 1
            for mesh_id, size in box_mesh_batch_overrides.items()
        )
        or (box_mesh_batch_overrides and collision_batch_size is None)
    ):
        raise ValueError(
            "BOX-MESH overrides require nonnegative integer mesh IDs, positive integer sizes and a default batch"
        )
    box_mesh_batch_overrides = dict(box_mesh_batch_overrides or {})
    originals = tuple((module, name, getattr(module, name)) for module, name in _STAGES)
    collision_table = collision_driver._COLLISION_FUNC
    original_collisions = dict(collision_table)
    original_gradient = solver._update_gradient
    original_point_factory = solver._LSPoint.__dict__["create"]
    original_collision = collision_driver.collision
    try:
        if cache_contact_templates:
            from .native_contact_templates import prepare_contact_templates

            prepared_contacts = None

            def cached_collision(model, data):
                nonlocal prepared_contacts
                if prepared_contacts is None:
                    prepared_contacts = prepare_contact_templates(model, data)
                return prepared_contacts(model, data)

            collision_driver.collision = cached_collision
        if stable_elliptic_linesearch:
            from .native_elliptic_linesearch import native_linesearch_point

            solver._LSPoint.create = classmethod(native_linesearch_point)
        if factored_elliptic_hessian:
            from .native_elliptic_hessian import native_update_gradient

            solver._update_gradient = native_update_gradient
        for module, name, function in originals:
            compiled = jax.jit(function)
            if stage_timing_observer is not None:
                compiled = _observe_timing(compiled, name, stage_timing_observer)
            setattr(module, name, compiled)
        for key, function in original_collisions.items():
            if native_parallel_box_faces and key == (GeomType.BOX, GeomType.BOX):
                from .native_contained_box_faces import wrap_native_contained_box_faces

                function = wrap_native_contained_box_faces(function)
            cylinder_family = key in (
                (GeomType.CYLINDER, GeomType.BOX),
                (GeomType.CYLINDER, GeomType.MESH),
            )
            candidate_queue = cylinder_candidate_capacity is not None and key == (
                GeomType.CYLINDER,
                GeomType.BOX,
            )
            if candidate_queue:
                from .native_candidate_packing import wrap_candidate_packing

                function = wrap_candidate_packing(
                    function, function.ncon, mesh_broadphase_margin, cylinder_candidate_capacity
                )
            elif mesh_broadphase_margin is not None and (
                key in ((GeomType.BOX, GeomType.MESH), (GeomType.MESH, GeomType.MESH))
                or (cylinder_broadphase and cylinder_family)
            ):
                from .native_mesh_broadphase import wrap_mesh_broadphase

                function = wrap_mesh_broadphase(function, function.ncon, mesh_broadphase_margin)
            compiled = jax.jit(function, static_argnums=(2,))
            group_precheck = group_mesh_broadphase and key in (
                (GeomType.BOX, GeomType.MESH),
                (GeomType.MESH, GeomType.MESH),
            )
            if stage_timing_observer is not None and not group_precheck:
                compiled = _observe_timing(compiled, f"collision:{key}", stage_timing_observer)
            compiled.ncon = function.ncon
            selected_batch = (
                None
                if candidate_queue
                else cylinder_collision_batch_size
                if cylinder_broadphase and cylinder_family
                else collision_batch_size
                if GeomType.MESH in key
                else None
            )
            scheduled = (
                compiled
                if selected_batch is None
                else _chunk_collision(compiled, selected_batch, box_mesh_batch_overrides)
            )
            if group_precheck:
                from .native_mesh_broadphase import wrap_group_broadphase

                scheduled = wrap_group_broadphase(scheduled, function.ncon, mesh_broadphase_margin)
                if stage_timing_observer is not None:
                    scheduled = _observe_timing(
                        scheduled, f"collision:{key}", stage_timing_observer
                    )
            collision_table[key] = scheduled
        yield
    finally:
        collision_driver.collision = original_collision
        solver._LSPoint.create = original_point_factory
        solver._update_gradient = original_gradient
        for module, name, function in originals:
            setattr(module, name, function)
        collision_table.clear()
        collision_table.update(original_collisions)
        collision_driver._COLLISION_FUNC = collision_table
