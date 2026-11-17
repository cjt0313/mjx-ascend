"""执行固定配置的 Ascend 回放，并与原生 CPU 逐步对照。"""

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .backend import configure_backend, native_entrypoint_guard, select_device
from .validation import (
    arrays_bitwise_equal,
    check_adapter_distances,
    compare_state,
    mesh_broadphase_margin,
    validate_native_parallel_box_model,
)

THRESHOLDS = {"qpos_nonquaternion": 1e-5, "qvel": 1e-4, "time": 1e-6, "orientation_radians": 1e-4}


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--stage-timing", action="store_true")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.steps > 75 or args.device < 0:
        parser.error("steps must be 1..75 and device nonnegative")
    if not args.plugin.is_file():
        parser.error("An existing PJRT plugin is required")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "verdict": "running",
        "states": [],
        "thresholds": THRESHOLDS,
        "scope": "Frozen controls, Euler, separated cylinder queries only",
        "device_index": args.device,
        "steps": args.steps,
    }
    path = args.output / "report.json"
    write_report(path, report)
    try:
        configure_backend("npu", args.plugin)
        import jax
        import mujoco
        from mujoco import mjx
        from .capture import config_hash, file_hash, load_scene, model_config, restore_state
        from .libero_native_adapter import register_cylinder_mesh
        from .staged_libero_step import staged_kernels

        device = select_device(jax.devices("npu"), "npu", args.device)
        if jax.config.jax_enable_x64 or jax.default_backend() != "npu":
            raise RuntimeError("Explicit FP32 NPU execution required")
        model, arrays, manifest = load_scene(args.scene, args.output / "cache")
        if model.opt.integrator != int(mujoco.mjtIntegrator.mjINT_EULER):
            raise ValueError("Only Euler integration supported")
        validate_native_parallel_box_model(model, mujoco.__version__)
        if args.steps > len(arrays["controls"]):
            raise ValueError("Requested horizon exceeds captured controls")
        report.update(
            plugin_sha256=file_hash(args.plugin),
            scene=manifest,
            device=str(device),
            stage_timing=args.stage_timing,
        )
        baseline = None
        if args.reference:
            if file_hash(args.reference) != manifest.get("reference_sha256"):
                raise ValueError("Reference checksum mismatch")
            with np.load(args.reference, allow_pickle=False) as archive:
                baseline = {name: archive[name] for name in ("qpos", "qvel", "time")}
            if any(len(value) < args.steps for value in baseline.values()):
                raise ValueError("Reference horizon too short")
        reference = []
        native = restore_state(model, arrays, manifest)
        for control in arrays["controls"][: args.steps]:
            native.ctrl[:] = control
            mujoco.mj_step(model, native)
            reference.append(
                {
                    "qpos": native.qpos.copy(),
                    "qvel": native.qvel.copy(),
                    "time": float(native.time),
                    "contacts": int(native.ncon),
                }
            )
        native = restore_state(model, arrays, manifest)
        calls, timings = {}, []
        with register_cylinder_mesh(batch_size=8):
            converted = mjx.put_model(model, device=device)
            data = mjx.put_data(model, native, device=device)
            with (
                native_entrypoint_guard(calls),
                staged_kernels(
                    collision_batch_size=16,
                    box_mesh_batch_overrides={61: 128},
                    mesh_broadphase_margin=mesh_broadphase_margin(model),
                    cylinder_broadphase=True,
                    cylinder_collision_batch_size=8,
                    cylinder_candidate_capacity=8,
                    native_parallel_box_faces=True,
                    stable_elliptic_linesearch=True,
                    cache_contact_templates=True,
                    group_mesh_broadphase=True,
                    factored_elliptic_hessian=True,
                    stage_timing_observer=timings.append if args.stage_timing else None,
                ),
            ):
                for index, control in enumerate(arrays["controls"][: args.steps]):
                    data = data.replace(ctrl=jax.device_put(control, device))
                    started = time.perf_counter()
                    data = mjx.step(converted, data)
                    for value in jax.tree_util.tree_leaves(data):
                        if isinstance(value, jax.Array):
                            value.block_until_ready()
                            if value.devices() != {device}:
                                raise RuntimeError("Physics output left selected NPU")
                    elapsed = time.perf_counter() - started
                    qpos, qvel = np.asarray(data.qpos), np.asarray(data.qvel)
                    distances = np.asarray(data.contact.dist)
                    types = model.geom_type[np.asarray(data.contact.geom)]
                    cylinder = np.any(types == int(mujoco.mjtGeom.mjGEOM_CYLINDER), axis=1)
                    custom = cylinder & np.any(
                        np.isin(
                            types, [int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_MESH)]
                        ),
                        axis=1,
                    )
                    check_adapter_distances(distances[custom])
                    if not np.isfinite(distances).all():
                        raise ValueError("Nonfinite contact distances")
                    consistency = compare_state(
                        model, qpos, qvel, float(data.time), reference[index], THRESHOLDS
                    )
                    active = int(np.sum(distances < np.asarray(data.contact.includemargin)))
                    parity = active == reference[index]["contacts"]
                    bitwise = baseline is None or all(
                        arrays_bitwise_equal(actual, baseline[name][index])
                        for name, actual in (
                            ("qpos", qpos),
                            ("qvel", qvel),
                            ("time", np.asarray(data.time)),
                        )
                    )
                    report["states"].append(
                        {
                            "step": index + 1,
                            "qpos": qpos.tolist(),
                            "qvel": qvel.tolist(),
                            "time": float(data.time),
                            "seconds_including_compile_sync": elapsed,
                            "cpu_consistency": consistency,
                            "active_contact_slots": active,
                            "reference_native_contacts": reference[index]["contacts"],
                            "contact_count_matches": parity,
                            "reference_bitwise_matches": bitwise if baseline else None,
                            "native_entrypoint_calls": sum(calls.values()),
                        }
                    )
                    write_report(path, report)
                    if not consistency["within_tolerance"] or not parity or not bitwise:
                        raise ValueError(
                            f"Step {index + 1} failed state/contact/reference validation"
                        )
        if config_hash(model_config(model)) != manifest["model_config_hash"]:
            raise RuntimeError("Model changed during replay")
        elapsed = [state["seconds_including_compile_sync"] for state in report["states"]]
        report.update(
            verdict="passed",
            native_entrypoint_calls=sum(calls.values()),
            timing={
                "first_step_seconds": elapsed[0],
                "total_seconds": sum(elapsed),
                "warm_mean_seconds": float(np.mean(elapsed[1:])) if len(elapsed) > 1 else None,
            },
        )
        if args.stage_timing:
            write_report(args.output / "stage-timings.json", timings)
    except Exception as error:
        report.update(
            verdict="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    write_report(path, report)
    print(json.dumps({"verdict": report["verdict"], "report": str(path)}))
    return 0 if report["verdict"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
