"""回放正确性与适用范围检查。"""

import numpy as np


def arrays_bitwise_equal(actual, expected):
    """同时检查布局无关的元素字节、shape 和 dtype（包括有符号零）。"""
    actual, expected = np.asarray(actual), np.asarray(expected)
    return (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and actual.tobytes() == expected.tobytes()
    )


def check_adapter_distances(values):
    values = np.asarray(values)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(
            "Native-equivalent cylinder contacts unsupported; bounded separated stepping gate failed"
        )


def compare_state(model, qpos, qvel, simulation_time, reference, thresholds):
    import mujoco

    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    reference_qpos = np.asarray(reference["qpos"], dtype=np.float64)
    if qpos.shape != reference_qpos.shape or qvel.shape != reference["qvel"].shape:
        raise ValueError("State shapes differ from native reference")
    if (
        not np.isfinite(qpos).all()
        or not np.isfinite(qvel).all()
        or not np.isfinite(simulation_time)
    ):
        raise ValueError("Nonfinite candidate state")
    joint_mask = np.ones(qpos.size, dtype=bool)
    orientation_errors = []
    for joint_type, address in zip(model.jnt_type, model.jnt_qposadr):
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            quaternion_address = int(address) + 3
        elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            quaternion_address = int(address)
        else:
            continue
        quaternion_slice = slice(quaternion_address, quaternion_address + 4)
        joint_mask[quaternion_slice] = False
        actual_quaternion = qpos[quaternion_slice]
        reference_quaternion = reference_qpos[quaternion_slice]
        actual_norm = np.linalg.norm(actual_quaternion)
        reference_norm = np.linalg.norm(reference_quaternion)
        if actual_norm == 0 or reference_norm == 0:
            raise ValueError("Invalid zero-norm quaternion")
        cosine = abs(np.dot(actual_quaternion / actual_norm, reference_quaternion / reference_norm))
        orientation_errors.append(float(2 * np.arccos(np.clip(cosine, 0, 1))))
    metrics = {
        "qpos_nonquaternion_max_error": float(
            np.max(np.abs(qpos[joint_mask] - reference_qpos[joint_mask]), initial=0)
        ),
        "orientation_max_error_radians": max(orientation_errors, default=0),
        "qvel_max_error": float(np.max(np.abs(qvel - reference["qvel"]), initial=0)),
        "time_error": abs(float(simulation_time) - reference["time"]),
    }
    metrics["within_tolerance"] = (
        metrics["qpos_nonquaternion_max_error"] <= thresholds["qpos_nonquaternion"]
        and metrics["orientation_max_error_radians"] <= thresholds["orientation_radians"]
        and metrics["qvel_max_error"] <= thresholds["qvel"]
        and metrics["time_error"] <= thresholds["time"]
    )
    return metrics


def validate_native_parallel_box_model(model, version):
    if version != "3.2.7":
        raise ValueError("Native parallel box convention requires pinned MuJoCo 3.2.7")
    mesh_broadphase_margin(model)
    if any(
        np.any(np.asarray(getattr(model, name)) != 0)
        for name in ("geom_margin", "geom_gap", "pair_margin", "pair_gap")
    ):
        raise ValueError("Native parallel box convention requires zero inclusion parameters")


def mesh_broadphase_margin(model):
    import mujoco

    if model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE):
        raise ValueError("Conservative mesh broadphase does not support contact override")
    fields = [
        np.asarray(getattr(model, name), dtype=np.float64)
        for name in ("geom_margin", "geom_gap", "pair_margin", "pair_gap")
    ]
    if any(not np.isfinite(value).all() for value in fields):
        raise ValueError("Mesh broadphase inclusion parameters must be finite")
    geom_margin, geom_gap, pair_margin, pair_gap = fields
    if geom_margin.shape != geom_gap.shape or pair_margin.shape != pair_gap.shape:
        raise ValueError("Mesh broadphase margin/gap shape mismatch")
    bound = max(
        0.0,
        float(np.max(geom_margin, initial=0.0) - np.min(geom_gap, initial=0.0)),
        float(np.max(pair_margin - pair_gap, initial=0.0)),
    )
    if not np.isfinite(bound):
        raise ValueError("Mesh broadphase inclusion bound overflowed")
    return bound
