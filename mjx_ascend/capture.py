"""独立场景的加载、完整性检查与状态恢复。"""

import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import shutil
import numpy as np
import mujoco

MODEL_PREFIXES = ("geom_", "mesh_", "body_", "pair_", "exclude_")


def array_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_config(model):
    fields = {
        name: getattr(model, name).copy()
        for name in dir(model)
        if name.startswith(MODEL_PREFIXES) and isinstance(getattr(model, name), np.ndarray)
    }
    fields["qpos0"] = model.qpos0.copy()
    options = {
        name: getattr(model.opt, name)
        for name in dir(model.opt)
        if not name.startswith("_")
        and isinstance(getattr(model.opt, name), (int, float, np.ndarray))
    }
    return {"fields": fields, "options": options}


def config_hash(config):
    def encode(value):
        if isinstance(value, np.ndarray):
            return {
                "sha256": array_hash(value),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        return value

    return hashlib.sha256(
        json.dumps(encode(config), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def load_scene(directory, cache):
    directory, cache = Path(directory), Path(cache)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["mujoco_version"] != mujoco.__version__ or mujoco.__version__ != "3.2.7":
        raise ValueError("Scene requires MuJoCo 3.2.7")
    if file_hash(directory / "replay.npz") != manifest["replay_sha256"]:
        raise ValueError("Replay checksum mismatch")
    if (
        "model_gzip_sha256" in manifest
        and file_hash(directory / "model.mjb.gz") != manifest["model_gzip_sha256"]
    ):
        raise ValueError("Compressed model checksum mismatch")
    expected = manifest["model_sha256"]
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("Invalid model checksum")
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{expected}.mjb"
    if not target.exists() or file_hash(target) != expected:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache, suffix=".mjb", delete=False) as output:
                temporary = Path(output.name)
                with gzip.open(directory / "model.mjb.gz", "rb") as source:
                    shutil.copyfileobj(source, output)
            if file_hash(temporary) != expected:
                raise ValueError("Model checksum mismatch")
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    model = mujoco.MjModel.from_binary_path(str(target))
    if config_hash(model_config(model)) != manifest["model_config_hash"]:
        raise ValueError("Model configuration mismatch")
    if model.nflex or model.nplugin:
        raise ValueError("Flexible/plugin models unsupported")
    for name in dir(mujoco):
        if name.startswith("get_mjcb_") and getattr(mujoco, name)() is not None:
            raise ValueError("Native callbacks must be absent")
    if manifest["state_specification"] != int(mujoco.mjtState.mjSTATE_INTEGRATION):
        raise ValueError("Full integration state required")
    with np.load(directory / "replay.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    controls = arrays["controls"]
    state = arrays["initial_state"]
    if controls.dtype != np.float32 or controls.ndim != 2 or controls.shape[1] != model.nu:
        raise ValueError("Controls require FP32 [steps, nu]")
    if state.dtype != np.float64 or state.shape != (
        mujoco.mj_stateSize(model, manifest["state_specification"]),
    ):
        raise ValueError("Invalid FP64 integration state")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite replay data")
    restore_state(model, arrays, manifest)
    return model, arrays, manifest


def restore_state(model, arrays, manifest):
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, arrays["initial_state"], manifest["state_specification"])
    for name in ("qpos", "qvel", "time"):
        if not np.array_equal(getattr(data, name), arrays[f"initial_{name}"]):
            raise ValueError(f"Initial {name} mismatch")
    return data
