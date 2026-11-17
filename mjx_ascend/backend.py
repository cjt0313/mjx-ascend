"""显式 NPU 选择，禁止静默回退。"""

import os
import ctypes
from pathlib import Path
from contextlib import contextmanager, ExitStack
from unittest.mock import patch


def select_device(devices, backend, device_index):
    matches = [device for device in devices if device.platform.lower() == backend.lower()]
    if not matches:
        raise RuntimeError(f"No {backend} device found; CPU fallback is forbidden")
    if not 0 <= device_index < len(matches):
        raise RuntimeError(
            f"Requested device index {device_index}; only {len(matches)} {backend} devices found"
        )
    return matches[device_index]


def configure_backend(backend, plugin_path):
    if backend == "npu":
        plugin_path = Path(plugin_path).expanduser().resolve()
        if not plugin_path.is_file():
            raise FileNotFoundError(f"PJRT plugin is missing: {plugin_path}")
        os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"npu:{plugin_path}"
        os.environ["SOC_VERSION"] = soc_name()
    else:
        os.environ.pop("PJRT_NAMES_AND_LIBRARY_PATHS", None)
    os.environ["JAX_PLATFORMS"] = backend
    os.environ["JAX_ENABLE_X64"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "4")


def soc_name():
    library = ctypes.CDLL("libascendcl.so")
    function = library.aclrtGetSocName
    function.restype = ctypes.c_char_p
    value = function()
    if value is None:
        raise RuntimeError("CANN did not report a SoC name")
    return value.decode()


@contextmanager
def native_entrypoint_guard(calls):
    import mujoco

    names = sorted(
        name
        for name in dir(mujoco)
        if (name.startswith("mj_") or name in ("MjModel", "MjData"))
        and callable(getattr(mujoco, name))
    )

    def forbidden(name):
        def reject(*args, **kwargs):
            calls[name] += 1
            raise RuntimeError(f"Native entrypoint forbidden during NPU candidate: {name}")

        return reject

    with ExitStack() as stack:
        for name in names:
            calls[name] = 0
            stack.enter_context(patch.object(mujoco, name, side_effect=forbidden(name)))
        yield
