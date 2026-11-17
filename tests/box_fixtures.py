from typing import NamedTuple
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import collision_driver
import numpy as np


class Model(NamedTuple):
    geom_size: object


class Data(NamedTuple):
    geom_xpos: object
    geom_xmat: object


def collision_inputs(native):
    native_data = mujoco.MjData(native)
    mujoco.mj_forward(native, native_data)
    model = mjx.put_model(native)
    data = mjx.put_data(native, native_data)
    geometry = jp.array([[0, 1]], dtype=jp.int32)
    key = collision_driver.FunctionKey(
        types=(6, 6), data_ids=(-1, -1), condim=3, subgrid_size=(-1, -1)
    )
    return native_data, model, data, key, geometry


def point_set_error(actual, expected):
    distances = np.linalg.norm(actual[:, None] - expected[None, :], axis=-1)
    return max(float(distances.min(axis=0).max()), float(distances.min(axis=1).max()))
