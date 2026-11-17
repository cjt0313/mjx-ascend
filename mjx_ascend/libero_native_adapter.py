"""仅接入严格分离圆柱/凸体结果的 JAX 适配器。"""

from contextlib import contextmanager
from functools import partial

import jax
import jax.numpy as jp
from mujoco.mjx._src import collision_driver, math, types

from .native_cylinder_mesh import collide_cylinder_mesh


def separated_contact(position, rotation, size, vertices):
    result = collide_cylinder_mesh(position, rotation, size, vertices)
    accepted = result.valid & (result.status == 0) & (result.distance > 0)
    distance = jp.where(accepted, result.distance, jp.nan)
    contact = jp.where(accepted, result.contact, jp.nan)
    frame = jp.where(accepted, math.make_frame(result.normal), jp.nan)
    return distance, contact, frame


def cylinder_mesh(model, data, key, geoms, *, batch_size=None):
    if key.types[1] == types.GeomType.MESH:
        mesh_id = key.data_ids[1]
        first = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        local_vertices = jp.asarray(model.mesh_vert[first : first + count], dtype=jp.float32)

    def collide_pair(pair):
        cylinder, mesh = pair
        if key.types[1] == types.GeomType.BOX:
            signs = jp.array(
                [
                    [first, second, third]
                    for first in (-1.0, 1.0)
                    for second in (-1.0, 1.0)
                    for third in (-1.0, 1.0)
                ]
            )
            vertices = (signs * model.geom_size[mesh]) @ data.geom_xmat[mesh].T + data.geom_xpos[
                mesh
            ]
        else:
            vertices = local_vertices @ data.geom_xmat[mesh].T + data.geom_xpos[mesh]
        return separated_contact(
            data.geom_xpos[cylinder],
            data.geom_xmat[cylinder],
            model.geom_size[cylinder][:2],
            vertices,
        )

    if batch_size is None or batch_size == 1:
        return jax.lax.map(collide_pair, geoms)
    return jax.lax.map(collide_pair, geoms, batch_size=batch_size)


cylinder_mesh.ncon = 1


@contextmanager
def register_cylinder_mesh(batch_size=None):
    """Opt into bounded vmap scheduling; None/1 retain the serial collider."""
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("Cylinder batch size must be a positive integer or None")
    keys = [
        (types.GeomType.CYLINDER, types.GeomType.MESH),
        (types.GeomType.CYLINDER, types.GeomType.BOX),
    ]
    previous = {key: collision_driver._COLLISION_FUNC.get(key) for key in keys}
    function = cylinder_mesh
    if batch_size is not None and batch_size > 1:
        function = partial(cylinder_mesh, batch_size=batch_size)
        function.ncon = cylinder_mesh.ncon
    for key in keys:
        collision_driver._COLLISION_FUNC[key] = function
    try:
        yield
    finally:
        for key in keys:
            if previous[key] is None:
                collision_driver._COLLISION_FUNC.pop(key, None)
            else:
                collision_driver._COLLISION_FUNC[key] = previous[key]
