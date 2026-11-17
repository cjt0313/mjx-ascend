"""Opt-in, model-bound MJX 3.2.7 collision scheduling; no global patches.

Prepare outside timed kernels, then call the returned executor with the exact
same model and live Data. Model JAX leaves are immutable and identity-bound;
MJX's pytree auxiliary bytes snapshot mutable NumPy model metadata. Contact
layout bytes/shapes are guarded on every call. Replacement or mutation requires
fresh preparation. This host scheduler is not a general nested-JIT interface.

Every original registered collider receives its original unexpanded pairs.
Only dist/pos/frame are assembled per step; model-only parameter and geom
arrays stay in the immutable, device-resident Contact template. No collision
state, geometry bridge, capacity reduction, or speedup claim is cached here.
"""

from dataclasses import dataclass
from importlib import metadata

import jax
import jax.numpy as jp
import mujoco
from mujoco.mjx._src import collision_driver
from mujoco.mjx._src.types import Contact, Data, Model
import numpy as np


def _check_version():
    if mujoco.__version__ != "3.2.7" or metadata.version("mujoco-mjx") != "3.2.7":
        raise ValueError("Contact templates require MuJoCo and MJX version 3.2.7")


def _array_layout(value):
    if not isinstance(value, jax.Array):
        raise ValueError("Expected concrete device arrays; nested JIT/tracing is unsupported")
    return value.shape, value.dtype, frozenset(value.devices())


def _model_signature(model):
    leaves, structure = jax.tree_util.tree_flatten(model)
    return structure, tuple((id(leaf), _array_layout(leaf)) for leaf in leaves)


def _static_bytes(value):
    if not isinstance(value, np.ndarray):
        raise ValueError("Contact layout auxiliary fields must be static NumPy arrays")
    return value.shape, value.dtype, value.tobytes()


def _layout_signature(data):
    contact_leaves = jax.tree_util.tree_leaves(data.contact)
    return (
        data.ncon,
        _static_bytes(data.contact.dim),
        _static_bytes(data.contact.efc_address),
        tuple(_array_layout(leaf) for leaf in contact_leaves),
        _array_layout(data.geom_xpos),
        _array_layout(data.geom_xmat),
    )


def _immutable_aux(value):
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class PreparedContactTemplates:
    """Standalone host executor with immutable model-only Contact metadata.

    Registry function identity and ncon must remain unchanged for selected
    groups. Static model NumPy arrays may be mutable upstream, but changes are
    rejected before any collider runs. Dynamic Data values are never snapshotted.
    Guards inspect host auxiliary bytes and device shape/residency, not device
    geometry or parameter values. Concurrent model/registry mutation is outside
    this contract; callers must keep them fixed throughout preparation/execution.
    """

    template: Contact
    _model: Model
    _model_state: tuple
    _layout: tuple
    _groups: tuple
    _order: tuple

    def __call__(self, model: Model, data: Data) -> Data:
        _check_version()
        if model is not self._model:
            raise ValueError("Foreign/replaced model requires new contact preparation")
        if _model_signature(model) != self._model_state:
            raise ValueError("Bound model metadata/leaves mutated; prepare contacts again")
        if _layout_signature(data) != self._layout:
            raise ValueError("Contact/geometry layout changed; prepare contacts again")
        for key, function, ncon, _ in self._groups:
            current_ncon = getattr(function, "ncon", None)
            if (
                collision_driver._COLLISION_FUNC.get(key.types) is not function
                or type(current_ncon) is not int
                or current_ncon != ncon
            ):
                raise ValueError("Selected collider registry identity/ncon changed; reprepare")
        if data.ncon == 0:
            return data

        dynamic = []
        for key, function, ncon, geom in self._groups:
            values = function(model, data, key, geom)
            count = geom.shape[0] * ncon
            expected_shapes = ((count,), (count, 3), (count, 3, 3))
            if len(values) != 3 or tuple(value.shape for value in values) != expected_shapes:
                raise ValueError("Collider output does not match prepared contact layout/ncon")
            if any(value.devices() != geom.devices() for value in values):
                raise ValueError("Collider output left prepared geometry device residency")
            dynamic.append(values)
        dist, pos, frame = (
            jp.concatenate([dynamic[index][field] for index in self._order]) for field in range(3)
        )
        return data.replace(contact=self.template.replace(dist=dist, pos=pos, frame=frame))


def prepare_contact_templates(model: Model, data: Data) -> PreparedContactTemplates:
    """Prepare original full-capacity parameter mixing/order once, without physics.

    Nonnegative max_geom_pairs/max_contact_points numeric caps are explicitly
    unsupported, even if they would not truncate this particular model. Empty
    layouts are supported and return Data unchanged at execution. Disabled
    contacts follow stock's empty-layout early return.
    """
    _check_version()
    if not isinstance(model, Model) or not isinstance(data, Data):
        raise ValueError("Preparation requires original MJX Model and Data")
    for name in ("max_geom_pairs", "max_contact_points"):
        if collision_driver._numeric(model, name) >= 0:
            raise ValueError(f"Unsupported numeric cap: {name}")

    model_state = _model_signature(model)
    layout = _layout_signature(data)
    device_set = data.geom_xpos.devices()
    if (
        data.geom_xpos.shape != (model.ngeom, 3)
        or data.geom_xmat.shape != (model.ngeom, 3, 3)
        or data.geom_xmat.devices() != device_set
    ):
        raise ValueError("Invalid initial geometry shape/residency layout")
    if any(leaf.devices() != device_set for leaf in jax.tree_util.tree_leaves(model)):
        raise ValueError("Bound model and geometry require matching device residency")
    expected_dim = collision_driver.make_condim(model)
    if (
        data.ncon != len(expected_dim)
        or data.contact.dim.shape != (data.ncon,)
        or not np.array_equal(data.contact.dim, expected_dim)
        or data.contact.efc_address.shape != (data.ncon,)
        or (data.ncon and not np.issubdtype(data.contact.dim.dtype, np.integer))
        or not np.issubdtype(data.contact.efc_address.dtype, np.integer)
    ):
        raise ValueError("Invalid initial contact layout for bound model/registry")
    for leaf in jax.tree_util.tree_leaves(data.contact):
        if not leaf.shape or leaf.shape[0] != data.ncon:
            raise ValueError("Invalid initial contact array shape/layout")
        if leaf.devices() != data.geom_xpos.devices():
            raise ValueError("Contact arrays and geometry require matching device residency")
    expected_shapes = {
        "dist": (data.ncon,),
        "pos": (data.ncon, 3),
        "frame": (data.ncon, 3, 3),
        "includemargin": (data.ncon,),
        "friction": (data.ncon, 5),
        "solref": (data.ncon, mujoco.mjNREF),
        "solreffriction": (data.ncon, mujoco.mjNREF),
        "solimp": (data.ncon, mujoco.mjNIMP),
        "geom1": (data.ncon,),
        "geom2": (data.ncon,),
        "geom": (data.ncon, 2),
    }
    if any(getattr(data.contact, name).shape != shape for name, shape in expected_shapes.items()):
        raise ValueError("Invalid initial contact array shape/layout")

    groups = collision_driver._contact_groups(model, data) if data.ncon else {}
    scheduled, repeated = [], []
    for key, contact in groups.items():
        function = collision_driver._COLLISION_FUNC[key.types]
        ncon = function.ncon
        if type(ncon) is not int or ncon < 1:
            raise ValueError("Collider ncon must be a positive static integer")
        scheduled.append((key, function, ncon, contact.geom))
        if ncon > 1:
            contact = jax.tree_util.tree_map(
                lambda value, repeat=ncon: jp.repeat(value, repeat, axis=0), contact
            )
        repeated.append(contact)

    order = tuple(sorted(range(len(scheduled)), key=lambda index: scheduled[index][0].condim))
    if order:
        template = jax.tree_util.tree_map(
            lambda *values: jp.concatenate(values), *(repeated[index] for index in order)
        )
        if template.geom.shape[0] != data.ncon:
            raise ValueError("Prepared collider count differs from contact layout")
    else:
        template = data.contact.replace(dist=None, pos=None, frame=None)
    template = template.replace(
        dim=_immutable_aux(data.contact.dim), efc_address=_immutable_aux(data.contact.efc_address)
    )
    return PreparedContactTemplates(template, model, model_state, layout, tuple(scheduled), order)
