"""Bounded reassociation of the pinned MJX 3.2.7 Newton/elliptic Hessian.

This experiment changes arithmetic scheduling only. It neither registers nor
patches solver functions; the original LIBERO staged runner owns registration.
Floating-point reassociation still requires the original-scene CPU state gate.
"""

import jax
import jax.numpy as jp
from mujoco.mjx._src import solver, support
from mujoco.mjx._src.types import ConeType, SolverType
import numpy as np


_ORIGINAL_UPDATE_GRADIENT = solver._update_gradient


def factored_hessian(
    jacobian, diagonal_weights, cone_hessians, static_np_addresses, static_np_dimensions
):
    """Return J.T diag(weights) J plus every original J_cone.T H_cone J_cone.

    Addresses and dimensions must be one-dimensional integer NumPy arrays,
    closed over by a JIT caller, ordered like the padded (ncone, 6, 6) Hessians.
    Only elliptic dimensions 3, 4 and 6 belong here; scalar constraints remain
    in diagonal_weights. Out-of-bounds blocks are rejected, never clipped.
    Overlap is allowed and summed, matching the stock additive block updates.
    A scalar Hessian is accepted only when no cone blocks exist.

    Runtime values stay on device. Storage is linear in constraint/contact
    count for fixed nv: weighted rows and grouped (ncontact, condim, nv)
    products, never a constraint square or per-contact (nv, nv) product.
    """
    addresses, dimensions = static_np_addresses, static_np_dimensions
    for name, metadata in (("addresses", addresses), ("dimensions", dimensions)):
        if not isinstance(metadata, np.ndarray):
            raise TypeError(f"{name} must be a static NumPy array")
        if metadata.ndim != 1 or not np.issubdtype(metadata.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array")
    if addresses.shape != dimensions.shape:
        raise ValueError("addresses and dimensions must have identical shapes")
    if not np.isin(dimensions, (3, 4, 6)).all():
        raise ValueError("elliptic cone dimensions must be 3, 4 or 6")

    jacobian = jp.asarray(jacobian)
    diagonal_weights = jp.asarray(diagonal_weights)
    cone_hessians = jp.asarray(cone_hessians)
    if jacobian.ndim != 2:
        raise ValueError("jacobian must have shape (nefc, nv)")
    nefc, nv = jacobian.shape
    if diagonal_weights.shape != (nefc,):
        raise ValueError("diagonal_weights must have shape (nefc,)")
    if np.any(addresses < 0) or np.any(dimensions > nefc):
        raise ValueError("cone blocks must lie within jacobian rows")
    if np.any(addresses > nefc - dimensions.astype(np.int64)):
        raise ValueError("cone blocks must lie within jacobian rows")
    if cone_hessians.shape != (addresses.size, 6, 6):
        if addresses.size or cone_hessians.ndim != 0:
            raise ValueError("cone_hessians must have shape (ncone, 6, 6)")

    hessian = (jacobian.T * diagonal_weights) @ jacobian
    for dimension in (3, 4, 6):
        group = np.flatnonzero(dimensions == dimension)
        if not group.size:
            continue
        indices = addresses[group].astype(np.int64)[:, None] + np.arange(dimension)
        cone_jacobian = jacobian[indices]
        cone_hessian = cone_hessians[group]
        valid = jp.arange(6) < dimension
        cone_hessian = jp.where(valid[None, :, None] & valid[None, None, :], cone_hessian, 0)
        if dimension < 6:
            cone_jacobian = jp.concatenate(
                (cone_jacobian, jp.zeros((group.size, 6 - dimension, nv), dtype=jacobian.dtype)),
                axis=1,
            )
        weighted = cone_hessian @ cone_jacobian
        hessian += cone_jacobian.reshape(-1, nv).T @ weighted.reshape(-1, nv)
    return hessian


def native_update_gradient(model, data, context):
    """Stock signature/context, factoring only the Newton/elliptic branch.

    The original function handle is captured before parent-owned registration,
    so CG and non-elliptic fallback cannot recurse through a patched module.
    Mass assembly, gradient, Cholesky and solve remain stock MJX operations.
    """
    if model.opt.solver != SolverType.NEWTON or model.opt.cone != ConeType.ELLIPTIC:
        return _ORIGINAL_UPDATE_GRADIENT(model, data, context)

    elliptic = data.contact.dim > 1
    hessian = factored_hessian(
        data.efc_J,
        data.efc_D * context.active,
        context.h,
        data.contact.efc_address[elliptic],
        data.contact.dim[elliptic],
    )
    grad = context.Ma - data.qfrc_smooth - context.qfrc_constraint
    hessian = support.full_m(model, data) + hessian
    factor = jax.scipy.linalg.cho_factor(hessian)
    mgrad = jax.scipy.linalg.cho_solve(factor, grad)
    return context.replace(grad=grad, Mgrad=mgrad)
