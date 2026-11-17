"""Pinned MJX 3.2.7 point evaluation with nonnegative tangent squared norms.

Only the expanded squared norm is clamped against negative FP32 roundoff.
All original zones, costs, derivatives and solver settings remain unchanged.
Registration belongs to the isolated staged context, not this module.
"""

import jax
import jax.numpy as jp
import mujoco
from mujoco.mjx._src import solver
from mujoco.mjx._src.types import ConeType, DisableBit


_ORIGINAL_CREATE = solver._LSPoint.create


def native_linesearch_point(
    class_type,
    model,
    data,
    context,
    alpha,
    jacobian_direction,
    quadratic,
    quadratic_gauss,
    tangent_square,
    normal_direction,
    tangent_dot,
    direction_square,
):
    """Original point factory with one squared-norm domain correction."""
    if model.opt.cone != ConeType.ELLIPTIC:
        return _ORIGINAL_CREATE(
            model,
            data,
            context,
            alpha,
            jacobian_direction,
            quadratic,
            quadratic_gauss,
            tangent_square,
            normal_direction,
            tangent_dot,
            direction_square,
        )

    cost, derivative_first, derivative_second = 0.0, 0.0, 0.0
    quadratic_total = quadratic_gauss
    residual = context.Jaref + alpha * jacobian_direction
    active = (residual < 0).at[: data.ne + data.nf].set(True)
    if (model.dof_hasfrictionloss.any() or model.tendon_hasfrictionloss.any()) and not (
        model.opt.disableflags & DisableBit.FRICTIONLOSS
    ):
        friction_loss = data.efc_frictionloss
        inverse_weight = 1.0 / (data.efc_D + (data.efc_D == 0.0) * mujoco.mjMINVAL)
        threshold = inverse_weight * friction_loss
        zero = jp.zeros_like(friction_loss)
        linear_negative = (residual <= -threshold)[:, None]
        linear_positive = (residual >= threshold)[:, None]
        friction_quadratic = (
            linear_negative
            * jp.array(
                [
                    friction_loss * (-0.5 * threshold - context.Jaref),
                    -friction_loss * jacobian_direction,
                    zero,
                ]
            ).T
        )
        friction_quadratic += (
            linear_positive
            * jp.array(
                [
                    friction_loss * (-0.5 * threshold + context.Jaref),
                    friction_loss * jacobian_direction,
                    zero,
                ]
            ).T
        )
        quadratic = jp.where(
            (linear_negative | linear_positive) & (friction_loss[:, None] > 0),
            friction_quadratic,
            quadratic,
        )

    coefficient, initial_normal = context.fri[:, 0], context.u[:, 0]
    normal = initial_normal + alpha * normal_direction
    norm_squared = jp.maximum(
        tangent_square + alpha * (2 * tangent_dot + alpha * direction_square), 0
    )
    tangent = jp.sqrt(norm_squared)
    bottom_zone = ((norm_squared <= 0) & (normal < 0)) | (
        (norm_squared > 0) & ((coefficient * normal + tangent) <= 0)
    )
    middle_zone = (
        (norm_squared > 0)
        & (normal < (coefficient * tangent))
        & ((coefficient * normal + tangent) > 0)
    )
    scalar_addresses = data.contact.efc_address[data.contact.dim == 1]
    noncontact_rows = data.ne + data.nf + data.nl
    active = (
        active.at[noncontact_rows:].set(False).at[scalar_addresses].set(active[scalar_addresses])
    )
    quadratic_active = jax.vmap(jp.multiply)(quadratic, active)
    quadratic_total += jp.sum(quadratic_active, axis=0)
    cone_addresses = data.contact.efc_address[data.contact.dim > 1]
    quadratic_cone = jax.vmap(jp.multiply)(quadratic[cone_addresses], bottom_zone)
    quadratic_total += jp.sum(quadratic_cone, axis=0)

    tangent += (tangent == 0) * mujoco.mjMINVAL
    norm_squared += (norm_squared == 0) * mujoco.mjMINVAL
    tangent_first = (tangent_dot + alpha * direction_square) / tangent
    tangent_second = (
        direction_square / tangent
        - (tangent_dot + alpha * direction_square) * tangent_first / norm_squared
    )
    weight = context.dm * middle_zone
    normal_minus_tangent = normal - coefficient * tangent
    cost = 0.5 * jp.sum(weight * jp.square(normal_minus_tangent))
    derivative_first = jp.sum(
        weight * normal_minus_tangent * (normal_direction - coefficient * tangent_first)
    )
    derivative_second = jp.sum(
        weight
        * (
            jp.square(normal_direction - coefficient * tangent_first)
            - normal_minus_tangent * coefficient * tangent_second
        )
    )
    alpha_squared = alpha * alpha
    cost += alpha_squared * quadratic_total[2] + alpha * quadratic_total[1] + quadratic_total[0]
    derivative_first += 2 * alpha * quadratic_total[2] + quadratic_total[1]
    derivative_second += 2 * quadratic_total[2] + (quadratic_total[2] == 0) * mujoco.mjMINVAL
    return class_type(alpha=alpha, cost=cost, deriv_0=derivative_first, deriv_1=derivative_second)
