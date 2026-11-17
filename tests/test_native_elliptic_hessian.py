"""CPU parity and allocation-shape gates for the pinned elliptic Hessian."""

import unittest
from unittest import mock

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import solver
import numpy as np


from mjx_ascend.native_elliptic_hessian import (
    factored_hessian,
    native_update_gradient,
)


def dense_reference(jacobian, weights, blocks, addresses, dimensions):
    constraint_matrix = np.diag(np.asarray(weights))
    for block, address, dimension in zip(blocks, addresses, dimensions):
        constraint_matrix[address : address + dimension, address : address + dimension] += block[
            :dimension, :dimension
        ]
    return jacobian.T @ constraint_matrix @ jacobian


def random_inputs(addresses, dimensions, nefc=31, nv=7):
    random = np.random.default_rng(7327)
    jacobian = random.normal(size=(nefc, nv)).astype(np.float32)
    weights = random.uniform(0.1, 2, size=nefc).astype(np.float32)
    factors = random.normal(size=(len(addresses), 6, 6)).astype(np.float32)
    blocks = factors @ factors.transpose(0, 2, 1)
    return jacobian, weights, blocks, np.asarray(addresses), np.asarray(dimensions)


class FactoredHessianTests(unittest.TestCase):
    def test_cone_gathers_keep_full_native_rows(self):
        jacobian, weights, blocks, addresses, dimensions = random_inputs([1, 8, 17], [3, 4, 6])
        graph = jax.make_jaxpr(
            lambda matrix, diagonal, cones: factored_hessian(
                matrix, diagonal, cones, addresses, dimensions
            )
        )(jacobian, weights, blocks)
        cone_gathers = [
            equation
            for equation in graph.jaxpr.eqns
            if equation.primitive.name == "gather" and equation.invars[0].aval.shape == blocks.shape
        ]
        self.assertTrue(cone_gathers)
        for equation in cone_gathers:
            self.assertEqual(equation.params["slice_sizes"], (1, 6, 6))

    def test_unused_padded_cone_entries_do_not_propagate_nan(self):
        jacobian, weights, blocks, addresses, dimensions = random_inputs([1, 8, 17], [3, 4, 6])
        for index, dimension in enumerate(dimensions):
            blocks[index, dimension:, :] = np.nan
            blocks[index, :, dimension:] = np.nan
        actual = jax.jit(
            lambda matrix, diagonal, cones: factored_hessian(
                matrix, diagonal, cones, addresses, dimensions
            )
        )(jacobian, weights, blocks)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(
            actual,
            dense_reference(jacobian, weights, blocks, addresses, dimensions),
            rtol=3e-6,
            atol=3e-5,
        )

    def setUp(self):
        self.device_scope = jax.default_device(jax.devices("cpu")[0])
        self.device_scope.__enter__()

    def tearDown(self):
        self.device_scope.__exit__(None, None, None)

    def test_mixed_unsorted_nonzero_blocks_match_explicit_dense(self):
        inputs = random_inputs([19, 1, 10, 25, 5], [6, 4, 3, 6, 3])
        np.testing.assert_allclose(
            factored_hessian(*inputs), dense_reference(*inputs), rtol=3e-6, atol=3e-5
        )

    def test_overlapping_blocks_are_added_not_overwritten(self):
        inputs = random_inputs([1, 3, 1], [6, 4, 3])
        np.testing.assert_allclose(
            factored_hessian(*inputs), dense_reference(*inputs), rtol=3e-6, atol=3e-5
        )

    def test_general_symmetric_blocks_and_zero_diagonal_rows(self):
        jacobian, weights, blocks, addresses, dimensions = random_inputs([2, 9, 17], [3, 4, 6])
        random = np.random.default_rng(327)
        blocks = random.normal(size=blocks.shape).astype(np.float32)
        blocks += blocks.transpose(0, 2, 1)
        weights[::2] = 0
        np.testing.assert_allclose(
            factored_hessian(jacobian, weights, blocks, addresses, dimensions),
            dense_reference(jacobian, weights, blocks, addresses, dimensions),
            rtol=3e-6,
            atol=3e-5,
        )

    def test_empty_groups_and_scalar_hessian(self):
        addresses = np.array([], dtype=np.int32)
        dimensions = np.array([], dtype=np.int32)
        jacobian, weights, blocks, _, _ = random_inputs(addresses, dimensions)
        for empty_blocks in (blocks, 0.0, jp.array(0.0)):
            np.testing.assert_allclose(
                factored_hessian(jacobian, weights, empty_blocks, addresses, dimensions),
                jacobian.T @ (weights[:, None] * jacobian),
                rtol=2e-6,
                atol=2e-6,
            )
        actual = factored_hessian(jp.zeros((0, 7)), jp.zeros(0), 0.0, addresses, dimensions)
        np.testing.assert_array_equal(actual, np.zeros((7, 7)))

    def test_each_dimension_with_other_groups_empty(self):
        for dimension in (3, 4, 6):
            inputs = random_inputs([0, 8], [dimension, dimension])
            np.testing.assert_allclose(
                factored_hessian(*inputs), dense_reference(*inputs), rtol=3e-6, atol=3e-5
            )

    def test_static_metadata_validation(self):
        jacobian, weights, blocks, _, _ = random_inputs([0], [3])
        invalid_metadata = (
            (np.array([-1]), np.array([3])),
            (np.array([29]), np.array([3])),
            (np.array([0]), np.array([1])),
            (np.array([0]), np.array([2])),
            (np.array([0]), np.array([5])),
            (np.array([0.0]), np.array([3])),
            (np.array([True]), np.array([3])),
            (np.array([0]), np.array([3.0])),
            (np.array([[0]]), np.array([3])),
            (np.array([0]), np.array([[3]])),
            (np.array([0, 5]), np.array([3])),
            (np.array([2**64 - 1], dtype=np.uint64), np.array([3])),
            (jp.array([0]), np.array([3])),
            (np.array([0]), jp.array([3])),
        )
        for addresses, dimensions in invalid_metadata:
            with self.subTest(addresses=addresses, dimensions=dimensions):
                with self.assertRaises((TypeError, ValueError)):
                    factored_hessian(jacobian, weights, blocks, addresses, dimensions)

    def test_runtime_shapes_rejected(self):
        jacobian, weights, blocks, addresses, dimensions = random_inputs([0], [3])
        for bad_jacobian, bad_weights, bad_blocks in (
            (jacobian.ravel(), weights, blocks),
            (jacobian, weights[:, None], blocks),
            (jacobian, weights[:-1], blocks),
            (jacobian, weights, blocks[:, :3, :3]),
            (jacobian, weights, np.zeros((2, 6, 6))),
            (jacobian, weights, 0.0),
        ):
            with self.assertRaises(ValueError):
                factored_hessian(bad_jacobian, bad_weights, bad_blocks, addresses, dimensions)

    def test_jit_changed_runtime_values_without_retracing(self):
        jacobian, weights, blocks, addresses, dimensions = random_inputs([2, 9, 17], [3, 4, 6])
        traces = []

        def evaluate(runtime_jacobian, runtime_weights, runtime_blocks):
            traces.append(True)
            return factored_hessian(
                runtime_jacobian, runtime_weights, runtime_blocks, addresses, dimensions
            )

        compiled = jax.jit(evaluate)
        for scale in (1.0, 1.7):
            values = (jacobian * scale, weights / scale, blocks * scale)
            actual = compiled(*values)
            np.testing.assert_allclose(
                actual, dense_reference(*values, addresses, dimensions), rtol=5e-6, atol=6e-5
            )
            self.assertEqual(actual.devices(), {jax.devices("cpu")[0]})
        self.assertEqual(len(traces), 1)

    def test_cpu_gradient_and_cholesky_solve_match_dense(self):
        jacobian, weights, blocks, addresses, dimensions = random_inputs([1, 8, 17], [3, 4, 6])

        def factored_loss(runtime_jacobian, runtime_weights, runtime_blocks):
            return jp.sum(
                jp.square(
                    factored_hessian(
                        runtime_jacobian, runtime_weights, runtime_blocks, addresses, dimensions
                    )
                )
            )

        def dense_loss(runtime_jacobian, runtime_weights, runtime_blocks):
            constraint_matrix = jp.diag(runtime_weights)
            for index, (address, dimension) in enumerate(zip(addresses, dimensions)):
                constraint_matrix = constraint_matrix.at[
                    address : address + dimension, address : address + dimension
                ].add(runtime_blocks[index, :dimension, :dimension])
            return jp.sum(jp.square(runtime_jacobian.T @ constraint_matrix @ runtime_jacobian))

        values = (jp.array(jacobian), jp.array(weights), jp.array(blocks))
        actual_gradients = jax.jit(jax.grad(factored_loss, argnums=(0, 1, 2)))(*values)
        reference_gradients = jax.jit(jax.grad(dense_loss, argnums=(0, 1, 2)))(*values)
        for actual, reference in zip(actual_gradients, reference_gradients):
            np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=0.01)
        mass = jp.eye(jacobian.shape[1]) * 2
        gradient = jp.arange(jacobian.shape[1], dtype=jp.float32) - 2
        hessian = factored_hessian(*values, addresses, dimensions) + mass
        actual = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(hessian), gradient)
        reference = np.linalg.solve(
            dense_reference(*values, addresses, dimensions) + mass, gradient
        )
        np.testing.assert_allclose(actual, reference, rtol=4e-6, atol=2e-7)

    def test_original_capacity_jaxpr_has_no_dense_or_per_contact_square(self):
        nefc, nv = 189631, 43
        dimensions = np.full(62709, 3, dtype=np.int32)
        dimensions[:300] = 4
        dimensions[300:600] = 6
        addresses = np.concatenate(([0], np.cumsum(dimensions[:-1])))
        traced = jax.make_jaxpr(
            lambda jacobian, weights, blocks: factored_hessian(
                jacobian, weights, blocks, addresses, dimensions
            )
        )(
            jax.ShapeDtypeStruct((nefc, nv), jp.float32),
            jax.ShapeDtypeStruct((nefc,), jp.float32),
            jax.ShapeDtypeStruct((len(addresses), 6, 6), jp.float32),
        )
        shapes = []
        dots = []

        def visit(jaxpr):
            for equation in jaxpr.eqns:
                for variable in (*equation.invars, *equation.outvars):
                    shapes.append(getattr(getattr(variable, "aval", None), "shape", ()))
                if equation.primitive.name == "dot_general":
                    dots.append(equation.outvars[0].aval.shape)
                for parameter in equation.params.values():
                    if hasattr(parameter, "jaxpr"):
                        visit(parameter.jaxpr)
                    elif hasattr(parameter, "eqns"):
                        visit(parameter)

        visit(traced.jaxpr)
        self.assertNotIn((nefc, nefc), shapes)
        self.assertFalse(any(len(shape) > 2 and shape[-2:] == (nv, nv) for shape in shapes))
        self.assertEqual(dots.count((nv, nv)), 4)
        self.assertEqual(
            sorted(shape for shape in dots if len(shape) == 3),
            [(300, 6, nv), (300, 6, nv), (62109, 6, nv)],
        )
        self.assertLessEqual(max(int(np.prod(shape)) for shape in shapes), 1 << 24)


class NativeGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bodies = "".join(
            f'<body pos="{index * 0.5} 0 0.095"><freejoint/>'
            f'<geom type="sphere" size="0.1" mass="1" condim="{dimension}"/></body>'
            for index, dimension in enumerate((3, 4, 6))
        )
        native = mujoco.MjModel.from_xml_string(
            '<mujoco><option cone="elliptic" solver="Newton"/>'
            '<worldbody><geom type="plane" size="3 3 0.1" condim="3"/>'
            f"{bodies}</worldbody></mujoco>"
        )
        with jax.default_device(jax.devices("cpu")[0]):
            cls.model = mjx.put_model(native)
            cls.data = jax.jit(mjx.forward)(cls.model, mjx.make_data(cls.model))
            cls.context = solver._Context.create(cls.model, cls.data, grad=False)
            count = np.count_nonzero(cls.data.contact.dim > 1)
            random = np.random.default_rng(327)
            factors = jp.array(random.normal(size=(count, 6, 6)), dtype=jp.float32)
            cls.context = cls.context.replace(
                h=factors @ factors.transpose(0, 2, 1),
                Ma=jp.arange(cls.model.nv, dtype=jp.float32) * 0.1,
            )

    def test_stock_context_gradient_and_solve_cpu_parity(self):
        original = solver._update_gradient
        expected = jax.jit(original)(self.model, self.data, self.context)
        actual = jax.jit(native_update_gradient)(self.model, self.data, self.context)
        self.assertIs(type(actual), type(self.context))
        np.testing.assert_array_equal(actual.grad, expected.grad)
        np.testing.assert_allclose(actual.Mgrad, expected.Mgrad, rtol=2e-5, atol=3e-6)
        for field in self.context.__dataclass_fields__:
            if field not in ("grad", "Mgrad"):
                np.testing.assert_array_equal(getattr(actual, field), getattr(self.context, field))
        self.assertIs(solver._update_gradient, original)

    def test_fallback_does_not_recurse_when_parent_patches_solver(self):
        for option in (
            self.model.opt.replace(solver=mjx.SolverType.CG),
            self.model.opt.replace(cone=mjx.ConeType.PYRAMIDAL),
        ):
            model = self.model.replace(opt=option)
            expected = solver._update_gradient(model, self.data, self.context)
            with mock.patch.object(solver, "_update_gradient", native_update_gradient):
                actual = native_update_gradient(model, self.data, self.context)
            np.testing.assert_array_equal(actual.grad, expected.grad)
            np.testing.assert_array_equal(actual.Mgrad, expected.Mgrad)


if __name__ == "__main__":
    unittest.main()
