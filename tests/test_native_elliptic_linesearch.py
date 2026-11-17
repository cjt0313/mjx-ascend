from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jp
from mujoco.mjx._src import solver
from mujoco.mjx._src.types import ConeType
import numpy as np

from mjx_ascend.native_elliptic_linesearch import native_linesearch_point
from mjx_ascend.staged_libero_step import staged_kernels


class NativeEllipticLineSearchTests(unittest.TestCase):
    def inputs(self, square=1.0, dot=-1.0000001, direction=1.0):
        model = SimpleNamespace(
            opt=SimpleNamespace(cone=ConeType.ELLIPTIC, disableflags=0),
            dof_hasfrictionloss=np.array([False]),
            tendon_hasfrictionloss=np.array([], bool),
        )
        data = SimpleNamespace(
            ne=0, nf=0, nl=0, contact=SimpleNamespace(dim=np.array([3]), efc_address=np.array([0]))
        )
        context = SimpleNamespace(
            Jaref=jp.zeros(3),
            fri=jp.array([[0.5, 1.0, 1.0, 0.0, 0.0, 0.0]]),
            u=jp.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            dm=jp.ones(1),
        )
        arguments = (
            model,
            data,
            context,
            jp.array(1.0),
            jp.zeros(3),
            jp.zeros((3, 3)),
            jp.zeros(3),
            jp.array([square]),
            jp.zeros(1),
            jp.array([dot]),
            jp.array([direction]),
        )
        return arguments

    def test_negative_roundoff_squared_norm_no_longer_poisons_inactive_cones(self):
        arguments = self.inputs()
        original = solver._LSPoint.create(*arguments)
        self.assertFalse(np.isfinite(float(original.cost)))
        actual = jax.jit(lambda: native_linesearch_point(solver._LSPoint, *arguments))()
        for value in (actual.cost, actual.deriv_0, actual.deriv_1):
            self.assertTrue(np.isfinite(float(value)))
        self.assertEqual(float(actual.cost), 0.0)

    def test_nonnegative_points_match_stock(self):
        for square, dot, direction in ((1.0, 0.0, 1.0), (0.01, 0.002, 0.03), (0.0, 0.0, 0.0)):
            arguments = self.inputs(square, dot, direction)
            original = solver._LSPoint.create(*arguments)
            actual = native_linesearch_point(solver._LSPoint, *arguments)
            for name in ("alpha", "cost", "deriv_0", "deriv_1"):
                np.testing.assert_array_equal(getattr(actual, name), getattr(original, name))

    def test_friction_loss_points_and_other_cones_match_stock(self):
        for cone in (ConeType.ELLIPTIC, ConeType.PYRAMIDAL):
            arguments = list(self.inputs(1.0, 0.0, 1.0))
            model, data, context = arguments[:3]
            model.opt.cone = cone
            model.dof_hasfrictionloss[:] = True
            data.efc_D = jp.ones(3)
            data.efc_frictionloss = jp.array([0.01, 0.02, 0.03])
            arguments[4] = jp.array([0.1, -0.2, 0.3])
            original = solver._LSPoint.create(*arguments)
            actual = native_linesearch_point(solver._LSPoint, *arguments)
            for name in ("alpha", "cost", "deriv_0", "deriv_1"):
                np.testing.assert_array_equal(getattr(actual, name), getattr(original, name))

    def test_invalid_selection_and_version_reject_before_descriptor_patch(self):
        original = solver._LSPoint.__dict__["create"]
        for invalid in (None, 0, 1, "true"):
            with self.assertRaises(ValueError):
                with staged_kernels(stable_elliptic_linesearch=invalid):
                    self.fail("Invalid selection accepted")
            self.assertIs(solver._LSPoint.__dict__["create"], original)
        with mock.patch("mujoco.__version__", "3.3.0"), self.assertRaises(ValueError):
            with staged_kernels(stable_elliptic_linesearch=True):
                self.fail("Unpinned version accepted")
        self.assertIs(solver._LSPoint.__dict__["create"], original)

    def test_descriptor_is_restored_on_normal_and_exception_exit(self):
        original = solver._LSPoint.__dict__["create"]
        with staged_kernels(stable_elliptic_linesearch=True):
            self.assertIs(solver._LSPoint.__dict__["create"].__func__, native_linesearch_point)
        self.assertIs(solver._LSPoint.__dict__["create"], original)
        with self.assertRaises(RuntimeError):
            with staged_kernels(stable_elliptic_linesearch=True):
                raise RuntimeError("test")
        self.assertIs(solver._LSPoint.__dict__["create"], original)


if __name__ == "__main__":
    unittest.main()
