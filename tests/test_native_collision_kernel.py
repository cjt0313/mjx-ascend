"""CPU validation of the device-native, single-contact convex kernel."""

import functools
import itertools
import unittest

import jax
import jax.numpy as jp
import numpy as np

from mjx_ascend import native_cylinder_mesh as kernel


def box(center, half=(0.5, 0.5, 0.5), rotation=None):
    vertices = np.array(list(itertools.product((-1, 1), repeat=3)), np.float32)
    rotation = np.eye(3) if rotation is None else rotation
    return jp.asarray((vertices * half) @ rotation.T + center, dtype=jp.float32)


class NativeCollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = staticmethod(jax.jit(kernel.collide_cylinder_mesh))

    def collide(self, vertices, **kwargs):
        return kernel.collide_cylinder_mesh(
            jp.zeros(3), jp.eye(3), jp.array([1.0, 1.0]), vertices, **kwargs
        )

    def check_result(self, result, distance, normal, atol=2e-4):
        self.assertTrue(bool(result.valid), f"status={int(result.status)}")
        np.testing.assert_allclose(result.distance, distance, atol=atol)
        np.testing.assert_allclose(result.normal, normal, atol=atol)
        np.testing.assert_allclose(
            result.mesh_witness - result.cylinder_witness,
            result.distance * result.normal,
            atol=atol,
        )
        np.testing.assert_allclose(
            result.contact, (result.mesh_witness + result.cylinder_witness) / 2, atol=atol
        )

    def test_analytic_support_local_z(self):
        point = kernel.cylinder_support(
            jp.zeros(3), jp.eye(3), jp.array([2.0, 3.0]), jp.array([3.0, 4.0, -1.0])
        )
        np.testing.assert_allclose(point, [1.2, 1.6, -3.0], atol=1e-6)
        np.testing.assert_allclose(
            kernel.cylinder_support(
                jp.zeros(3), jp.eye(3), jp.array([2.0, 3.0]), jp.array([0.0, 0.0, 1.0])
            ),
            [0, 0, 3],
        )

    def test_axis_separation_eager_and_jit(self):
        for center, expected, normal in [
            ((3, 0, 0), 1.5, (1, 0, 0)),
            ((0, 0, 3), 1.5, (0, 0, 1)),
            ((-3, 0, 0), 1.5, (-1, 0, 0)),
        ]:
            vertices = box(center)
            self.check_result(self.collide(vertices), expected, normal)
            result = self.compiled(jp.zeros(3), jp.eye(3), jp.array([1.0, 1.0]), vertices)
            self.check_result(result, expected, normal)
            self.assertEqual(int(result.status), kernel.Status.SEPARATED)

    def test_rigid_transform_equivariance(self):
        angle = 0.61
        rotation = jp.array(
            [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]],
            dtype=jp.float32,
        )
        translation = jp.array([2.0, -3.0, 4.0])
        for center, distance in [((3, 0, 0), 1.5), ((1.3, 0, 0), -0.2)]:
            vertices = box(center) @ rotation.T + translation
            result = self.compiled(translation, rotation, jp.array([1.0, 1.0]), vertices)
            if distance > 0:
                self.check_result(result, distance, rotation @ jp.array([1.0, 0, 0]), atol=6e-4)
            else:
                self.assertFalse(bool(result.valid))
                self.assertTrue(np.isnan(float(result.distance)))

    def test_diagonal_separation(self):
        result = self.compiled(jp.zeros(3), jp.eye(3), jp.array([1.0, 1.0]), box((2, 2, 0)))
        self.check_result(result, np.sqrt(4.5) - 1, (2**-0.5, 2**-0.5, 0), atol=6e-4)

    def test_oblique_box_faces_against_analytic_cylinder_support(self):
        for angle in np.linspace(0.1, 1.45, 9):
            normal = np.array([np.cos(angle), 0, np.sin(angle)], np.float32)
            rotation = np.array(
                [[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]],
                np.float32,
            )
            cylinder_plane = np.cos(angle) + np.sin(angle)
            for distance in (0.3, -0.15):
                vertices = box(
                    normal * (cylinder_plane + 0.5 + distance), half=(0.5, 3, 3), rotation=rotation
                )
                result = self.compiled(jp.zeros(3), jp.eye(3), jp.array([1.0, 1.0]), vertices)
                with self.subTest(angle=angle, distance=distance):
                    if distance > 0:
                        self.check_result(result, distance, normal, atol=1e-3)
                    else:
                        self.assertFalse(bool(result.valid))
                        self.assertTrue(np.isnan(float(result.distance)))

    def test_scale_and_vertex_permutation(self):
        for scale in (0.01, 1.0, 100.0):
            for center, distance in [((3, 0, 0), 1.5), ((1.3, 0, 0), -0.2)]:
                vertices = box(center)[::-1] * scale
                result = self.compiled(
                    jp.zeros(3), jp.eye(3), jp.array([1.0, 1.0]) * scale, vertices
                )
                if distance > 0:
                    self.check_result(
                        result, distance * scale, (1, 0, 0), atol=6e-4 * max(1, scale)
                    )
                    self.assertLessEqual(float(result.error_bound), 1.01e-5 * scale)
                else:
                    self.assertFalse(bool(result.valid))
                    self.assertTrue(np.isnan(float(result.distance)))

    def test_vmap_and_native_three_entry_size(self):
        vertices = jp.stack((box((3, 0, 0)), box((1.3, 0, 0))))
        compiled = jax.jit(
            jax.vmap(
                lambda mesh: kernel.collide_cylinder_mesh(
                    jp.zeros(3), jp.eye(3), jp.array([1.0, 1.0, 0.0]), mesh
                )
            )
        )
        results = compiled(vertices)
        np.testing.assert_array_equal(results.valid, [True, False])
        np.testing.assert_allclose(results.distance[0], 1.5, atol=5e-4)
        self.assertTrue(np.isnan(float(results.distance[1])))

    def test_simplex_projection_preserves_axis_aligned_plane(self):
        points = jp.array(
            [[0.0001, 0.5, 0.5], [0.0001, -0.7, -0.7], [0.0001, 0.4, -0.8], [0.0, 0.0, 0.0]]
        )
        for active in ([True, True, False, False], [True, True, True, False]):
            closest, weights, good = kernel._closest_simplex(points, jp.array(active))
            self.assertTrue(bool(good))
            np.testing.assert_allclose(closest, [0.0001, 0.0, 0.0], atol=1e-10, rtol=0)
            np.testing.assert_allclose(weights @ points, closest, atol=1e-7, rtol=0)

    def test_interior_triangle_projection_preserves_plane(self):
        points = jp.array(
            [[0.0001, 0.5, 0.5], [0.0001, -0.7, 0.5], [0.0001, 0.4, -0.8], [0.0, 0.0, 0.0]]
        )
        closest, weights, good = kernel._closest_simplex(
            points, jp.array([True, True, True, False])
        )
        self.assertTrue(bool(good))
        self.assertTrue(bool(jp.all(weights[:3] > 0.01)))
        np.testing.assert_allclose(closest, [0.0001, 0.0, 0.0], atol=1e-10, rtol=0)
        np.testing.assert_allclose(weights @ points, closest, atol=1e-7, rtol=0)

    def test_support_under_rotation_and_translated_mesh(self):
        rotation = jp.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        position = jp.array([2.0, 3.0, 4.0])
        point = kernel.cylinder_support(
            position, rotation, jp.array([2.0, 3.0, 0.0]), jp.array([1.0, 0.0, 0.0])
        )
        np.testing.assert_allclose(point, [5, 3, 4])
        vertices = box((3, 4, 5))
        supported = kernel.mesh_support(vertices, jp.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(supported, [3.5, 4.5, 5.5])

    def test_touch_penetration_and_containment_are_rejected(self):
        for center in ((1.5, 0, 0), (1.3, 0, 0), (0, 0, 1.3), (0, 0, 0)):
            result = self.compiled(jp.zeros(3), jp.eye(3), jp.array([1.0, 1.0]), box(center))
            self.assertFalse(bool(result.valid))
            self.assertTrue(np.isnan(float(result.distance)))
            self.assertEqual(int(result.epa_iterations), 0)

    def test_iteration_exhaustion_is_rejected(self):
        for limit in (0, 1):
            fn = jax.jit(functools.partial(kernel.collide_cylinder_mesh, max_gjk_iterations=limit))
            result = fn(jp.zeros(3), jp.eye(3), jp.array([1.0, 1.0]), box((2, 2, 0)))
            self.assertFalse(bool(result.valid))
            self.assertEqual(int(result.status), kernel.Status.GJK_LIMIT)

    def test_invalid_and_degenerate_inputs(self):
        for vertices, size in [
            (box((3, 0, 0)).at[0, 0].set(jp.nan), [1.0, 1.0]),
            (box((3, 0, 0)), [-1.0, 1.0]),
            (box((3, 0, 0), half=(0.5, 0.5, 0)), [1.0, 1.0]),
        ]:
            result = self.compiled(jp.zeros(3), jp.eye(3), jp.array(size), vertices)
            self.assertFalse(bool(result.valid))
            self.assertIn(
                int(result.status), (kernel.Status.INVALID_INPUT, kernel.Status.DEGENERATE)
            )

    def test_rotation_validation_and_static_configuration(self):
        for rotation in (jp.eye(3) * 2, jp.diag(jp.array([-1.0, 1.0, 1.0]))):
            result = self.compiled(jp.zeros(3), rotation, jp.array([1.0, 1.0]), box((3, 0, 0)))
            self.assertFalse(bool(result.valid))
            self.assertEqual(int(result.status), kernel.Status.INVALID_INPUT)
        for config in (
            {"max_gjk_iterations": -1},
            {"tolerance": 0},
        ):
            with self.assertRaises(ValueError):
                self.collide(box((3, 0, 0)), **config)
        with self.assertRaises(ValueError):
            self.collide(jp.zeros((3, 3)))

    def test_lowered_program_has_no_callbacks(self):
        program = str(
            self.compiled.lower(
                jp.zeros(3), jp.eye(3), jp.array([1.0, 1.0]), box((3, 0, 0))
            ).compiler_ir()
        )
        for forbidden in (
            "xla_python_cpu_callback",
            "pure_callback",
            "io_callback",
            "mj_collision",
        ):
            self.assertNotIn(forbidden, program)


if __name__ == "__main__":
    unittest.main()
