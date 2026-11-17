"""CPU contract tests; native device, forces and trajectories need parent integration.

MJX 3.2.7 rejects BOX-MESH models with nonzero geom margin/gap in put_model.
The real convex fixtures therefore use zero model margins and a positive wrapper
threshold, including grazing and threshold-adjacent positions. Its collider
emits inactive distance-one sentinels for grazing/separated cube faces.
"""

from functools import partial
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jp
import numpy as np

from mjx_ascend import native_mesh_broadphase as broadphase


class SphereBoundTests(unittest.TestCase):
    def bound(self, positions, radii=(1.0, 1.0), margin=0.0, geometry=((0, 1),)):
        return jax.jit(partial(broadphase.sphere_separation_lower_bound, margin=margin))(
            jp.array(positions, dtype=jp.float32),
            jp.array(radii, dtype=jp.float32),
            jp.array(geometry, dtype=jp.int32),
        )

    def test_near_grazing_and_separated(self):
        for distance, expected in [(1.5, False), (2.0, False), (2.000001, False), (4.0, True)]:
            lower, skip = self.bound([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
            self.assertEqual(bool(skip), expected)
            self.assertEqual(lower.shape, (1,))
            self.assertEqual(skip.shape, ())
            self.assertLessEqual(float(lower[0]), distance - 2.0)

    def test_margin_independent_and_strict_boundary(self):
        positions = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        lower, _ = self.bound(positions)
        same, skip = self.bound(positions, margin=float(lower[0]))
        np.testing.assert_array_equal(lower, same)
        self.assertFalse(bool(skip))
        self.assertFalse(bool(self.bound(positions, margin=2.0)[1]))
        self.assertTrue(bool(self.bound(positions, margin=1.0)[1]))

    def test_pad_and_large_coordinate_translation(self):
        for origin, distance in [(0.0, 4.0), (1e6, 4.0), (1e6, 100.0)]:
            lower, skip = self.bound([[origin, 0.0, 0.0], [origin + distance, 0.0, 0.0]])
            minimum_pad = 64 * np.finfo(np.float32).eps * (origin + distance + 3) + 1e-5
            self.assertLessEqual(float(lower[0]), distance - 2 - minimum_pad)
            self.assertEqual(bool(skip), distance == 100.0 or origin == 0.0)

    def test_invalid_and_overflow_fail_open(self):
        for bad in [np.nan, np.inf, -np.inf]:
            self.assertFalse(bool(self.bound([[0.0, 0.0, 0.0], [bad, 0.0, 0.0]])[1]))
            self.assertFalse(bool(self.bound([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], (1.0, bad))[1]))
        for radius in [0.0, -1.0]:
            self.assertFalse(bool(self.bound([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], (1.0, radius))[1]))
        for positions, radii in [
            ([[-3e38, 0.0, 0.0], [3e38, 0.0, 0.0]], (1.0, 1.0)),
            ([[0.0, 0.0, 0.0], [2e20, 0.0, 0.0]], (1.0, 1.0)),
            ([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], (3e38, 3e38)),
        ]:
            self.assertFalse(bool(self.bound(positions, radii)[1]))
        for geometry in [[[-1, 1]], [[0, 2]]]:
            self.assertFalse(
                bool(self.bound([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], geometry=geometry)[1])
            )

    def test_whole_chunk_requires_every_pair(self):
        lower, skip = self.bound(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            (1.0, 1.0, 1.0),
            geometry=((0, 1), (0, 2)),
        )
        self.assertFalse(bool(skip))
        self.assertGreater(float(lower[0]), 0.0)
        self.assertLess(float(lower[1]), 0.0)

    def test_static_guards(self):
        for margin in [-1.0, np.nan, np.inf, -np.inf, True]:
            with self.assertRaises((TypeError, ValueError)):
                self.bound([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], margin=margin)
            with self.assertRaises((TypeError, ValueError)):
                broadphase.wrap_mesh_broadphase(lambda *args: None, 4, margin)
        for ncon in [0, -1, 1.5, True]:
            with self.assertRaises((TypeError, ValueError)):
                broadphase.wrap_mesh_broadphase(lambda *args: None, ncon, 0.0)


class WrapperTests(unittest.TestCase):
    def test_slot_count_matches_collider(self):
        def narrow(*args):
            return None

        narrow.ncon = 4
        with self.assertRaises(ValueError):
            broadphase.wrap_mesh_broadphase(narrow, 1, 0.01)

    def test_actual_convex_mesh_contact_and_recontact_fields(self):
        import mujoco
        from mujoco import mjx
        from mujoco.mjx._src import collision_driver

        for stationary_type in ["box", "mesh"]:
            with self.subTest(stationary_type=stationary_type):
                stationary = (
                    'type="box" size="1 1 1"'
                    if stationary_type == "box"
                    else 'type="mesh" mesh="cube"'
                )
                xml = f"""<mujoco><asset><mesh name="cube" vertex="
                    -1 -1 -1 1 -1 -1 -1 1 -1 1 1 -1
                    -1 -1 1 1 -1 1 -1 1 1 1 1 1"/></asset>
                    <worldbody><geom {stationary}/><body pos="2 0 0">
                    <freejoint/><geom type="mesh" mesh="cube"/>
                    </body></worldbody></mujoco>"""
                cpu_model = mujoco.MjModel.from_xml_string(xml)
                cpu_data = mujoco.MjData(cpu_model)
                mujoco.mj_forward(cpu_model, cpu_data)
                model = mjx.put_model(cpu_model)
                data = mjx.put_data(cpu_model, cpu_data)
                key, pairs = next(iter(collision_driver._geom_groups(model).items()))
                geometry = jp.array(np.array(pairs)[:, :2], dtype=jp.int32)
                original = collision_driver._COLLISION_FUNC[key.types]
                wrapped = broadphase.wrap_mesh_broadphase(original, original.ncon, 0.1)
                reference = jax.jit(lambda state: original(model, state, key, geometry))
                candidate = jax.jit(lambda state: wrapped(model, state, key, geometry))
                for distance in [1.8, 2.0, 2.05, 2.1, 8.0, 1.9]:
                    state = data.replace(
                        geom_xpos=data.geom_xpos.at[1].set(
                            jp.array([distance, 0.0, 0.0], dtype=jp.float32)
                        )
                    )
                    expected = reference(state)
                    actual = candidate(state)
                    active = np.asarray(expected[0]) < 0.1
                    self.assertEqual(bool(active.any()), distance < 2.0)
                    for field, source in zip(actual, expected):
                        self.assertEqual(field.shape, source.shape)
                        self.assertEqual(field.devices(), state.geom_xpos.devices())
                        np.testing.assert_array_equal(
                            np.asarray(field)[active], np.asarray(source)[active]
                        )
                        if distance != 8.0:
                            np.testing.assert_array_equal(field, source)
                    if distance == 8.0:
                        self.assertTrue(np.all(np.asarray(actual[0]) > 0.1))
                        self.assertTrue(all(np.isfinite(field).all() for field in actual))

    def test_runtime_cond_shape_order_and_residency(self):
        geometry = jp.array([[1, 0], [0, 2]], dtype=jp.int32)
        radii = jp.ones(3, dtype=jp.float32)

        def narrow(model, data, key, pairs):
            count = pairs.shape[0] * 4
            return (
                jp.full((count,), -7.0, dtype=data.geom_xpos.dtype),
                jp.full((count, 3), 13.0, dtype=data.geom_xpos.dtype),
                jp.full((count, 3, 3), 17.0, dtype=data.geom_xpos.dtype),
            )

        wrapped = broadphase.wrap_mesh_broadphase(narrow, 4, 0.1)
        self.assertEqual(wrapped.ncon, 4)

        def operation(positions):
            return wrapped(
                SimpleNamespace(geom_rbound=radii),
                SimpleNamespace(geom_xpos=positions),
                None,
                geometry,
            )

        execute = jax.jit(operation)
        far = jp.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        near = far.at[1, 0].set(1.0)
        lower, _ = broadphase.sphere_separation_lower_bound(far, radii, geometry, 0.1)
        dist, pos, frame = execute(far)
        np.testing.assert_array_equal(dist, jp.repeat(lower, 4))
        np.testing.assert_array_equal(pos, [[4.0, 0.0, 0.0]] * 4 + [[10.0, 0.0, 0.0]] * 4)
        np.testing.assert_array_equal(frame, np.broadcast_to(np.eye(3), (8, 3, 3)))
        for result in (dist, pos, frame):
            self.assertIsInstance(result, jax.Array)
            self.assertEqual(result.devices(), far.devices())
            self.assertTrue(np.isfinite(result).all())
        for positions in [near, far.at[1, 0].set(np.nan), near, far]:
            outputs = execute(positions)
            if positions is not far:
                expected = narrow(None, SimpleNamespace(geom_xpos=positions), None, geometry)
                for actual, original in zip(outputs, expected):
                    np.testing.assert_array_equal(actual, original)
        lowered = execute.lower(far).as_text()
        self.assertIn("stablehlo.case", lowered)
        self.assertNotIn("callback", lowered.lower())


if __name__ == "__main__":
    unittest.main()
