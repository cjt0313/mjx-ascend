"""CPU-JIT contracts for isolated packing; no native trajectory claims."""

from types import SimpleNamespace
from typing import NamedTuple
import unittest

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import collision_driver
import numpy as np

from mjx_ascend.native_candidate_packing import wrap_candidate_packing
from mjx_ascend.native_mesh_broadphase import sphere_separation_lower_bound


class Model(NamedTuple):
    geom_rbound: object


class Data(NamedTuple):
    geom_xpos: object


class CandidatePackingTests(unittest.TestCase):
    def setUp(self):
        self.executed = []
        self.model = Model(jp.full((10,), 0.2, jp.float32))
        self.data = Data(
            jp.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.1, 1.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [3.0, 2.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.1, 3.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [3.0, 4.0, 0.0],
                ],
                jp.float32,
            )
        )
        self.geometry = jp.array([[8, 9], [2, 3], [0, 1], [6, 7], [4, 5]], jp.int32)

    def collider(self, unsupported=False):
        def record(geometry):
            self.executed.append(np.array(geometry, copy=True))

        def collision(model, data, key, geometry):
            jax.debug.callback(record, geometry, ordered=True)
            pairs = jp.repeat(geometry, 3, axis=0)
            slots = jp.tile(jp.arange(3, dtype=jp.float32), geometry.shape[0])
            distance = -(pairs[:, 0] * 10 + slots + 1)
            if unsupported:
                distance = jp.where(pairs[:, 0] == 6, jp.nan, distance)
            position = jp.stack((pairs[:, 0], pairs[:, 1], slots), axis=1).astype(jp.float32)
            frame = jp.broadcast_to(jp.eye(3, dtype=jp.float32), (distance.size, 3, 3))
            if unsupported:
                position = jp.where((pairs[:, 0] == 6)[:, None], jp.nan, position)
                frame = jp.where((pairs[:, 0] == 6)[:, None, None], jp.nan, frame)
            return distance, position, frame

        collision.ncon = 3
        return collision

    def ready(self, output):
        jax.block_until_ready(output)
        jax.effects_barrier()
        for value in output:
            self.assertEqual(value.devices(), self.geometry.devices())
            self.assertTrue(all(device.platform == "cpu" for device in value.devices()))
        return tuple(np.asarray(value) for value in output)

    def run_wrapper(self, capacity=3, margin=0.0, unsupported=False, geometry=None):
        geometry = self.geometry if geometry is None else geometry
        operation = wrap_candidate_packing(self.collider(unsupported), 3, margin, capacity)
        self.assertEqual(operation.ncon, 3)
        before = [
            np.asarray(value).tobytes()
            for value in (self.model.geom_rbound, self.data.geom_xpos, geometry)
        ]
        output = self.ready(jax.jit(operation)(self.model, self.data, None, geometry))
        after = [
            np.asarray(value).tobytes()
            for value in (self.model.geom_rbound, self.data.geom_xpos, geometry)
        ]
        self.assertEqual(before, after)
        return output

    def test_mixed_stable_padding_inverse_slots_and_inactive_outputs(self):
        distance, position, frame = self.run_wrapper()
        self.assertEqual(len(self.executed), 1)
        np.testing.assert_array_equal(self.executed[0], [[2, 3], [6, 7], [2, 3]])
        self.assertEqual(distance.shape, (15,))
        self.assertEqual(position.shape, (15, 3))
        self.assertEqual(frame.shape, (15, 3, 3))
        for pair_index in [1, 3]:
            pair = np.asarray(self.geometry[pair_index])
            start = pair_index * 3
            np.testing.assert_array_equal(
                distance[start : start + 3], -(pair[0] * 10 + np.arange(3) + 1)
            )
            np.testing.assert_array_equal(
                position[start : start + 3],
                np.stack((np.repeat(pair[0], 3), np.repeat(pair[1], 3), np.arange(3)), axis=1),
            )
        lower, _ = sphere_separation_lower_bound(
            self.data.geom_xpos, self.model.geom_rbound, self.geometry, 0.0
        )
        for pair_index in [0, 2, 4]:
            start = pair_index * 3
            centers = np.asarray(self.data.geom_xpos[self.geometry[pair_index]])
            np.testing.assert_array_equal(
                distance[start : start + 3], np.repeat(lower[pair_index], 3)
            )
            np.testing.assert_array_equal(
                position[start : start + 3], np.tile(centers.mean(axis=0), (3, 1))
            )
            np.testing.assert_array_equal(frame[start : start + 3], np.tile(np.eye(3), (3, 1, 1)))

    def test_zero_candidates_skips_collider(self):
        output = self.run_wrapper(geometry=self.geometry[jp.array([0, 2, 4])])
        self.assertEqual(self.executed, [])
        self.assertTrue(np.all(output[0] > 0))

    def test_empty_pairs(self):
        output = self.run_wrapper(geometry=jp.empty((0, 2), jp.int32))
        self.assertEqual(self.executed, [])
        self.assertEqual([value.shape for value in output], [(0,), (0, 3), (0, 3, 3)])

    def test_exact_capacity(self):
        self.run_wrapper(capacity=2)
        np.testing.assert_array_equal(self.executed[0], [[2, 3], [6, 7]])

    def test_overflow_uses_full_geometry_unmodified(self):
        output = self.run_wrapper(capacity=1, unsupported=True)
        np.testing.assert_array_equal(self.executed[0], self.geometry)
        expected = self.ready(self.collider(True)(self.model, self.data, None, self.geometry))
        for actual, reference in zip(output, expected):
            np.testing.assert_array_equal(actual, reference)

    def test_invalid_bounds_fail_open(self):
        for invalid in [np.nan, np.inf, -1.0, 0.0]:
            with self.subTest(invalid=invalid):
                self.executed.clear()
                self.model = Model(jp.array([invalid] + [0.2] * 9, jp.float32))
                self.run_wrapper(capacity=4)
                np.testing.assert_array_equal(self.executed[0], [[2, 3], [0, 1], [6, 7], [2, 3]])

    def test_unsupported_candidate_nan_preserved(self):
        distance, position, frame = self.run_wrapper(unsupported=True)
        self.assertTrue(np.all(np.isnan(distance[9:12])))
        self.assertTrue(np.all(np.isnan(position[9:12])))
        self.assertTrue(np.all(np.isnan(frame[9:12])))
        self.assertFalse(np.any(np.isnan(distance[:9])))

    def test_margin_and_strict_boundary(self):
        lower, _ = sphere_separation_lower_bound(
            self.data.geom_xpos, self.model.geom_rbound, self.geometry, 0.0
        )
        for margin in [3.0, float(lower[0])]:
            self.executed.clear()
            self.run_wrapper(capacity=5, margin=margin)
            candidate_pairs = np.asarray(self.geometry)[np.asarray(lower) <= margin]
            expected = np.concatenate(
                (candidate_pairs, np.repeat(candidate_pairs[:1], 5 - len(candidate_pairs), axis=0))
            )
            np.testing.assert_array_equal(self.executed[0], expected)

    def test_same_jit_switches_zero_packed_overflow_at_runtime(self):
        operation = jax.jit(wrap_candidate_packing(self.collider(), 3, 0.0, 2))
        far = Data(self.data.geom_xpos.at[3, 0].set(3.0).at[7, 0].set(3.0))
        all_near = Data(self.data.geom_xpos.at[1::2, 0].set(0.1))
        for data, expected in [
            (far, None),
            (self.data, [[2, 3], [6, 7]]),
            (all_near, self.geometry),
            (far, None),
        ]:
            self.executed.clear()
            self.ready(operation(self.model, data, None, self.geometry))
            if expected is None:
                self.assertEqual(self.executed, [])
            else:
                self.assertEqual(len(self.executed), 1)
                np.testing.assert_array_equal(self.executed[0], expected)

    def test_invalid_geometry_and_position_are_not_certified_far(self):
        geometry = jp.array([[-1, 1], [0, 12], [8, 9]], jp.int32)
        self.run_wrapper(capacity=2, geometry=geometry)
        np.testing.assert_array_equal(self.executed[0], geometry[:2])
        self.executed.clear()
        self.data = Data(self.data.geom_xpos.at[0, 0].set(jp.inf))
        self.run_wrapper(capacity=3)
        np.testing.assert_array_equal(self.executed[0], [[2, 3], [0, 1], [6, 7]])

    def test_actual_separated_cylinder_box_adapter_cpu_fixture(self):
        from mujoco.mjx._src.types import GeomType
        from mjx_ascend.libero_native_adapter import cylinder_mesh

        model = SimpleNamespace(
            geom_size=jp.array([[0.25, 0.3, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], jp.float32),
            geom_rbound=jp.array([np.hypot(0.25, 0.3), np.sqrt(0.75), np.sqrt(0.75)], jp.float32),
        )
        data = SimpleNamespace(
            geom_xpos=jp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.0, 0.0, 0.0]], jp.float32),
            geom_xmat=jp.broadcast_to(jp.eye(3, dtype=jp.float32), (3, 3, 3)),
        )
        key = SimpleNamespace(types=(GeomType.CYLINDER, GeomType.BOX))
        geometry = jp.array([[0, 2], [0, 1]], jp.int32)
        operation = wrap_candidate_packing(cylinder_mesh, 1, 0.0, 2)
        inputs = (model.geom_size, model.geom_rbound, data.geom_xpos, data.geom_xmat, geometry)
        before = [np.asarray(value).tobytes() for value in inputs]
        actual = self.ready(jax.jit(lambda pairs: operation(model, data, key, pairs))(geometry))
        reference = self.ready(
            jax.jit(lambda pairs: cylinder_mesh(model, data, key, pairs))(geometry)
        )
        for packed, original in zip(actual, reference):
            np.testing.assert_array_equal(packed[1], original[1])
        self.assertAlmostEqual(float(actual[0][1]), 0.25, places=5)
        self.assertTrue(np.all(np.isfinite(actual[1])))
        self.assertTrue(np.all(np.isfinite(actual[2])))
        self.assertGreater(float(actual[0][0]), 0.0)
        self.assertEqual(before, [np.asarray(value).tobytes() for value in inputs])
        penetrating = SimpleNamespace(
            geom_xpos=data.geom_xpos.at[0, 0].set(0.6), geom_xmat=data.geom_xmat
        )
        rejected = self.ready(
            jax.jit(lambda pairs: operation(model, penetrating, key, pairs))(geometry)
        )
        for value in rejected:
            self.assertTrue(np.all(np.isnan(value[1])))

    def test_static_validation(self):
        collision = self.collider()
        for name in ["ncon", "capacity"]:
            for invalid in [True, 1.5, jp.array(2)]:
                arguments = dict(ncon=3, margin=0.0, capacity=2)
                arguments[name] = invalid
                with self.assertRaises(TypeError):
                    wrap_candidate_packing(collision, **arguments)
            for invalid in [0, -1]:
                arguments = dict(ncon=3, margin=0.0, capacity=2)
                arguments[name] = invalid
                with self.assertRaises(ValueError):
                    wrap_candidate_packing(collision, **arguments)
        for margin in [True, jp.array(0.0), "0"]:
            with self.assertRaises(TypeError):
                wrap_candidate_packing(collision, 3, margin, 2)
        for margin in [-1.0, np.inf, np.nan]:
            with self.assertRaises(ValueError):
                wrap_candidate_packing(collision, 3, margin, 2)
        with self.assertRaises(ValueError):
            wrap_candidate_packing(collision, 2, 0.0, 2)


class MeshCandidatePackingTests(unittest.TestCase):
    def fixture(self, stationary_type):
        stationary = (
            'type="box" size="1 1 1"' if stationary_type == "box" else 'type="mesh" mesh="cube"'
        )
        bodies = "".join(
            f'<body pos="{distance} 0 0"><freejoint/>'
            f'<geom name="moving{index}" type="mesh" mesh="cube" contype="0" conaffinity="0"/>'
            "</body>"
            for index, distance in enumerate((8.0, 1.8, 1.99999, 2.05, 1.9))
        )
        pairs = "".join(f'<pair geom1="stationary" geom2="moving{index}"/>' for index in range(5))
        native = mujoco.MjModel.from_xml_string(f"""<mujoco><asset><mesh name="cube" vertex="
            -1 -1 -1 1 -1 -1 -1 1 -1 1 1 -1
            -1 -1 1 1 -1 1 -1 1 1 1 1 1"/></asset>
            <worldbody><geom name="stationary" {stationary} contype="0" conaffinity="0"/>
            {bodies}</worldbody><contact>{pairs}</contact></mujoco>""")
        native_data = mujoco.MjData(native)
        mujoco.mj_forward(native, native_data)
        model = mjx.put_model(native)
        data = mjx.put_data(native, native_data)
        groups = collision_driver._geom_groups(model)
        self.assertEqual(len(groups), 1)
        key, pairs = next(iter(groups.items()))
        self.assertEqual(tuple(key.types), (6, 7) if stationary_type == "box" else (7, 7))
        geometry = jp.array(np.asarray(pairs, dtype=np.int32)[:, :2])
        geometry = geometry[jp.array([3, 0, 4, 2, 1])]
        return model, data, key, geometry, collision_driver._COLLISION_FUNC[key.types]

    def input_bytes(self, model, data, geometry):
        return [
            (value.shape, str(value.dtype), np.asarray(value).tobytes())
            for value in jax.tree_util.tree_leaves((model, data, geometry))
        ]

    def compare(self, model, data, key, geometry, stock, capacity, margin=0.1):
        before = self.input_bytes(model, data, geometry)
        wrapped = wrap_candidate_packing(stock, stock.ncon, margin, capacity)
        self.assertEqual(wrapped.ncon, stock.ncon)
        self.assertEqual(stock.ncon, 4)
        reference = jax.jit(stock, static_argnums=(2,))(model, data, key, geometry)
        actual = jax.jit(wrapped, static_argnums=(2,))(model, data, key, geometry)
        jax.block_until_ready((reference, actual))
        lower, _ = sphere_separation_lower_bound(
            data.geom_xpos, model.geom_rbound, geometry, margin
        )
        candidates = ~(np.asarray(lower) > margin)
        slots = np.repeat(candidates, stock.ncon)
        overflow = candidates.sum() > capacity
        for value, expected in zip(actual, reference):
            self.assertEqual(value.shape, expected.shape)
            self.assertEqual(value.shape[0], geometry.shape[0] * stock.ncon)
            self.assertEqual(value.devices(), geometry.devices())
            self.assertTrue(all(device.platform == "cpu" for device in value.devices()))
            self.assertTrue(np.isfinite(np.asarray(value)).all())
            np.testing.assert_array_equal(np.asarray(value)[slots], np.asarray(expected)[slots])
            if overflow:
                np.testing.assert_array_equal(value, expected)
        if not overflow:
            far_slots = ~slots
            midpoint = np.asarray(data.geom_xpos[geometry]).mean(axis=1)
            np.testing.assert_array_equal(
                np.asarray(actual[0])[far_slots], np.repeat(lower, stock.ncon)[far_slots]
            )
            np.testing.assert_array_equal(
                np.asarray(actual[1])[far_slots], np.repeat(midpoint, stock.ncon, axis=0)[far_slots]
            )
            np.testing.assert_array_equal(
                np.asarray(actual[2])[far_slots],
                np.broadcast_to(np.eye(3), (far_slots.sum(), 3, 3)),
            )
            self.assertTrue(np.all(np.asarray(actual[0])[far_slots] > margin))
        self.assertEqual(before, self.input_bytes(model, data, geometry))
        return reference, candidates

    def test_actual_box_mesh_mixed_near_far_grazing_order_and_padding(self):
        self.mixed("box")

    def test_actual_mesh_mesh_mixed_near_far_grazing_order_and_padding(self):
        self.mixed("mesh")

    def mixed(self, stationary_type):
        with jax.default_device(jax.devices("cpu")[0]):
            model, data, key, geometry, stock = self.fixture(stationary_type)
            reference, candidates = self.compare(model, data, key, geometry, stock, capacity=8)
            np.testing.assert_array_equal(candidates, [True, False, True, True, True])
            distances = np.asarray(reference[0]).reshape((5, stock.ncon))
            self.assertTrue(np.any(distances[2] < 0.0))
            self.assertTrue(np.any(distances[3] < 0.0))
            np.testing.assert_allclose(distances[3], 0.0, atol=2e-5)
            self.assertTrue(np.all(distances[0] > 0.0))
            self.compare(model, data, key, geometry, stock, capacity=4)
            touching = data.replace(geom_xpos=data.geom_xpos.at[3, 0].set(2.0))
            self.compare(model, touching, key, geometry, stock, capacity=8)

    def test_actual_mesh_families_zero_and_full_geometry_overflow(self):
        for stationary_type in ("box", "mesh"):
            with (
                self.subTest(stationary_type=stationary_type),
                jax.default_device(jax.devices("cpu")[0]),
            ):
                model, data, key, geometry, stock = self.fixture(stationary_type)
                _, candidates = self.compare(model, data, key, geometry, stock, capacity=1)
                self.assertEqual(candidates.sum(), 4)
                far = data.replace(geom_xpos=data.geom_xpos.at[1:, 0].set(8.0))
                _, candidates = self.compare(model, far, key, geometry, stock, capacity=8)
                self.assertFalse(candidates.any())


if __name__ == "__main__":
    unittest.main()
