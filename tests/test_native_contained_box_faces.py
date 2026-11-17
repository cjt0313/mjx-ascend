from contextlib import contextmanager
import importlib
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import collision_driver
import numpy as np

from mjx_ascend.native_contained_box_faces import (
    _box_position_status,
    wrap_native_contained_box_faces,
)
from box_fixtures import Model, Data, collision_inputs, point_set_error


def fixture(
    angles=(0.0, 0.0),
    depth=0.003,
    swapped=False,
    offset=0.0,
    large=0.8,
    thickness=0.1,
    sizes=(0.07, 0.09, 0.11),
):
    rotation_x, rotation_y = angles
    quaternion_x = np.array([np.cos(rotation_x / 2), np.sin(rotation_x / 2), 0.0, 0.0])
    quaternion_y = np.array([np.cos(rotation_y / 2), 0.0, np.sin(rotation_y / 2), 0.0])
    quaternion = np.empty(4)
    mujoco.mju_mulQuat(quaternion, quaternion_y, quaternion_x)
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    height = thickness + np.abs(matrix.reshape(3, 3)[2]) @ np.array(sizes) - depth
    fixed = f'<geom type="box" size="{large} {large} {thickness}"/>'
    moving = (
        f'<body pos="{offset} 0 {height}" quat="{" ".join(map(str, quaternion))}">'
        f'<freejoint/><geom type="box" size="{" ".join(map(str, sizes))}" mass="1"/></body>'
    )
    return mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep=".002" gravity="0 0 0" cone="elliptic" '
        'solver="Newton" iterations="100" tolerance="1e-10"/>'
        f"<worldbody>{moving + fixed if swapped else fixed + moving}</worldbody></mujoco>"
    )


@contextmanager
def installed_wrapper():
    original = collision_driver._COLLISION_FUNC[(6, 6)]
    collision_driver._COLLISION_FUNC[(6, 6)] = wrap_native_contained_box_faces(original)
    try:
        yield
    finally:
        collision_driver._COLLISION_FUNC[(6, 6)] = original


class NativeContainedBoxFacesTests(unittest.TestCase):
    def test_pruning_certifies_final_predicate_not_irrelevant_thin_box_band(self):
        centers = jp.array([[[0.0, 0.0, 0.0], [-0.001, 0.0, 0.0]]], dtype=jp.float32)
        matrices = jp.broadcast_to(jp.eye(3), (1, 2, 3, 3))
        sizes = jp.array([[[1.0, 1.0, 1.0], [0.001, 1.0, 1.0]]], dtype=jp.float32)
        removed, stable = _box_position_status(
            centers, matrices, sizes, jp.zeros((1, 1, 3)), jp.array([3e-5])
        )
        np.testing.assert_array_equal(removed, [[False]])
        np.testing.assert_array_equal(stable, [[True]])

    def test_pruning_ambiguous_final_predicate_stays_uncertified(self):
        centers = jp.array([[[0.99, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=jp.float32)
        matrices = jp.broadcast_to(jp.eye(3), (1, 2, 3, 3))
        sizes = jp.ones((1, 2, 3))
        removed, stable = _box_position_status(
            centers, matrices, sizes, jp.zeros((1, 1, 3)), jp.array([0.001])
        )
        np.testing.assert_array_equal(removed, [[False]])
        np.testing.assert_array_equal(stable, [[False]])
        for offset, expected_removed in ((-0.002, False), (0.002, True)):
            changed = centers.at[0, 0, 0].add(offset)
            actual, _ = _box_position_status(
                changed, matrices, sizes, jp.zeros((1, 1, 3)), jp.array([0.001])
            )
            np.testing.assert_array_equal(actual, [[expected_removed]])

    def test_pruning_keeps_when_both_inside_states_are_ambiguous_but_outside_impossible(self):
        centers = jp.array([[[0.99, 0.0, 0.0], [0.99, 0.0, 0.0]]], dtype=jp.float32)
        matrices = jp.broadcast_to(jp.eye(3), (1, 2, 3, 3))
        removed, stable = _box_position_status(
            centers, matrices, jp.ones((1, 2, 3)), jp.zeros((1, 1, 3)), jp.array([0.001])
        )
        np.testing.assert_array_equal(removed, [[False]])
        np.testing.assert_array_equal(stable, [[True]])

    def test_native_incident_face_threshold_and_unselected_near_plane(self):
        self.assertEqual(mujoco.__version__, "3.2.7")
        original = collision_driver._COLLISION_FUNC[(6, 6)]
        wrapped = jax.jit(wrap_native_contained_box_faces(original), static_argnums=(2,))
        cases = (
            (0.5 - 1e-5, 0.013, (0.07, 0.00992, 0.01187), 4),
            (0.5 + 1e-5, 0.013, (0.07, 0.00992, 0.01187), 2),
            (0.49, 0.00134, (0.00071, 0.00992, 0.01187), 2),
            (0.49, 2 * 0.00071 * np.sqrt(1 - 0.49**2) - 5e-6, (0.00071, 0.00992, 0.01187), 2),
        )
        for component, depth, sizes, expected_count in cases:
            for swapped in (False, True):
                with self.subTest(component=component, depth=depth, swapped=swapped):
                    native_data, model, data, key, geometry = collision_inputs(
                        fixture(
                            (0.0, np.arccos(component)),
                            depth=depth,
                            sizes=sizes,
                            swapped=swapped,
                            thickness=0.025,
                        )
                    )
                    stock = original(model, data, key, geometry)
                    actual = wrapped(model, data, key, geometry)
                    contacts = [contact for contact in native_data.contact if contact.dist < 0]
                    active = np.asarray(actual[0]) < 0
                    self.assertEqual(len(contacts), expected_count)
                    self.assertEqual(int(active.sum()), expected_count)
                    np.testing.assert_allclose(
                        np.sort(np.asarray(actual[0])[active]),
                        np.sort([contact.dist for contact in contacts]),
                        atol=8e-8,
                        rtol=3e-5,
                    )
                    self.assertLess(
                        point_set_error(
                            np.asarray(actual[1])[active],
                            np.array([contact.pos for contact in contacts]),
                        ),
                        2e-7,
                    )
                    np.testing.assert_array_equal(actual[2], stock[2])

    def test_incident_face_threshold_ambiguity_preserves_stock(self):
        original = collision_driver._COLLISION_FUNC[(6, 6)]
        wrapped = jax.jit(wrap_native_contained_box_faces(original), static_argnums=(2,))
        for component in (0.5 - 1e-8, 0.5, 0.5 + 1e-8):
            with self.subTest(component=component):
                _, model, data, key, geometry = collision_inputs(
                    fixture(
                        (0.0, np.arccos(component)),
                        depth=0.013,
                        sizes=(0.07, 0.00992, 0.01187),
                        thickness=0.025,
                    )
                )
                stock = jax.jit(original, static_argnums=(2,))(model, data, key, geometry)
                actual = wrapped(model, data, key, geometry)
                for actual_field, expected_field in zip(actual, stock):
                    np.testing.assert_array_equal(
                        np.asarray(actual_field).view(np.uint32),
                        np.asarray(expected_field).view(np.uint32),
                    )

    def test_native_driver_pruning_boundary_and_ambiguous_fallback(self):
        original = collision_driver._COLLISION_FUNC[(6, 6)]
        wrapped = jax.jit(wrap_native_contained_box_faces(original), static_argnums=(2,))
        boundary = 2 * 0.003 * 0.49 + 2 * (0.025 - 0.025 / 1.01)
        for offset, expected_count in ((-2e-4, 2), (0.0, None), (2e-4, 4)):
            for swapped in (False, True):
                with self.subTest(offset=offset, swapped=swapped):
                    native_data, model, data, key, geometry = collision_inputs(
                        fixture(
                            (0.0, np.arccos(0.49)),
                            depth=boundary + offset,
                            sizes=(0.07, 0.00992, 0.003),
                            thickness=0.025,
                            swapped=swapped,
                        )
                    )
                    stock = jax.jit(original, static_argnums=(2,))(model, data, key, geometry)
                    actual = wrapped(model, data, key, geometry)
                    if expected_count is None:
                        for actual_field, expected_field in zip(actual, stock):
                            np.testing.assert_array_equal(
                                np.asarray(actual_field).view(np.uint32),
                                np.asarray(expected_field).view(np.uint32),
                            )
                    else:
                        contacts = [contact for contact in native_data.contact if contact.dist < 0]
                        active = np.asarray(actual[0]) < 0
                        self.assertEqual(len(contacts), expected_count)
                        self.assertEqual(int(active.sum()), expected_count)
                        np.testing.assert_allclose(
                            np.sort(np.asarray(actual[0])[active]),
                            np.sort([contact.dist for contact in contacts]),
                            atol=8e-8,
                            rtol=3e-5,
                        )
                        self.assertLess(
                            point_set_error(
                                np.asarray(actual[1])[active],
                                np.array([contact.pos for contact in contacts]),
                            ),
                            2e-7,
                        )
                        np.testing.assert_array_equal(actual[2], stock[2])

    def test_actual_native_parallel_corner_edge_and_swapped_contacts(self):
        original = collision_driver._COLLISION_FUNC[(6, 6)]
        wrapped = jax.jit(wrap_native_contained_box_faces(original), static_argnums=(2,))
        for angles, expected_count in (((0.0, 0.0), 4), ((0.2, 0.0), 2), ((0.2, 0.3), 1)):
            for swapped in (False, True):
                with self.subTest(angles=angles, swapped=swapped):
                    native_data, model, data, key, geometry = collision_inputs(
                        fixture(angles, swapped=swapped)
                    )
                    stock = original(model, data, key, geometry)
                    actual = wrapped(model, data, key, geometry)
                    active = np.asarray(actual[0]) < 0
                    contacts = [contact for contact in native_data.contact if contact.dist < 0]
                    self.assertEqual(len(contacts), expected_count)
                    self.assertEqual(int(active.sum()), expected_count)
                    np.testing.assert_allclose(
                        np.sort(np.asarray(actual[0])[active]),
                        np.sort([contact.dist for contact in contacts]),
                        atol=8e-8,
                        rtol=3e-5,
                    )
                    self.assertLess(
                        point_set_error(
                            np.asarray(actual[1])[active],
                            np.array([contact.pos for contact in contacts]),
                        ),
                        2e-7,
                    )
                    np.testing.assert_array_equal(actual[2], stock[2])

    def test_clipping_and_deep_penetration_preserve_stock(self):
        original = collision_driver._COLLISION_FUNC[(6, 6)]
        wrapped = jax.jit(wrap_native_contained_box_faces(original), static_argnums=(2,))
        for native in (
            fixture((0.2, 0.3), offset=0.77),
            fixture(depth=0.3, thickness=1.0),
            fixture(offset=0.73),
        ):
            _, model, data, key, geometry = collision_inputs(native)
            expected = jax.jit(original, static_argnums=(2,))(model, data, key, geometry)
            actual = wrapped(model, data, key, geometry)
            for actual_field, expected_field in zip(actual, expected):
                np.testing.assert_array_equal(
                    np.asarray(actual_field).view(np.uint32),
                    np.asarray(expected_field).view(np.uint32),
                )

    def test_force_and_full_state_match_native_cpu(self):
        native = fixture((0.2, 0.3), depth=0.0015)
        native_data = mujoco.MjData(native)
        native_data.qvel[2] = -0.05
        model = mjx.put_model(native)
        initial = mjx.put_data(native, native_data)
        stock = jax.jit(mjx.step)(model, initial)
        jax.block_until_ready(stock)
        jax.clear_caches()
        with installed_wrapper():
            corrected = jax.jit(mjx.step)(model, initial)
        jax.block_until_ready(corrected)
        mujoco.mj_step(native, native_data)
        self.assertGreater(np.linalg.norm(native_data.qfrc_constraint), 0.0)
        for field, tolerance in (
            ("qfrc_constraint", 2e-5),
            ("qacc", 2e-4),
            ("qvel", 5e-7),
            ("qpos", 2e-7),
        ):
            reference = getattr(native_data, field)
            error = np.linalg.norm(np.asarray(getattr(corrected, field)) - reference)
            stock_error = np.linalg.norm(np.asarray(getattr(stock, field)) - reference)
            with self.subTest(field=field):
                self.assertLess(error, tolerance)
                self.assertLess(error, stock_error * 0.02)

    def test_wrong_slot_count_rejected(self):
        with self.assertRaises(ValueError):
            wrap_native_contained_box_faces(SimpleNamespace(ncon=1))

    def test_pair_major_packing_residency_and_single_stock_call(self):
        original = collision_driver._COLLISION_FUNC[(6, 6)]
        inputs = [collision_inputs(fixture(angles)) for angles in ((0.2, 0.3), (0.2, 0.0))]
        model = Model(jp.concatenate([item[1].geom_size for item in inputs]))
        data = Data(
            jp.concatenate([item[2].geom_xpos for item in inputs]),
            jp.concatenate([item[2].geom_xmat for item in inputs]),
        )
        geometry = jp.array([[0, 1], [2, 3]], jp.int32)
        stocks = [original(*item[1:]) for item in inputs]
        fields = tuple(jp.concatenate([stock[index] for stock in stocks]) for index in range(3))
        calls = []

        def stock(model, data, key, geometry):
            calls.append(True)
            return fields

        stock.ncon = 4
        wrapped = wrap_native_contained_box_faces(stock)
        self.assertEqual(wrapped.ncon, 4)
        leaves = jax.tree.leaves((model, data, geometry))
        before = [np.asarray(value).tobytes() for value in leaves]
        actual = jax.jit(wrapped)(model, data, None, geometry)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(
            np.asarray(actual[0]).reshape(2, 4) < 0,
            [[True, False, False, False], [True, True, False, False]],
        )
        for pair, item in enumerate(inputs):
            contacts = [contact for contact in item[0].contact if contact.dist < 0]
            active = np.asarray(actual[0])[pair * 4 : pair * 4 + 4] < 0
            self.assertLess(
                point_set_error(
                    np.asarray(actual[1])[pair * 4 : pair * 4 + 4][active],
                    np.array([contact.pos for contact in contacts]),
                ),
                2e-7,
            )
        np.testing.assert_array_equal(actual[2], fields[2])
        self.assertEqual(before, [np.asarray(value).tobytes() for value in leaves])
        self.assertTrue(
            all(device.platform == "cpu" for field in actual for device in field.devices())
        )

    def test_invalid_oblique_nonfinite_and_inactive_fallback(self):
        original = collision_driver._COLLISION_FUNC[(6, 6)]
        _, full_model, full_data, key, geometry = collision_inputs(fixture((0.2, 0.3)))
        model = Model(full_model.geom_size)
        data = Data(full_data.geom_xpos, full_data.geom_xmat)
        base = original(full_model, full_data, key, geometry)
        cases = (
            (model, data, base, jp.array([[-1, 1]], jp.int32)),
            (Model(jp.full_like(model.geom_size, -1)), data, base, geometry),
            (model, Data(jp.full_like(data.geom_xpos, jp.nan), data.geom_xmat), base, geometry),
            (model, Data(data.geom_xpos, jp.zeros_like(data.geom_xmat)), base, geometry),
            (model, data, (jp.full_like(base[0], jp.nan), base[1], base[2]), geometry),
            (model, data, (jp.ones_like(base[0]), base[1], base[2]), geometry),
            (
                model,
                data,
                (
                    base[0],
                    base[1],
                    jp.broadcast_to(
                        jp.array([[0.8, 0.6, 0.0], [0.0, 0.0, 1.0], [0.6, -0.8, 0.0]]), (4, 3, 3)
                    ),
                ),
                geometry,
            ),
        )
        for case_model, case_data, fields, case_geometry in cases:

            def stock(model, data, key, geometry):
                return fields

            stock.ncon = 4
            actual = jax.jit(wrap_native_contained_box_faces(stock))(
                case_model, case_data, None, case_geometry
            )
            for actual_field, expected_field in zip(actual, fields):
                np.testing.assert_array_equal(
                    np.asarray(actual_field).view(np.uint32),
                    np.asarray(expected_field).view(np.uint32),
                )

    def test_direct_import(self):
        self.assertTrue(
            callable(
                importlib.import_module(
                    "mjx_ascend.native_contained_box_faces"
                ).wrap_native_contained_box_faces
            )
        )


if __name__ == "__main__":
    unittest.main()
