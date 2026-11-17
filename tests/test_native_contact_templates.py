import dataclasses
from unittest import mock
import unittest

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import collision_driver
import numpy as np

from mjx_ascend.native_contact_templates import prepare_contact_templates


def fixture(custom="", empty=False):
    bodies = (
        ""
        if empty
        else """
      <geom name="floor" type="plane" size="3 3 .1" condim="3"/>
      <body pos="0 0 .08"><freejoint/>
        <geom name="a" type="sphere" size=".1" condim="1" priority="2"
              friction=".2 .03 .004" solmix="0" solref=".03 .8"/>
      </body>
      <body pos=".19 0 .09"><freejoint/>
        <geom name="b" type="sphere" size=".1" condim="6" priority="0"
              friction=".7 .02 .009" solmix="0" solref="-.4 -.2"/>
      </body>
      <body pos="1 0 .06"><freejoint/>
        <geom name="c" type="capsule" size=".07 .15" condim="4"
              solmix=".3" solref=".04 .7"/>
      </body>
      <body pos="2 0 .08"><freejoint/>
        <geom name="d" type="sphere" size=".1" condim="3"
              solmix=".7" solref=".05 .9"/>
      </body>"""
    )
    contact = (
        ""
        if empty
        else """<contact><pair geom1="a" geom2="b" condim="6"
      friction="0 .4 .02 .003 .004" margin=".012" gap=".002"
      solref=".06 .7" solreffriction=".08 .9" solimp=".8 .9 .002 .6 3"/></contact>"""
    )
    native = mujoco.MjModel.from_xml_string(
        f'<mujoco><option gravity="0 0 0"/>{custom}<worldbody>{bodies}</worldbody>'
        f"{contact}</mujoco>"
    )
    host = mujoco.MjData(native)
    mujoco.mj_forward(native, host)
    model = mjx.put_model(native)
    data = mjx.make_data(model) if custom else mjx.put_data(native, host)
    return native, host, model, data


def contact_bytes(contact):
    return {
        field.name: (
            np.asarray(getattr(contact, field.name)).dtype.str,
            np.asarray(getattr(contact, field.name)).shape,
            np.asarray(getattr(contact, field.name)).tobytes(),
        )
        for field in dataclasses.fields(contact)
        if getattr(contact, field.name) is not None
    }


class NativeContactTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native, cls.host, cls.model, cls.data = fixture()

    def assert_stock(self, model, data, prepared):
        expected = collision_driver.collision(model, data)
        actual = prepared(model, data)
        self.assertEqual(contact_bytes(actual.contact), contact_bytes(expected.contact))
        for field in dataclasses.fields(data):
            if field.name != "contact":
                self.assertIs(getattr(actual, field.name), getattr(data, field.name))
        for leaf in jax.tree_util.tree_leaves(actual.contact):
            self.assertEqual(leaf.devices(), data.geom_xpos.devices())
        return actual

    def test_stock_bytes_order_counts_and_input_residency(self):
        before = [
            np.asarray(leaf).tobytes()
            for leaf in jax.tree_util.tree_leaves((self.model, self.data))
        ]
        layout = contact_bytes(self.data.contact)
        prepared = prepare_contact_templates(self.model, self.data)
        self.assert_stock(self.model, self.data, prepared)
        self.assertEqual(
            before,
            [
                np.asarray(leaf).tobytes()
                for leaf in jax.tree_util.tree_leaves((self.model, self.data))
            ],
        )
        self.assertEqual(layout, contact_bytes(self.data.contact))
        actual = prepared(self.model, self.data)
        for name in (
            "includemargin",
            "friction",
            "solref",
            "solreffriction",
            "solimp",
            "geom1",
            "geom2",
            "geom",
            "dim",
            "efc_address",
        ):
            self.assertIs(getattr(actual.contact, name), getattr(prepared.template, name))
        self.assertIsNone(prepared.template.dist)
        self.assertIsNone(prepared.template.pos)
        self.assertIsNone(prepared.template.frame)
        self.assertEqual(sorted(set(self.data.contact.dim)), [1, 3, 4, 6])

    def test_priority_explicit_pair_and_mixed_parameters(self):
        prepared = prepare_contact_templates(self.model, self.data)
        contact = self.assert_stock(self.model, self.data, prepared).contact
        geoms = np.asarray(contact.geom)
        explicit = np.flatnonzero(np.all(geoms == [1, 2], axis=1))
        self.assertEqual(len(explicit), 1)
        np.testing.assert_array_equal(
            contact.friction[explicit[0]],
            np.array([mujoco.mjMINVAL, 0.4, 0.02, 0.003, 0.004], np.float32),
        )
        np.testing.assert_array_equal(
            contact.solreffriction[explicit[0]], np.array([0.08, 0.9], np.float32)
        )
        priority = np.flatnonzero(np.all(geoms == [0, 1], axis=1))[0]
        np.testing.assert_array_equal(
            contact.friction[priority], np.array([0.2, 0.2, 0.03, 0.004, 0.004], np.float32)
        )
        direct = np.flatnonzero(np.all(geoms == [0, 2], axis=1))[0]
        np.testing.assert_array_equal(contact.solref[direct], np.array([-0.4, -0.2], np.float32))
        mixed = np.flatnonzero(np.all(geoms == [0, 4], axis=1))[0]
        mix = np.float32(1.0) / np.float32(1.7)
        np.testing.assert_allclose(
            contact.solref[mixed],
            mix * np.array([0.02, 1.0]) + (1 - mix) * np.array([0.05, 0.9]),
            rtol=2e-7,
        )

    def test_motion_separation_and_recontact_do_not_pollute_template(self):
        prepared = prepare_contact_templates(self.model, self.data)
        template = contact_bytes(prepared.template)
        states = []
        for height in (0.08, 2.0, 0.07):
            host = mujoco.MjData(self.native)
            host.qpos[:] = self.host.qpos
            host.qpos[2] = height
            mujoco.mj_forward(self.native, host)
            data = mjx.put_data(self.native, host)
            result = self.assert_stock(self.model, data, prepared)
            states.append(np.asarray(result.contact.dist).copy())
            self.assertEqual(template, contact_bytes(prepared.template))
            active = np.asarray(result.contact.dist) < np.asarray(result.contact.includemargin)
            native_pairs = {(entry.geom1, entry.geom2) for entry in host.contact}
            actual_pairs = {tuple(pair) for pair in np.asarray(result.contact.geom)[active]}
            self.assertEqual(actual_pairs, native_pairs)
            for entry in host.contact:
                candidates = np.flatnonzero(
                    active
                    & np.all(np.asarray(result.contact.geom) == [entry.geom1, entry.geom2], axis=1)
                )
                index = candidates[
                    np.argmin(
                        np.linalg.norm(
                            np.asarray(result.contact.pos)[candidates] - entry.pos, axis=1
                        )
                    )
                ]
                np.testing.assert_allclose(result.contact.dist[index], entry.dist, atol=2e-7)
                np.testing.assert_allclose(result.contact.pos[index], entry.pos, atol=2e-7)
                np.testing.assert_allclose(
                    result.contact.frame[index, 0], entry.frame[:3], atol=2e-7
                )
        self.assertFalse(np.array_equal(states[0], states[1]))
        self.assertFalse(np.array_equal(states[1], states[2]))

    def test_preparation_only_once_and_original_unexpanded_calls(self):
        original_groups = collision_driver._contact_groups(self.model, self.data)
        originals = dict(collision_driver._COLLISION_FUNC)
        calls = []
        wrappers = {}
        for types in {key.types for key in original_groups}:

            def wrapped(model, data, key, geom, types=types):
                calls.append((key, geom))
                return originals[types](model, data, key, geom)

            wrapped.ncon = originals[types].ncon
            wrappers[types] = wrapped
        with mock.patch.dict(collision_driver._COLLISION_FUNC, wrappers):
            with mock.patch.object(
                collision_driver, "_contact_groups", wraps=collision_driver._contact_groups
            ) as groups:
                prepared = prepare_contact_templates(self.model, self.data)
                self.assertEqual(groups.call_count, 1)
                with mock.patch.object(jp, "repeat", side_effect=AssertionError("step repeat")):
                    prepared(self.model, self.data)
                    prepared(self.model, self.data)
                self.assertEqual(groups.call_count, 1)
        self.assertEqual([key for key, _ in calls], list(original_groups) * 2)
        for key, geom in calls:
            np.testing.assert_array_equal(geom, original_groups[key].geom)

    def test_independent_immutable_templates(self):
        first = prepare_contact_templates(self.model, self.data)
        second = prepare_contact_templates(self.model, self.data)
        self.assertIsNot(first.template, second.template)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.template.dist = self.data.contact.dist
        with self.assertRaises(ValueError):
            first.template.dim[0] = 9
        with self.assertRaises(ValueError):
            first.template.efc_address[0] = 9
        with self.assertRaises(ValueError):
            first.template.dim.setflags(write=True)
        first(self.model, self.data)
        self.assertIsNone(second.template.dist)

    def test_foreign_replaced_model_rejected(self):
        prepared = prepare_contact_templates(self.model, self.data)
        with self.assertRaisesRegex(ValueError, "model|prepar"):
            prepared(self.model.replace(geom_friction=self.model.geom_friction), self.data)

    def test_mutable_model_metadata_rejected(self):
        _, _, model, data = fixture()
        model = model.replace(geom_priority=model.geom_priority.copy())
        prepared = prepare_contact_templates(model, data)
        model.geom_priority[1] += 1
        with self.assertRaisesRegex(ValueError, "model|metadata|mutat"):
            prepared(model, data)

    def test_replaced_jax_leaf_on_same_model_rejected(self):
        _, _, model, data = fixture()
        prepared = prepare_contact_templates(model, data)
        object.__setattr__(model, "geom_friction", model.geom_friction + 0.1)
        with self.assertRaisesRegex(ValueError, "model|mutat|prepar"):
            prepared(model, data)

    def test_layout_bytes_shapes_and_ncon_rejected(self):
        prepared = prepare_contact_templates(self.model, self.data)
        for name in ("dim", "efc_address"):
            for change in ("value", "shape", "dtype"):
                values = getattr(self.data.contact, name).copy()
                if change == "value":
                    values[0] += 1
                elif change == "shape":
                    values = values[:-1]
                else:
                    values = values.astype(np.int32 if values.dtype == np.int64 else np.int64)
                changed = self.data.replace(contact=self.data.contact.replace(**{name: values}))
                with self.subTest(name=name, change=change):
                    with self.assertRaisesRegex(ValueError, "layout|contact"):
                        prepared(self.model, changed)
        with self.assertRaisesRegex(ValueError, "layout|contact"):
            prepared(self.model, self.data.replace(ncon=self.data.ncon - 1))
        with self.assertRaisesRegex(ValueError, "layout|geometry"):
            prepared(self.model, self.data.replace(geom_xpos=self.data.geom_xpos[:-1]))

    def test_mutable_layout_rejected(self):
        _, _, model, data = fixture()
        prepared = prepare_contact_templates(model, data)
        values = data.contact.dim.copy()
        data = data.replace(contact=data.contact.replace(dim=values))
        values[0] += 1
        with self.assertRaisesRegex(ValueError, "layout|contact"):
            prepared(model, data)

    def test_registry_function_and_ncon_mutation_rejected(self):
        prepared = prepare_contact_templates(self.model, self.data)
        key = next(iter(collision_driver._contact_groups(self.model, self.data)))
        function = collision_driver._COLLISION_FUNC[key.types]
        replacement = mock.Mock(ncon=function.ncon)
        with mock.patch.dict(collision_driver._COLLISION_FUNC, {key.types: replacement}):
            with self.assertRaisesRegex(ValueError, "registry|collider"):
                prepared(self.model, self.data)
        with mock.patch.object(function, "ncon", function.ncon + 1):
            with self.assertRaisesRegex(ValueError, "registry|collider|ncon"):
                prepared(self.model, self.data)
        with mock.patch.object(function, "ncon", float(function.ncon)):
            with self.assertRaisesRegex(ValueError, "registry|collider|ncon"):
                prepared(self.model, self.data)

    def test_invalid_initial_layout_rejected(self):
        with self.assertRaisesRegex(ValueError, "layout|contact"):
            prepare_contact_templates(
                self.model,
                self.data.replace(contact=self.data.contact.replace(dim=self.data.contact.dim + 1)),
            )

    def test_invalid_initial_geometry_and_contact_shapes_rejected(self):
        with self.assertRaisesRegex(ValueError, "geometry|layout"):
            prepare_contact_templates(
                self.model, self.data.replace(geom_xpos=self.data.geom_xpos[:-1])
            )
        with self.assertRaisesRegex(ValueError, "geometry|layout"):
            prepare_contact_templates(
                self.model, self.data.replace(geom_xmat=self.data.geom_xmat[:, :, :2])
            )
        with self.assertRaisesRegex(ValueError, "contact|layout"):
            prepare_contact_templates(
                self.model,
                self.data.replace(
                    contact=self.data.contact.replace(pos=self.data.contact.pos[:, :2])
                ),
            )

    def test_deleted_registry_ncon_rejected_clearly(self):
        prepared = prepare_contact_templates(self.model, self.data)
        key = next(iter(collision_driver._contact_groups(self.model, self.data)))
        function = collision_driver._COLLISION_FUNC[key.types]
        original = function.ncon
        try:
            del function.ncon
            with self.assertRaisesRegex(ValueError, "registry|collider|ncon"):
                prepared(self.model, self.data)
        finally:
            function.ncon = original

    def test_nested_model_option_mutation_rejected(self):
        _, _, model, data = fixture()
        prepared = prepare_contact_templates(model, data)
        object.__setattr__(model.opt, "disableflags", mujoco.mjtDisableBit.mjDSBL_CONTACT)
        with self.assertRaisesRegex(ValueError, "model|mutat"):
            prepared(model, data)

    def test_numeric_caps_rejected_including_zero(self):
        for name in ("max_geom_pairs", "max_contact_points"):
            for count in (0, 1, 1000):
                with self.subTest(name=name, count=count):
                    _, _, model, data = fixture(
                        f'<custom><numeric name="{name}" data="{count}"/></custom>'
                    )
                    with self.assertRaisesRegex(ValueError, name):
                        prepare_contact_templates(model, data)

    def test_incompatible_version_rejected(self):
        with mock.patch.object(mujoco, "__version__", "3.3.0"):
            with self.assertRaisesRegex(ValueError, "3.2.7|version"):
                prepare_contact_templates(self.model, self.data)

    def test_empty_contacts(self):
        _, _, model, data = fixture(empty=True)
        prepared = prepare_contact_templates(model, data)
        self.assertEqual(data.ncon, 0)
        self.assertIs(prepared(model, data), data)

    def test_disabled_contacts(self):
        native, _, _, _ = fixture()
        native.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        model = mjx.put_model(native)
        data = mjx.make_data(model)
        prepared = prepare_contact_templates(model, data)
        self.assertIs(prepared(model, data), collision_driver.collision(model, data))

    def test_zero_solmix_edge_cases_match_stock(self):
        model = self.model.replace(geom_solmix=jp.zeros_like(self.model.geom_solmix))
        prepared = prepare_contact_templates(model, self.data)
        self.assert_stock(model, self.data, prepared)

    def test_no_native_cpu_physics_entrypoint_at_execution(self):
        prepared = prepare_contact_templates(self.model, self.data)
        with mock.patch.object(mujoco, "mj_collision", side_effect=AssertionError("CPU collision")):
            with mock.patch.object(mujoco, "mj_forward", side_effect=AssertionError("CPU forward")):
                with mock.patch.object(mujoco, "mj_step", side_effect=AssertionError("CPU step")):
                    prepared(self.model, self.data)

    def test_real_contained_face_wrapper_full_contact_equivalence(self):
        from mjx_ascend.native_contained_box_faces import wrap_native_contained_box_faces

        native = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
          <geom type="box" size=".8 .8 .1"/>
          <body pos="0 0 .207"><freejoint/>
            <geom type="box" size=".07 .09 .11" mass="1"/>
          </body></worldbody></mujoco>""")
        key = (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_BOX)
        wrapped = wrap_native_contained_box_faces(collision_driver._COLLISION_FUNC[key])
        with mock.patch.dict(collision_driver._COLLISION_FUNC, {key: wrapped}):
            model = mjx.put_model(native)
            host = mujoco.MjData(native)
            mujoco.mj_forward(native, host)
            data = mjx.put_data(native, host)
            prepared = prepare_contact_templates(model, data)
            contact = self.assert_stock(model, data, prepared).contact
            active = np.asarray(contact.dist) < np.asarray(contact.includemargin)
            self.assertEqual(int(active.sum()), len(host.contact))
            self.assertEqual(len(host.contact), 4)
            for entry in host.contact:
                candidates = np.flatnonzero(active)
                index = candidates[
                    np.argmin(
                        np.linalg.norm(np.asarray(contact.pos)[candidates] - entry.pos, axis=1)
                    )
                ]
                np.testing.assert_allclose(contact.pos[index], entry.pos, atol=2e-7)
                np.testing.assert_allclose(contact.dist[index], entry.dist, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
