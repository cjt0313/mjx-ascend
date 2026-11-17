import unittest
from unittest import mock

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import collision_driver, smooth
import numpy as np

from mjx_ascend import native_mesh_broadphase, staged_libero_step


class GroupMeshBroadphaseStagingTests(unittest.TestCase):
    def test_cpu_separation_contact_and_recontact_match_original_chunk_schedule(self):
        native = mujoco.MjModel.from_xml_string("""<mujoco>
            <option timestep=".002" gravity="0 0 0" cone="elliptic" solver="Newton"/>
            <asset><mesh name="cube"
            vertex="-.1 -.1 -.1 -.1 -.1 .1 -.1 .1 -.1 -.1 .1 .1 .1 -.1 -.1 .1 -.1 .1 .1 .1 -.1 .1 .1 .1"/></asset>
            <worldbody><geom type="mesh" mesh="cube"/>
            <body pos="1.5 .01 .015"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
            </worldbody></mujoco>""")
        model = mjx.put_model(native)
        initial = mjx.make_data(model)
        trajectories = []
        for precheck in (False, True):
            data = initial
            states = []
            with staged_libero_step.staged_kernels(
                collision_batch_size=2,
                mesh_broadphase_margin=0.0,
                group_mesh_broadphase=precheck,
                cache_contact_templates=True,
            ):
                for position in (1.5, 0.15, 1.5, 0.15):
                    data = data.replace(
                        qpos=data.qpos.at[0].set(position), qvel=jp.zeros_like(data.qvel)
                    )
                    data = mjx.step(model, data)
                    jax.block_until_ready(data)
                    states.append(data)
            trajectories.append(states)
        for expected, actual in zip(*trajectories):
            for field in ("qpos", "qvel", "qacc", "qfrc_constraint"):
                np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
            expected_active = np.asarray(expected.contact.dist) < np.asarray(
                expected.contact.includemargin
            )
            np.testing.assert_array_equal(
                np.asarray(actual.contact.dist) < np.asarray(actual.contact.includemargin),
                expected_active,
            )
            for field in ("dist", "pos", "frame", "geom"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(actual.contact, field))[expected_active],
                    np.asarray(getattr(expected.contact, field))[expected_active],
                )

    def test_invalid_flag_rejects_before_patching(self):
        original = smooth.kinematics
        for invalid in (None, 1, "yes"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                with staged_libero_step.staged_kernels(group_mesh_broadphase=invalid):
                    self.fail("invalid group option accepted")
            self.assertIs(smooth.kinematics, original)

    def test_missing_prerequisites_reject_before_patching(self):
        original = smooth.kinematics
        for options in ({}, {"collision_batch_size": 16}, {"mesh_broadphase_margin": 0.0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                with staged_libero_step.staged_kernels(group_mesh_broadphase=True, **options):
                    self.fail("missing prerequisite accepted")
            self.assertIs(smooth.kinematics, original)

    def test_supported_families_receive_chunk_scheduler_and_restore_on_error(self):
        def collision(model, data, key, geometry):
            return ()

        collision.ncon = 4
        types = ((6, 7), (7, 7), (0, 7), (5, 7))
        table = collision_driver._COLLISION_FUNC
        wrapped = []

        def wrap(function, ncon, margin):
            self.assertEqual(ncon, 4)
            self.assertEqual(margin, 0.01)
            self.assertTrue(hasattr(function, "__wrapped__"))
            wrapped.append(function)
            return function

        with mock.patch.dict(table, dict.fromkeys(types, collision), clear=True):
            with mock.patch.object(
                native_mesh_broadphase, "wrap_group_broadphase", side_effect=wrap, create=True
            ):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    with staged_libero_step.staged_kernels(
                        group_mesh_broadphase=True,
                        collision_batch_size=16,
                        mesh_broadphase_margin=0.01,
                    ):
                        self.assertEqual(len(wrapped), 2)
                        self.assertIs(table[(6, 7)], wrapped[0])
                        self.assertIs(table[(7, 7)], wrapped[1])
                        raise RuntimeError("stop")
            for key in types:
                self.assertIs(table[key], collision)

    def test_default_never_prepares_group_wrapper(self):
        with mock.patch.object(
            native_mesh_broadphase, "wrap_group_broadphase", create=True
        ) as wrap:
            with staged_libero_step.staged_kernels(
                collision_batch_size=16, mesh_broadphase_margin=0.0
            ):
                wrap.assert_not_called()


if __name__ == "__main__":
    unittest.main()
