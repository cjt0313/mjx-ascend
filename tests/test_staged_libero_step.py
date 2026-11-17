"""CPU acceptance tests for device-preserving staged MJX scheduling."""

import unittest
from unittest import mock

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import collision_driver, constraint, forward, sensor, smooth, solver
import numpy as np


from mjx_ascend.libero_native_adapter import register_cylinder_mesh
from mjx_ascend.staged_libero_step import _observe_timing, staged_kernels


STAGES = (
    (smooth, "kinematics"),
    (smooth, "com_pos"),
    (smooth, "camlight"),
    (smooth, "tendon"),
    (smooth, "crb"),
    (smooth, "factor_m"),
    (smooth, "transmission"),
    (constraint, "make_constraint"),
    (forward, "fwd_velocity"),
    (forward, "fwd_actuation"),
    (forward, "fwd_acceleration"),
    (forward, "euler"),
    (sensor, "sensor_pos"),
    (sensor, "sensor_vel"),
    (sensor, "sensor_acc"),
    (solver, "solve"),
)


class StagedKernelsTests(unittest.TestCase):
    def test_timing_observer_failure_is_visible(self):
        def observer(record):
            raise RuntimeError("cannot record timing")

        operation = _observe_timing(lambda model, data: jp.ones(1), "unit", observer)
        with self.assertRaisesRegex(RuntimeError, "cannot record timing"):
            operation(None, None)

    def test_factored_elliptic_registration_restores_on_failure(self):
        original = solver._update_gradient
        from mjx_ascend.native_elliptic_hessian import native_update_gradient

        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            with staged_kernels(factored_elliptic_hessian=True):
                self.assertIs(solver._update_gradient, native_update_gradient)
                raise RuntimeError("intentional failure")
        self.assertIs(solver._update_gradient, original)

    def test_factored_elliptic_cpu_contact_steps_match_stock(self):
        native = mujoco.MjModel.from_xml_string("""
            <mujoco><option timestep="0.002" integrator="Euler" cone="elliptic" solver="Newton"/>
              <worldbody><geom type="plane" size="2 2 .1"/>
                <body pos="0 0 .095"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
              </worldbody></mujoco>""")
        with jax.default_device(jax.devices("cpu")[0]):
            model = mjx.put_model(native)
            initial = mjx.make_data(model).replace(
                qvel=jp.array([0.1, -0.05, -0.2, 0.02, -0.03, 0.01])
            )
            expected = initial
            states = []
            for _ in range(3):
                expected = jax.jit(mjx.step)(model, expected)
                jax.block_until_ready(expected)
                states.append(expected)
            actual = initial
            with staged_kernels(factored_elliptic_hessian=True):
                for expected in states:
                    actual = mjx.step(model, actual)
                    jax.block_until_ready(actual)
                    for field in ("qpos", "qvel", "qacc", "qfrc_constraint"):
                        np.testing.assert_allclose(
                            getattr(actual, field), getattr(expected, field), rtol=1e-5, atol=1e-5
                        )

    def test_mesh_chunks_match_stock_cpu_contacts_away_from_ties(self):
        bodies = "".join(
            f'<body pos="{index * 0.15} {index * 0.011} {index * 0.007}" '
            f'euler="{index * 0.017} {index * 0.023} {index * 0.031}">'
            '<freejoint/><geom type="mesh" mesh="cube"/></body>'
            for index in range(4)
        )
        native = mujoco.MjModel.from_xml_string(f"""<mujoco><asset><mesh name="cube"
            vertex="-.1 -.1 -.1 -.1 -.1 .1 -.1 .1 -.1 -.1 .1 .1 .1 -.1 -.1 .1 -.1 .1 .1 .1 -.1 .1 .1 .1"/>
            </asset><worldbody>{bodies}</worldbody></mujoco>""")
        model = mjx.put_model(native)
        data = smooth.kinematics(model, mjx.make_data(model))
        reference = jax.jit(collision_driver.collision)(model, data)
        with staged_kernels(collision_batch_size=2):
            actual = collision_driver.collision(model, data)
        self.assertEqual(actual.ncon, reference.ncon)
        for field in ("dist", "pos", "frame", "geom"):
            values = np.asarray(getattr(actual.contact, field))
            self.assertTrue(np.isfinite(values).all(), field)
            np.testing.assert_allclose(
                values, getattr(reference.contact, field), rtol=2e-5, atol=2e-6
            )

    def test_collision_chunks_keep_all_pair_contact_order_and_device(self):
        traced_batch_sizes = []

        def collision(model, data, key, geom):
            traced_batch_sizes.append(geom.shape[0])
            distances = jp.repeat(geom[:, 0].astype(jp.float32), 2)
            return (
                distances,
                jp.broadcast_to(distances[:, None], (distances.size, 3)),
                jp.broadcast_to(distances[:, None, None], (distances.size, 3, 3)),
            )

        collision.ncon = 2
        geometry = jp.arange(10, dtype=jp.int32).reshape(5, 2)
        expected = collision(None, None, None, geometry)
        traced_batch_sizes.clear()
        table = collision_driver._COLLISION_FUNC
        with mock.patch.dict(table, {(6, 7): collision}, clear=True):
            with staged_kernels(collision_batch_size=2):
                operation = table[(6, 7)]
                self.assertEqual(operation.ncon, 2)
                actual = operation(None, None, None, geometry)
                for value, reference in zip(actual, expected):
                    np.testing.assert_array_equal(value, reference)
                    self.assertEqual(value.devices(), geometry.devices())
                self.assertEqual(sorted(traced_batch_sizes), [1, 2])
            self.assertIs(table[(6, 7)], collision)

    def test_invalid_collision_batch_rejects_without_patching(self):
        original = smooth.kinematics
        for batch_size in (0, -1):
            with self.assertRaises(ValueError):
                with staged_kernels(collision_batch_size=batch_size):
                    self.fail("Invalid collision batch was accepted")
            self.assertIs(smooth.kinematics, original)

    def test_three_euler_steps_match_outer_jit_on_cpu(self):
        native_model = mujoco.MjModel.from_xml_string("""
            <mujoco>
              <option timestep="0.002" integrator="Euler" cone="elliptic"
                      solver="Newton" iterations="8"/>
              <worldbody>
                <geom type="plane" size="1 1 0.1"/>
                <body pos="0 0 0.095">
                  <freejoint/>
                  <geom type="sphere" size="0.1" mass="1"/>
                  <site name="center"/>
                </body>
              </worldbody>
              <sensor>
                <framepos objtype="site" objname="center"/>
                <velocimeter site="center"/>
                <accelerometer site="center"/>
              </sensor>
            </mujoco>
        """)
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            model = jax.device_put(mjx.put_model(native_model), cpu)
            initial = mjx.make_data(model).replace(
                qvel=jp.array([0.1, -0.05, -0.2, 0.02, -0.03, 0.01])
            )
            normal_step = jax.jit(mjx.step)
            reference = initial
            expected = []
            for _ in range(3):
                reference = normal_step(model, reference)
                jax.block_until_ready(reference)
                expected.append(reference)
            staged = initial
            with staged_kernels():
                for reference in expected:
                    staged = mjx.step(model, staged)
                    jax.block_until_ready(staged)
                    self.assertGreater(staged.ncon, 0)
                    self.assertLess(float(staged.contact.dist[0]), 0)
                    for field in ("qpos", "qvel", "time", "sensordata"):
                        np.testing.assert_allclose(
                            getattr(staged, field),
                            getattr(reference, field),
                            rtol=2e-5,
                            atol=2e-6,
                            err_msg=field,
                        )
                    np.testing.assert_allclose(
                        staged.contact.dist, reference.contact.dist, rtol=2e-5, atol=2e-6
                    )
                    for leaf in jax.tree_util.tree_leaves(staged):
                        if isinstance(leaf, jax.Array):
                            self.assertEqual(leaf.devices(), {cpu})

    def test_exact_jit_boundaries_and_normal_restoration(self):
        originals = {(module, name): getattr(module, name) for module, name in STAGES}
        orchestration = (
            forward.step,
            forward.forward,
            forward.fwd_position,
            collision_driver.collision,
        )
        table = collision_driver._COLLISION_FUNC
        original_table = dict(table)
        with staged_kernels():
            for (module, name), original in originals.items():
                compiled = getattr(module, name)
                self.assertIsNot(compiled, original)
                self.assertIs(compiled.__wrapped__, original)
                self.assertTrue(callable(compiled.lower))
            self.assertEqual(
                orchestration,
                (forward.step, forward.forward, forward.fwd_position, collision_driver.collision),
            )
            self.assertIs(collision_driver._COLLISION_FUNC, table)
            self.assertEqual(table.keys(), original_table.keys())
            for key, original in original_table.items():
                self.assertIs(table[key].__wrapped__, original)
                self.assertEqual(table[key].ncon, original.ncon)
        for (module, name), original in originals.items():
            self.assertIs(getattr(module, name), original)
        for key, original in original_table.items():
            self.assertIs(table[key], original)

    def test_exception_restores_registered_adapter_and_stock_identities(self):
        table = collision_driver._COLLISION_FUNC
        stock = dict(table)
        originals = {(module, name): getattr(module, name) for module, name in STAGES}
        with register_cylinder_mesh():
            registered = dict(table)
            with self.assertRaisesRegex(RuntimeError, "test exception"):
                with staged_kernels():
                    for key, original in registered.items():
                        self.assertIs(table[key].__wrapped__, original)
                        self.assertEqual(table[key].ncon, original.ncon)
                    raise RuntimeError("test exception")
            self.assertIs(collision_driver._COLLISION_FUNC, table)
            self.assertEqual(table.keys(), registered.keys())
            for key, original in registered.items():
                self.assertIs(table[key], original)
            for (module, name), original in originals.items():
                self.assertIs(getattr(module, name), original)
        self.assertEqual(table.keys(), stock.keys())
        for key, original in stock.items():
            self.assertIs(table[key], original)

    def test_partial_setup_failure_restores_identities(self):
        originals = {(module, name): getattr(module, name) for module, name in STAGES}
        table = collision_driver._COLLISION_FUNC
        original_table = dict(table)
        real_jit = jax.jit
        calls = 0

        def failing_jit(function, **kwargs):
            nonlocal calls
            calls += 1
            if calls == len(STAGES) + 2:
                raise RuntimeError("setup failure")
            return real_jit(function, **kwargs)

        with mock.patch.object(jax, "jit", side_effect=failing_jit):
            with self.assertRaisesRegex(RuntimeError, "setup failure"):
                with staged_kernels():
                    self.fail("setup must fail before entering the context")
        for (module, name), original in originals.items():
            self.assertIs(getattr(module, name), original)
        self.assertIs(collision_driver._COLLISION_FUNC, table)
        self.assertEqual(table.keys(), original_table.keys())
        for key, original in original_table.items():
            self.assertIs(table[key], original)


if __name__ == "__main__":
    unittest.main()
