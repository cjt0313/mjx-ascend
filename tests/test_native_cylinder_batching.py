"""Scheduling-only regression coverage for the native cylinder adapter."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "mjx_ascend"


class CylinderBatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        specification = importlib.util.spec_from_file_location(
            "mjx_ascend.cylinder_batch_adapter", ROOT / "libero_native_adapter.py"
        )
        cls.adapter = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cls.adapter)

    def test_default_and_one_keep_original_callable(self):
        from mujoco.mjx._src import collision_driver, types

        key = (types.GeomType.CYLINDER, types.GeomType.BOX)
        original = collision_driver._COLLISION_FUNC.get(key)
        for batch_size in (None, 1):
            with self.adapter.register_cylinder_mesh(batch_size=batch_size):
                self.assertIs(collision_driver._COLLISION_FUNC[key], self.adapter.cylinder_mesh)
            self.assertIs(collision_driver._COLLISION_FUNC.get(key), original)

    def test_invalid_sizes_rejected_before_registration(self):
        from mujoco.mjx._src import collision_driver

        original = dict(collision_driver._COLLISION_FUNC)
        for batch_size in (0, -1, 2.5, True, "2"):
            with self.assertRaises(ValueError):
                with self.adapter.register_cylinder_mesh(batch_size=batch_size):
                    self.fail("Invalid batch size entered registration")
            self.assertEqual(collision_driver._COLLISION_FUNC, original)

    def test_nested_registration_and_exception_restore(self):
        from mujoco.mjx._src import collision_driver, types

        keys = (
            (types.GeomType.CYLINDER, types.GeomType.BOX),
            (types.GeomType.CYLINDER, types.GeomType.MESH),
        )
        original = dict(collision_driver._COLLISION_FUNC)
        with self.adapter.register_cylinder_mesh(batch_size=2):
            outer = collision_driver._COLLISION_FUNC[keys[0]]
            self.assertEqual(outer.ncon, 1)
            self.assertIs(outer, collision_driver._COLLISION_FUNC[keys[1]])
            with self.assertRaisesRegex(RuntimeError, "sentinel"):
                with self.adapter.register_cylinder_mesh(batch_size=4):
                    self.assertIsNot(collision_driver._COLLISION_FUNC[keys[0]], outer)
                    raise RuntimeError("sentinel")
            self.assertIs(collision_driver._COLLISION_FUNC[keys[0]], outer)
        self.assertEqual(collision_driver._COLLISION_FUNC, original)

    def test_bounded_map_receives_opt_in_size(self):
        from mujoco.mjx._src import collision_driver, types

        key = SimpleNamespace(types=(types.GeomType.CYLINDER, types.GeomType.BOX))
        with mock.patch.object(self.adapter.jax.lax, "map", return_value="mapped") as operation:
            with self.adapter.register_cylinder_mesh(batch_size=4):
                result = collision_driver._COLLISION_FUNC[key.types](None, None, key, "pairs")
            self.assertEqual(result, "mapped")
            self.assertEqual(operation.call_args.kwargs, {"batch_size": 4})

    def test_real_box_mesh_tail_equivalence_and_inputs_unchanged(self):
        import jax
        import jax.numpy as jp
        from mujoco.mjx._src import collision_driver, types

        vertices = np.array(
            [
                [first, second, third]
                for first in (-0.5, 0.5)
                for second in (-0.5, 0.5)
                for third in (-0.5, 0.5)
            ],
            np.float32,
        )
        model = SimpleNamespace(
            geom_size=jp.array([[0.25, 0.3, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            mesh_vertadr=np.array([0]),
            mesh_vertnum=np.array([8]),
            mesh_vert=vertices,
        )
        pairs = jp.array([[0, 1], [0, 2], [0, 1]], dtype=jp.int32)
        for position in (1.0, 0.6):
            data = SimpleNamespace(
                geom_xpos=jp.array([[position, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.2, 0.0, 0.0]]),
                geom_xmat=jp.broadcast_to(jp.eye(3), (3, 3, 3)),
            )
            snapshots = [
                np.asarray(value).copy()
                for value in (pairs, data.geom_xpos, data.geom_xmat, model.geom_size)
            ]
            for other_type in (types.GeomType.BOX, types.GeomType.MESH):
                key = SimpleNamespace(types=(types.GeomType.CYLINDER, other_type), data_ids=(-1, 0))
                serial = jax.jit(lambda value: self.adapter.cylinder_mesh(model, data, key, value))(
                    pairs
                )
                for batch_size in (2, 4, 8):
                    with self.adapter.register_cylinder_mesh(batch_size=batch_size):
                        function = collision_driver._COLLISION_FUNC[key.types]
                        result = jax.jit(lambda value: function(model, data, key, value))(pairs)
                    for expected, actual in zip(serial, result):
                        np.testing.assert_allclose(
                            actual, expected, atol=1e-5, rtol=1e-5, equal_nan=True
                        )
                        if position == 0.6:
                            self.assertTrue(np.isnan(actual[0]).all())
            for expected, actual in zip(
                snapshots, (pairs, data.geom_xpos, data.geom_xmat, model.geom_size)
            ):
                self.assertEqual(expected.tobytes(), np.asarray(actual).tobytes())


if __name__ == "__main__":
    unittest.main()
