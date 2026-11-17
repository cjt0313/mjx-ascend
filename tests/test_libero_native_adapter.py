import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "mjx_ascend"


class NativeAdapterTests(unittest.TestCase):
    def load_adapter(self):
        path = ROOT / "libero_native_adapter.py"
        self.assertTrue(path.is_file(), "Experiment-local native adapter required")
        specification = importlib.util.spec_from_file_location(
            "mjx_ascend.native_adapter_test", path
        )
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module

    def test_registration_restores_stock_collision_table(self):
        from mujoco.mjx._src import collision_driver, types

        adapter = self.load_adapter()
        key = (types.GeomType.CYLINDER, types.GeomType.MESH)
        prior = collision_driver._COLLISION_FUNC.get(key)
        with adapter.register_cylinder_mesh():
            self.assertEqual(collision_driver._COLLISION_FUNC[key].ncon, 1)
            self.assertEqual(
                collision_driver._COLLISION_FUNC[
                    (types.GeomType.CYLINDER, types.GeomType.BOX)
                ].ncon,
                1,
            )
        self.assertIs(collision_driver._COLLISION_FUNC.get(key), prior)

    def test_separated_query_retains_distance_and_contact_frame(self):
        import jax.numpy as jp

        adapter = self.load_adapter()
        vertices = np.array(
            [
                [first, second, third]
                for first in (-0.5, 0.5)
                for second in (-0.5, 0.5)
                for third in (-0.5, 0.5)
            ],
            dtype=np.float32,
        )
        distance, position, frame = adapter.separated_contact(
            jp.array([1.0, 0.0, 0.0]), jp.eye(3), jp.array([0.25, 0.3]), jp.asarray(vertices)
        )
        self.assertAlmostEqual(float(distance), 0.25, places=5)
        self.assertTrue(np.isfinite(position).all())
        np.testing.assert_allclose(np.asarray(frame) @ np.asarray(frame).T, np.eye(3), atol=1e-5)

    def test_penetration_is_not_silently_used_as_native_manifold(self):
        import jax.numpy as jp

        adapter = self.load_adapter()
        vertices = np.array(
            [
                [first, second, third]
                for first in (-0.5, 0.5)
                for second in (-0.5, 0.5)
                for third in (-0.5, 0.5)
            ],
            dtype=np.float32,
        )
        distance, position, frame = adapter.separated_contact(
            jp.array([0.6, 0.0, 0.0]), jp.eye(3), jp.array([0.25, 0.3]), jp.asarray(vertices)
        )
        self.assertTrue(np.isnan(distance))
        self.assertTrue(np.isnan(position).all())
        self.assertTrue(np.isnan(frame).all())

    def test_pair_batch_uses_bounded_map_and_preserves_pair_order(self):
        import jax
        import jax.numpy as jp
        from mujoco.mjx._src import types

        adapter = self.load_adapter()
        source = (ROOT / "libero_native_adapter.py").read_text()
        self.assertIn("jax.lax.map", source)
        model = SimpleNamespace(
            geom_size=jp.array([[0.25, 0.3, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        )
        data = SimpleNamespace(
            geom_xpos=jp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.2, 0.0, 0.0]]),
            geom_xmat=jp.broadcast_to(jp.eye(3), (3, 3, 3)),
        )
        key = SimpleNamespace(types=(types.GeomType.CYLINDER, types.GeomType.BOX))
        operation = jax.jit(lambda pairs: adapter.cylinder_mesh(model, data, key, pairs))
        distances, positions, frames = operation(jp.array([[0, 1], [0, 2]], dtype=jp.int32))
        np.testing.assert_allclose(np.asarray(distances), [0.25, 0.45], atol=1e-5)
        self.assertEqual(positions.shape, (2, 3))
        self.assertEqual(frames.shape, (2, 3, 3))


if __name__ == "__main__":
    unittest.main()
