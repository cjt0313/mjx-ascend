"""独立场景完整性、物理调用隔离和状态误差合同。"""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from mjx_ascend.backend import native_entrypoint_guard, select_device
from mjx_ascend.capture import config_hash, file_hash, load_scene, model_config
from mjx_ascend.validation import arrays_bitwise_equal, compare_state
from mjx_ascend.__main__ import THRESHOLDS


class ReplayTests(unittest.TestCase):
    def test_bitwise_comparison_rejects_signed_zero_and_dtype_changes(self):
        positive = np.array([0.0], dtype=np.float32)
        self.assertTrue(arrays_bitwise_equal(positive, positive.copy()))
        self.assertFalse(arrays_bitwise_equal(positive, -positive))
        self.assertFalse(arrays_bitwise_equal(positive, positive.astype(np.float64)))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
        mujoco.mj_getState(model, data, state, spec)
        binary = self.root / "model.mjb"
        mujoco.mj_saveModel(model, str(binary))
        (self.root / "model.mjb.gz").write_bytes(gzip.compress(binary.read_bytes()))
        np.savez(
            self.root / "replay.npz",
            initial_state=state,
            controls=np.zeros((2, model.nu), np.float32),
            initial_qpos=data.qpos,
            initial_qvel=data.qvel,
            initial_time=np.array(data.time),
        )
        self.manifest = {
            "mujoco_version": mujoco.__version__,
            "state_specification": spec,
            "model_sha256": file_hash(binary),
            "replay_sha256": file_hash(self.root / "replay.npz"),
            "model_config_hash": config_hash(model_config(model)),
        }
        self.write_manifest()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_load_and_repair_corrupt_cache(self):
        model, arrays, manifest = load_scene(self.root, self.root / "cache")
        self.assertEqual(model.nq, 7)
        self.assertEqual(arrays["controls"].dtype, np.float32)
        (self.root / "cache" / f"{manifest['model_sha256']}.mjb").write_bytes(b"corrupt")
        load_scene(self.root, self.root / "cache")

    def test_reject_replay_corruption(self):
        (self.root / "replay.npz").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Replay checksum"):
            load_scene(self.root, self.root / "cache")

    def test_reject_model_corruption(self):
        (self.root / "model.mjb.gz").write_bytes(gzip.compress(b"corrupt"))
        with self.assertRaisesRegex(ValueError, "Model checksum"):
            load_scene(self.root, self.root / "cache")

    def test_reject_config_changes(self):
        self.manifest["model_config_hash"] = "wrong"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            load_scene(self.root, self.root / "cache")

    def test_native_guard_restores_after_rejection(self):
        original = mujoco.mj_step
        calls = {}
        with self.assertRaisesRegex(RuntimeError, "Native entrypoint forbidden"):
            with native_entrypoint_guard(calls):
                mujoco.mj_step(None, None)
        self.assertEqual(calls["mj_step"], 1)
        self.assertIs(mujoco.mj_step, original)

    def test_quaternion_sign_and_velocity_error(self):
        model, arrays, _ = load_scene(self.root, self.root / "cache")
        position, velocity = arrays["initial_qpos"].copy(), arrays["initial_qvel"].copy()
        reference = {"qpos": position.copy(), "qvel": velocity.copy(), "time": 0.0}
        position[3:] *= -1
        self.assertTrue(
            compare_state(model, position, velocity, 0.0, reference, THRESHOLDS)["within_tolerance"]
        )
        velocity[0] += 1e-3
        self.assertFalse(
            compare_state(model, position, velocity, 0.0, reference, THRESHOLDS)["within_tolerance"]
        )

    def test_missing_npu_rejects(self):
        with self.assertRaisesRegex(RuntimeError, "fallback is forbidden"):
            select_device([], "npu", 0)
