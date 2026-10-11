"""S1-04 model contracts and independent scan reference checks."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tests.pose3d_fixtures import input_windows
from tests.test_pose_adapter import replay_files
from video_to_3d_motion.pose_adapter.exceptions import PoseAdapterError
from video_to_3d_motion.pose_adapter.windows import prepare_sequence_windows
from video_to_3d_motion.pose3d.backend import PoseMambaBackend, flip_h36m
from video_to_3d_motion.pose3d.config import ModelSettings, OutputSettings, Pose3DConfig, load_pose3d_config
from video_to_3d_motion.pose3d.exceptions import Pose3DError
from video_to_3d_motion.pose3d.vendor.scan import selective_scan_torch, CrossScan_plus_poselimbs, CrossMerge_plus_poselimbs

ROOT = Path(__file__).resolve().parents[1]


class ScanTests(unittest.TestCase):
    def test_grouped_scan_matches_independent_serial_recurrence(self):
        rng = np.random.default_rng(47)
        for length, chunk in ((1, 1), (7, 3), (257, 64), (519, 256)):
            with self.subTest(length=length, chunk=chunk):
                u = rng.normal(size=(2, 6, length)).astype(np.float32)
                delta = rng.normal(size=u.shape).astype(np.float32)
                A = -rng.uniform(.1, 20, size=(6, 3)).astype(np.float32)
                B = rng.normal(size=(2, 2, 3, length)).astype(np.float32)
                C = rng.normal(size=B.shape).astype(np.float32)
                D = rng.normal(size=6).astype(np.float32)
                bias = rng.normal(size=6).astype(np.float32)
                # NumPy float64 reference, directly advancing every state.
                dt = np.logaddexp(0, delta.astype(np.float64) + bias[None, :, None])
                state = np.zeros((2, 6, 3))
                expected = np.empty((2, 6, length))
                for t in range(length):
                    expanded_B = np.repeat(B[..., t], 3, axis=1)
                    expanded_C = np.repeat(C[..., t], 3, axis=1)
                    state = np.exp(dt[..., t, None] * A) * state + dt[..., t, None] * expanded_B * u[..., t, None]
                    expected[..., t] = np.sum(expanded_C * state, axis=-1) + D * u[..., t]
                with torch.inference_mode():
                    actual = selective_scan_torch(*(torch.from_numpy(v) for v in (u, delta, A, B, C, D, bias)), chunk_size=chunk)
                np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-5, atol=2e-5)

    def test_scan_without_skip_or_bias_handles_strong_decay(self):
        u = torch.ones(1, 4, 531)
        dt = torch.full_like(u, 100)
        A = -torch.ones(4, 2)
        B = torch.ones(1, 2, 2, 531)
        with torch.inference_mode():
            result = selective_scan_torch(u, dt, A, B, B, delta_softplus=False, chunk_size=64)
        self.assertTrue(torch.isfinite(result).all())
        np.testing.assert_allclose(result.numpy(), 200)

    def test_four_directions_and_merge_preserve_official_limb_scan(self):
        x = torch.arange(2 * 3 * 17, dtype=torch.float32).reshape(1, 2, 3, 17)
        scans = CrossScan_plus_poselimbs.apply(x)
        indices = [0, 0, 1, 2, 3, 0, 4, 5, 6, 8, 11, 12, 13, 8, 14, 15, 16]
        np.testing.assert_array_equal(scans[:, 0], (x + x[..., indices]).flatten(2))
        np.testing.assert_array_equal(scans[:, 1], x.transpose(2, 3).flatten(2))
        np.testing.assert_array_equal(scans[:, 2:], scans[:, :2].flip([-1]))
        merged = CrossMerge_plus_poselimbs.apply(scans.reshape(1, 4, 2, 3, 17))
        np.testing.assert_array_equal(merged.reshape_as(x), 4 * x + 2 * x[..., indices])



class InputAndModelTests(unittest.TestCase):
    def test_full_window_tail_alignment_and_upstream_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path, summary, _ = replay_files(Path(tmp), 472)
            windows, metrics = prepare_sequence_windows(input_path, summary, mode="full")
            self.assertEqual([(int(w.frame_ids[0]), int(w.frame_ids[-1])) for w in windows], [(1,243),(82,324),(163,405),(230,472)])
            self.assertEqual(metrics.frames_received, 472)
            baseline, _ = prepare_sequence_windows(input_path, summary)
            self.assertEqual([w.window_id for w in baseline], [windows[-1].window_id])
            records = input_path.read_text().splitlines()
            input_path.write_text("\n".join(records[:-1]) + "\n")
            with self.assertRaises(PoseAdapterError):
                prepare_sequence_windows(input_path, summary, mode="full")

    def test_exact_window_stride_tail_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path, summary, _ = replay_files(Path(tmp), 405)
            windows, _ = prepare_sequence_windows(input_path, summary, mode="full")
            self.assertEqual(len(windows), 3)
            self.assertEqual(len({w.window_id for w in windows}), 3)

    def test_flip_is_involution_and_exchanges_the_verified_joint_indices(self):
        points = np.arange(2 * 17 * 3, dtype=np.float32).reshape(2, 17, 3)
        flipped = flip_h36m(points)
        np.testing.assert_array_equal(flip_h36m(flipped), points)
        np.testing.assert_array_equal(flipped[:, 4, 1:], points[:, 1, 1:])
        np.testing.assert_array_equal(flipped[:, 4, 0], -points[:, 1, 0])
        np.testing.assert_array_equal(flip_h36m(torch.from_numpy(points)).numpy(), flipped)

    def test_config_loads_nested_settings_and_resolves_paths(self):
        config = load_pose3d_config(ROOT / "configs/pose3d.yaml")
        self.assertIsInstance(config.output, OutputSettings)
        self.assertTrue(config.model.checkpoint.is_absolute())
        for changes in ({"clip_len": 81}, {"window_stride": 0}, {"pipeline_mode": "batch"}):
            with self.assertRaises(ValueError):
                Pose3DConfig(**changes)
        with self.assertRaises(ValueError):
            OutputSettings(tau_seconds=float("nan"))

    def test_missing_or_wrong_weights_are_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.bin"
            with self.assertRaises(Pose3DError) as missing:
                PoseMambaBackend(ModelSettings(checkpoint=path))
            self.assertEqual(missing.exception.code, "checkpoint_missing")
            path.write_bytes(b"wrong-checkpoint")
            with self.assertRaises(Pose3DError) as invalid:
                PoseMambaBackend(ModelSettings(checkpoint=path))
            self.assertEqual(invalid.exception.code, "checkpoint_mismatch")

    def test_model_output_root_is_zeroed_without_subtracting_noise_from_other_joints(self):
        backend = PoseMambaBackend.__new__(PoseMambaBackend)
        backend.settings = ModelSettings(device="cpu", flip_test=False)
        backend.torch = torch
        backend.model_info = {}
        backend.model = lambda inputs: torch.ones(1, 243, 17, 3)
        result = backend.predict(input_windows(243)[0])
        np.testing.assert_array_equal(result.pose_3d[:, 0], 0)
        np.testing.assert_array_equal(result.pose_3d[:, 1:], 1)
        with self.assertRaises(Pose3DError):
            backend.predict(replace(result.input_window, normalization_id="bbox"))
        with self.assertRaises(Pose3DError):
            backend.predict(replace(result.input_window, joint_order_id="unverified_order"))
        with self.assertRaises(Pose3DError):
            backend.predict(replace(result.input_window, source_frame_ids=result.input_window.source_frame_ids.astype(np.float64)))



if __name__ == "__main__":
    unittest.main()
