"""S1-05 fusion, metrics, export and viewer behavior checks."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from tests.test_pose_adapter import frame, replay_files, terminal
from tests.pose3d_fixtures import input_windows
from video_to_3d_motion.pose_adapter.config import PoseAdapterConfig
from video_to_3d_motion.pose_adapter.exceptions import PoseAdapterError
from video_to_3d_motion.pose_adapter.window_buffer import WindowBuffer
from video_to_3d_motion.pose_adapter.windows import prepare_sequence_windows
from video_to_3d_motion.pose3d.backend import PoseMambaBackend, flip_h36m
from video_to_3d_motion.pose3d.cli import main as lift_main
from video_to_3d_motion.pose3d.config import ModelSettings, OutputSettings, Pose3DConfig, load_pose3d_config
from video_to_3d_motion.pose3d.exceptions import Pose3DError
from video_to_3d_motion.pose3d.models import Pose3DSequence, Pose3DWindow
from video_to_3d_motion.pose3d.vendor.scan import selective_scan_torch, CrossScan_plus_poselimbs, CrossMerge_plus_poselimbs
from video_to_3d_motion.stage_output.annotations import H5_TO_H36M17, SequenceAnnotations, load_annotations
from video_to_3d_motion.stage_output.archive import load_sequence, save_sequence
from video_to_3d_motion.stage_output.compare import compare_runs
from video_to_3d_motion.stage_output.fusion import fuse_windows, smooth_sequence
from video_to_3d_motion.stage_output.metrics import acceleration, align_procrustes, align_scale, evaluate_sequence, apply_official_25d
from video_to_3d_motion.stage_output.calibration import CAMERA_54138969, project_camera, reconstruct_camera, undistort_pixels, verify_camera_labels

ROOT = Path(__file__).resolve().parents[1]


class CalibrationTests(unittest.TestCase):
    def test_official_factor_preserves_metric_error_without_fitting_scale(self):
        poses = np.arange(2 * 17 * 3).reshape(2, 17, 3).astype(np.float32) / 100
        poses -= poses[:, :1]
        sequence = SimpleNamespace(P_3d=poses, P_3d_raw=poses.copy(), frame_ids=np.array([1,2]),
                                   source_frame_ids=np.array([1,6]), metadata=dict(image_size=[1000,1002]))
        factor = np.array([4.,5.])
        target = poses.astype(np.float64) * 500 * factor[:,None,None]
        target[:,1,0] += 17
        def evaluation():
            return (dict(sequence='S9_Test', variants=dict(raw={}, output={})),
                    [dict(variant='output',frame_id=1)], [dict(variant='output',joint_index=1)], {})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'labels.npz'
            np.savez(path,source_frame_ids=sequence.source_frame_ids,sequence='S9_Test',factors=factor,
                     target_25d_mm=target,original_pickle_sha256='a'*64)
            result=apply_official_25d(evaluation(),sequence,path)
            self.assertAlmostEqual(result[0]['mpjpe_mm'],1.)
            self.assertAlmostEqual(result[2][0]['mpjpe_official_25d_mm'],17.)
            np.testing.assert_allclose(result[3]['prediction_output_official_25d_mm'],poses*500*factor[:,None,None],atol=.001)
            sequence.source_frame_ids=np.array([1,7])
            with self.assertRaises(Pose3DError):
                apply_official_25d(evaluation(),sequence,path)

    def test_lcn_image_depth_reconstructs_known_camera_pose_without_alignment(self):
        rng = np.random.default_rng(19)
        camera_pose = rng.normal(size=(3, 17, 3)) * [200, 300, 180] + [300, 150, 5100]
        pixels = project_camera(camera_pose, distorted=False)
        ratio = CAMERA_54138969["focal"][0] / camera_pose[:, :1, 2:] + 1 / 2000
        normalized = np.concatenate((pixels - pixels[:, :1],
                                     (camera_pose[..., 2:] - camera_pose[:, :1, 2:]) * ratio), axis=-1) / 500
        actual = reconstruct_camera(normalized, camera_pose, 1000)
        np.testing.assert_allclose(actual, camera_pose - camera_pose[:, :1], atol=1e-9)
        shifted = normalized.copy()
        shifted[:, 1, 0] += .1
        self.assertGreater(np.linalg.norm(reconstruct_camera(shifted, camera_pose, 1000) - actual), 100)

    def test_distortion_inverse_and_camera_validation_reject_wrong_labels(self):
        camera_pose = np.arange(51).reshape(1, 17, 3) * [10, 13, 3] + [-200, -300, 5000]
        pixels = project_camera(camera_pose)
        np.testing.assert_allclose(undistort_pixels(pixels), project_camera(camera_pose, distorted=False), atol=1e-9)
        self.assertEqual(verify_camera_labels(camera_pose, pixels)["max_projection_error_px"], 0)
        with self.assertRaisesRegex(ValueError, "intrinsics"):
            verify_camera_labels(camera_pose, pixels + 2)




def prediction(window, bias=0):
    xy = window.keypoints_norm[0] - window.keypoints_norm[0, :, :1]
    z = np.broadcast_to(np.sin(np.arange(17)).astype(np.float32)[None, :, None] * .04, (243, 17, 1))
    poses = np.concatenate((xy, z), axis=-1).astype(np.float32)
    poses[:, 1:] += bias
    poses[:, 0] = 0
    return Pose3DWindow(poses, window, dict(model="test-inference", checkpoint_sha256="fixed-test-weights",
                                          backend="test-only", flip_test=False, units="h36m_image_normalized"), .001)


def annotation_files(directory, ids):
    names_path = directory / "valid_images.txt"
    names_path.write_text("\n".join(f"S9_Test_1.54138969_{source_id:06d}.jpg" for source_id in ids), encoding="utf-8")
    h5_path = directory / "valid.h5"
    original = []
    points = []
    for source_id in ids:
        offsets = np.column_stack((np.arange(17) * 40, np.arange(17) * 80, np.sin(np.arange(17)) * 80))
        original.append(offsets + [source_id, 200, 5000])
        points.append(np.column_stack((np.arange(17) * 10 + source_id, np.arange(17) * 20 + source_id)))
    with h5py.File(h5_path, "w") as dataset:
        dataset["S"] = np.array(original)[:, H5_TO_H36M17]
        dataset["part"] = np.array(points)[:, H5_TO_H36M17]
        dataset["normalize"] = np.zeros(len(ids), dtype=np.uint8)
    return h5_path, names_path, np.array(original), np.array(points)






class FusionAndOutputTests(unittest.TestCase):
    def test_overlap_mean_counts_and_last_ten_export(self):
        inputs = input_windows()
        predictions = [prediction(window, bias=index) for index, window in enumerate(inputs)]
        sequence = fuse_windows(predictions, np.arange(1, 473))
        self.assertEqual(sequence.P_3d.shape, (472, 17, 3))
        # Frame 240 occurs in all four windows; its four biases average to 1.5.
        base = prediction(inputs[0]).pose_3d[239, 1]
        np.testing.assert_allclose(sequence.P_3d[239, 1], base + 1.5, atol=1e-7)
        self.assertEqual(sequence.contribution_count[239], 4)
        self.assertEqual(sequence.contribution_count[0], 1)
        with tempfile.TemporaryDirectory() as tmp:
            save_sequence(tmp, sequence)
            with np.load(Path(tmp) / "sequence_3d.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["P_obs"], sequence.P_3d[-10:])
                np.testing.assert_array_equal(saved["obs_frame_ids"], np.arange(463, 473))
                self.assertTrue(all(saved[key].dtype.kind != 'O' for key in saved.files))
            loaded = load_sequence(tmp)
            np.testing.assert_array_equal(loaded.P_3d, sequence.P_3d)

    def test_conflicting_overlap_duplicate_window_and_coverage_gap_rejected(self):
        windows = [prediction(window) for window in input_windows(600)]
        with self.assertRaises(Pose3DError):
            fuse_windows([windows[0], windows[-1]], np.arange(1, 601))
        with self.assertRaises(Pose3DError):
            fuse_windows([windows[0], windows[0]], np.arange(1, 244))
        bad_times = windows[1].input_window.timestamps.copy()
        bad_times[0] += .001
        changed = replace(windows[1], input_window=replace(windows[1].input_window, timestamps=bad_times))
        with self.assertRaises(Pose3DError) as mismatch:
            fuse_windows([windows[0], changed, *windows[2:]], np.arange(1, 601))
        self.assertEqual(mismatch.exception.code, "metadata_mismatch")

    def test_time_aware_ema_is_causal_and_preserves_first_sample(self):
        poses = np.zeros((4, 17, 3), dtype=np.float32)
        poses[:, 1:, 0] = np.array([1, 2, 4, 8])[:, None]
        timestamps = np.array([0, .02, .1, .2])
        settings = OutputSettings(smoothing="ema", tau_seconds=.04)
        actual = smooth_sequence(poses, timestamps, settings)
        expected = [1]
        for previous, value, dt in zip(range(3), [2, 4, 8], [.02,.08,.1]):
            a = 1 - np.exp(-dt / .04)
            expected.append(a * value + (1-a) * expected[-1])
        np.testing.assert_allclose(actual[:, 1, 0], expected, rtol=1e-6)
        changed = poses.copy()
        changed[-1] = 1000
        np.testing.assert_array_equal(smooth_sequence(changed, timestamps, settings)[:-1], actual[:-1])
        np.testing.assert_array_equal(actual[:, 0], 0)


class MetricTests(unittest.TestCase):
    def test_procrustes_removes_similarity_but_not_reflection(self):
        rng = np.random.default_rng(10)
        target = rng.normal(size=(5, 17, 3)) * 100
        angle = .7
        rotation = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        transformed = (target @ rotation) / 3 + [20,50,80]
        np.testing.assert_allclose(align_procrustes(transformed, target), target, atol=1e-10)
        reflected = target.copy()
        reflected[..., 0] *= -1
        self.assertGreater(np.linalg.norm(align_procrustes(reflected, target)-target, axis=-1).mean(), 50)
        with self.assertRaises(Pose3DError):
            align_procrustes(np.zeros_like(target), target)

    def test_scale_only_and_real_time_acceleration(self):
        rng = np.random.default_rng(11)
        target = rng.normal(size=(4,17,3)) * 100
        actual, scale = align_scale(target / 7, target)
        np.testing.assert_allclose(actual, target, atol=1e-10)
        np.testing.assert_allclose(scale, 7)
        reflected, scale = align_scale(-target, target)
        np.testing.assert_array_equal(scale, 0)
        times = np.array([0,.02,.1,.3,.4])
        poses = np.broadcast_to((times ** 2)[:,None,None], (5,17,3)).copy()
        acc, centers = acceleration(poses, times)
        np.testing.assert_allclose(acc, 2, atol=1e-12)
        np.testing.assert_array_equal(centers, [1,2,3])

    def test_h5_row_join_source_ids_and_label_only_permutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path, names, originals, points = annotation_files(Path(tmp), [11,1,16,6])
            annotations = load_annotations(h5_path, names, "S9_Test_1.54138969", [1,6,11,21])
            np.testing.assert_array_equal(annotations.row_indices, [1,3,0,-1])
            np.testing.assert_array_equal(annotations.joints_camera_mm[:3], originals[[1,3,0]])
            np.testing.assert_array_equal(annotations.joints_pixel[:3], points[[1,3,0]])
            self.assertEqual(annotations.provenance["missing_source_frame_ids"], [21])

    def test_perfect_normalized_prediction_has_zero_labeled_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path, names, originals, points = annotation_files(Path(tmp), [1,6,11,16])
            annotations = load_annotations(h5_path, names, "S9_Test_1.54138969", [1,6,11,16])
            gt = originals - originals[:,:1]
            normalized = (gt / 2000).astype(np.float32)
            sequence = Pose3DSequence(normalized, normalized.copy(), np.arange(1,5), np.array([1,6,11,16]),
                                     np.array([0,.1,.2,.3]), np.array([0,.1,.2,.3]), np.ones((4,17),np.float32),
                                     np.ones((4,17),bool), np.ones(4,np.int64), dict(image_size=[1000,1002],fps=50))
            metrics, rows, joints, arrays = evaluate_sequence(sequence, annotations)
            self.assertIsNone(metrics["mpjpe_mm"])
            for name in ["n_mpjpe_mm","pa_mpjpe_mm","mpjpe_gt_scale_mm"]:
                self.assertLess(metrics["variants"]["output"][name]["mean"], 1e-4)
            self.assertAlmostEqual(metrics["sampling"]["median_effective_fps"], 10)


class ReplayIntegrationTests(unittest.TestCase):
    def test_partial_replay_never_initializes_inference_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            input_path, summary, _ = replay_files(directory, 244)
            input_path.write_text("\n".join(input_path.read_text().splitlines()[:-1]) + "\n")
            with patch("video_to_3d_motion.pose3d.cli.PoseMambaBackend") as backend:
                code = lift_main(["--input",str(input_path),"--input-summary",str(summary),
                                  "--config",str(ROOT / "configs/pose3d.yaml"),"--mode","full",
                                  "--output-dir",str(directory / "failed")])
                self.assertEqual(code, 1)
                backend.assert_not_called()
            result = json.loads((directory / "failed/run_summary.json").read_text())
            self.assertEqual(result["error_code"], "input_summary_invalid")
            self.assertFalse((directory / "failed/sequence_3d.npz").exists())

    def test_no_gt_run_exports_a_viewer_without_evaluation(self):
        class TestBackend:
            def __init__(self, settings):
                pass
            def predict(self, window):
                return prediction(window)
            def runtime_info(self):
                return dict(backend="test-only")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            input_path, summary, _ = replay_files(directory, 243)
            with patch("video_to_3d_motion.pose3d.cli.PoseMambaBackend", TestBackend):
                code = lift_main(["--input",str(input_path),"--input-summary",str(summary),
                                  "--config",str(ROOT / "configs/pose3d.yaml"),"--output-dir",str(directory / "no_gt"),
                                  "--image-dir",str(directory),"--image-prefix","custom"])
            self.assertEqual(code, 0)
            result = json.loads((directory / "no_gt/run_summary.json").read_text())
            self.assertEqual(result["evaluation_status"], "not_requested")
            self.assertTrue((directory / "no_gt/viewer.html").is_file())
            self.assertFalse((directory / "no_gt/metrics.json").exists())

    def test_cli_export_viewer_and_comparison_share_exact_source_frames(self):
        class TestBackend:
            def __init__(self, settings):
                self.calls = 0
            def predict(self, window):
                self.calls += 1
                return prediction(window)
            def runtime_info(self):
                return dict(backend="test-only", calls=self.calls)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            input_path, summary, _ = replay_files(directory, 472)
            h5_path, names, _, _ = annotation_files(directory, list(range(1,473)))
            runs = []
            for mode in ("baseline", "full"):
                out = directory / mode
                args = ["--input",str(input_path),"--input-summary",str(summary),"--config",str(ROOT / "configs/pose3d.yaml"),
                        "--output-dir",str(out),"--mode",mode,"--annotations",str(h5_path),"--annotation-names",str(names),"--sequence","S9_Test_1.54138969"]
                with patch("video_to_3d_motion.pose3d.cli.PoseMambaBackend", TestBackend):
                    self.assertEqual(lift_main(args), 0)
                    # No overwrite, even though the second command is otherwise valid.
                    self.assertEqual(lift_main(args), 1)
                result = json.loads((out / "run_summary.json").read_text())
                self.assertEqual(result["status"], "completed")
                viewer = (out / "viewer.html").read_text(encoding="utf-8")
                self.assertNotIn("/*__POSE_DATA__*/", viewer)
                self.assertNotIn('<script src=', viewer)
                runs.append(out)
            report = compare_runs(runs, directory / "comparison")
            self.assertEqual(report["common_source_frame_ids"], list(range(230,473)))
            self.assertTrue(all(row["common_frames"]==243 for row in report["rows"]))
            self.assertEqual(report["rows"][0]["n_mpjpe_mm"], report["rows"][2]["n_mpjpe_mm"])

    def test_relative_annotation_cli_paths_survive_snapshot_reload(self):
        class TestBackend:
            def __init__(self, settings):
                pass
            def predict(self, window):
                return prediction(window)
            def runtime_info(self):
                return dict(backend="test-only")
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            directory = Path(tmp)
            input_path, summary, _ = replay_files(directory, 243)
            h5_path, names, _, _ = annotation_files(directory, list(range(1, 244)))
            out = directory / "output"
            args = ["--input", str(input_path), "--input-summary", str(summary),
                    "--config", str(ROOT / "configs/pose3d.yaml"), "--output-dir", str(out),
                    "--annotations", str(h5_path.relative_to(Path.cwd())),
                    "--annotation-names", str(names.relative_to(Path.cwd())),
                    "--sequence", "S9_Test_1.54138969"]
            with patch("video_to_3d_motion.pose3d.cli.PoseMambaBackend", TestBackend):
                self.assertEqual(lift_main(args), 0)
            loaded = load_pose3d_config(out / "config_snapshot.yaml")
            self.assertEqual(loaded.evaluation.annotation_h5, h5_path.resolve())
            self.assertEqual(loaded.evaluation.annotation_names, names.resolve())


if __name__ == "__main__":
    unittest.main()
