"""Behavioral checks for S1-03 without loading GPU or pose models."""

from __future__ import annotations

import csv
import json
import queue
import sys
import tempfile
import threading
import unittest
import subprocess
from argparse import Namespace
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_to_3d_motion.pose2d.cli import _record
from video_to_3d_motion.pose2d.models import EndOfPose2D, PersonBBox, Pose2DFailure, Pose2DFrame, PosePrediction
from video_to_3d_motion.pose_adapter.archive import WindowArchive
from video_to_3d_motion.pose_adapter.cli import main
from video_to_3d_motion.pose_adapter.config import (
    AdapterQueueSettings, AdapterSettings, PoseAdapterConfig, load_pose_adapter_config,
)
from video_to_3d_motion.pose_adapter.exceptions import PoseAdapterError
from video_to_3d_motion.pose_adapter.jsonl_reader import iter_pose2d_jsonl
from video_to_3d_motion.pose_adapter.models import EndOfPoseAdapter, Pose2DWindow
from video_to_3d_motion.pose_adapter.normalization import normalize_h36m
from video_to_3d_motion.pose_adapter.skeleton import VERIFIED_CHECKPOINT_SHA256
from video_to_3d_motion.pose_adapter.window_buffer import WindowBuffer
from video_to_3d_motion.pose_adapter.worker import PoseAdapterWorker


def frame(number: int, **changes) -> Pose2DFrame:
    # Different values at each joint reveal accidental leg reordering.
    points = np.column_stack((np.arange(17) * 10 + number, np.arange(17) * 20 + number)).astype(np.float32)
    scores = np.full(17, 1.2, dtype=np.float32)
    scores[2] = 0.05
    return replace(Pose2DFrame(
        keypoints=points, keypoint_scores_raw=scores, valid_mask=scores >= 0.2,
        joint_schema="h36m17", score_source="rtmpose_simcc",
        bbox=np.array([0, 0, 1000, 1002], dtype=np.float32), bbox_score=0.9, bbox_source="external",
        video_id="one-sequence", frame_id=number, source_frame_id=number,
        timestamp=(number - 1) / 50, source_timestamp=(number - 1) / 50,
        timestamp_source="filename", image_size=(1000, 1002), fps=50,
        model="rtmpose_l_h36m", model_config="audited.py", checkpoint_sha256=VERIFIED_CHECKPOINT_SHA256,
    ), **changes)


def terminal(count: int) -> EndOfPose2D:
    return EndOfPose2D("one-sequence", count, count, 0, (count + 26) // 27)


def run_frames(items):
    output = []
    result = PoseAdapterWorker(PoseAdapterConfig()).run_stream(items, output.append)
    return result, output


def replay_files(directory: Path, count: int = 243):
    input_path = directory / "frames.jsonl"
    with input_path.open("w", encoding="utf-8") as stream:
        for number in range(1, count + 1):
            stream.write(json.dumps(_record(frame(number))) + "\n")
    summary = dict(status="completed", producer_status="completed", terminal_message="EndOfPose2D",
                   video_id="one-sequence", records_written=count, frames_output=str(input_path),
                   model="rtmpose_l_h36m", pose2d_metrics=dict(
                       frames_received=count, frames_succeeded=count, frames_failed=0,
                       batches_processed=(count + 26) // 27))
    path = directory / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return input_path, path, summary


class PoseAdapterCoordinatesTests(unittest.TestCase):
    def test_aspect_ratio_joint_identity_and_no_mutation(self):
        points = np.tile([500, 501], (17, 1)).astype(np.float32)
        points[1] = [0, 0]
        points[4] = [1000, 1002]
        original = points.copy()
        actual = normalize_h36m(points, (1000, 1002))
        np.testing.assert_allclose(actual[0], [0, 0], atol=1e-7)
        np.testing.assert_allclose(actual[1], [-1, -1.002], atol=1e-7)
        np.testing.assert_allclose(actual[4], [1, 1.002], atol=1e-7)
        np.testing.assert_array_equal(points, original)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertEqual(actual.dtype, np.float32)

    def test_terminal_window_metadata_and_low_confidence_survive(self):
        for count in (243, 472, 1384):
            with self.subTest(count=count):
                result, output = run_frames([*(frame(n) for n in range(1, count + 1)), terminal(count)])
                self.assertEqual(result.status, "completed")
                window, end = output
                self.assertIsInstance(window, Pose2DWindow)
                self.assertIsInstance(end, EndOfPoseAdapter)
                np.testing.assert_array_equal(window.frame_ids, np.arange(count - 242, count + 1))
                np.testing.assert_array_equal(window.source_frame_ids, window.frame_ids)
                np.testing.assert_allclose(window.timestamps, (window.frame_ids - 1) / 50)
                self.assertEqual(window.keypoints_norm.shape, (1, 243, 17, 2))
                # Compute expected values independently using image-center scaling.
                expected = (frame(count).keypoints.astype(np.float64) - [500, 501]) / 500
                np.testing.assert_allclose(window.keypoints_norm[0, -1], expected, atol=2e-7)
                self.assertFalse(window.valid_mask[-1, 2])
                self.assertEqual(window.keypoint_scores_raw[-1, 2], np.float32(0.05))
                self.assertEqual(result.metrics.low_confidence_points, count)
                self.assertEqual(result.metrics.max_buffer_size, 243)
                self.assertEqual(result.metrics.windows_emitted, 1)

    def test_input_mutation_cannot_change_cached_window(self):
        buffer = WindowBuffer(PoseAdapterConfig())
        first = frame(1)
        buffer.push(first)
        first.keypoints[:] = 999
        first.valid_mask[:] = False
        for n in range(2, 244):
            buffer.push(frame(n))
        window = buffer.finish(terminal(243))
        np.testing.assert_allclose(window.keypoints_norm[0, 0], normalize_h36m(frame(1).keypoints, (1000, 1002)))
        self.assertTrue(window.valid_mask[0, 0])


class PoseAdapterValidationTests(unittest.TestCase):
    def test_short_empty_and_bad_terminal_fail_without_window(self):
        cases = [([terminal(0)], "insufficient_frames"),
                 ([*(frame(n) for n in range(1, 243)), terminal(242)], "insufficient_frames"),
                 ([frame(1), terminal(2)], "metadata_mismatch"),
                 ([frame(1)], "metadata_mismatch")]
        for items, code in cases:
            result, output = run_frames(items)
            self.assertEqual(result.error_code, code)
            self.assertFalse(any(isinstance(x, Pose2DWindow) for x in output))

    def test_sparse_sampling_preserves_source_order_and_real_timestamps(self):
        for nominal_fps in (50, 10):
            with self.subTest(nominal_fps=nominal_fps):
                items = [frame(n, source_frame_id=5 * (n - 1) + 1, timestamp=(n - 1) / 10,
                               source_timestamp=(n - 1) / 10, fps=nominal_fps)
                         for n in range(1, 473)]
                result, output = run_frames([*items, terminal(472)])
                self.assertEqual(result.status, "completed")
                window = output[0]
                np.testing.assert_array_equal(window.frame_ids, np.arange(230, 473))
                np.testing.assert_array_equal(window.source_frame_ids, np.arange(1146, 2357, 5))
                np.testing.assert_array_equal(window.timestamps, np.arange(229, 472) / 10)
                np.testing.assert_array_equal(window.source_timestamps, window.timestamps)
                self.assertEqual(window.fps, nominal_fps)
                self.assertAlmostEqual(result.metrics.interval_min_seconds, 0.1)
                self.assertAlmostEqual(result.metrics.interval_max_seconds, 0.1)

    def test_irregular_sampling_preserves_gaps_without_resampling(self):
        source_ids = 1 + np.concatenate(([0], np.cumsum(np.tile([1, 5], 121))))
        timestamps = (source_ids - 1) / 50
        items = [frame(n, source_frame_id=int(source_id), timestamp=float(timestamp),
                       source_timestamp=float(timestamp))
                 for n, (source_id, timestamp) in enumerate(zip(source_ids, timestamps), 1)]
        result, output = run_frames([*items, terminal(243)])
        self.assertEqual(result.status, "completed")
        np.testing.assert_array_equal(output[0].source_frame_ids, source_ids)
        np.testing.assert_array_equal(output[0].timestamps, timestamps)
        self.assertAlmostEqual(result.metrics.interval_min_seconds, 0.02)
        self.assertAlmostEqual(result.metrics.interval_max_seconds, 0.1)

    def test_high_fps_source_indices_may_skip(self):
        items = [frame(n, source_frame_id=2 * n - 1) for n in range(1, 244)]
        result, output = run_frames([*items, terminal(243)])
        self.assertEqual(result.status, "completed")
        np.testing.assert_array_equal(output[0].source_frame_ids, np.arange(1, 486, 2))

    def test_sequence_and_model_changes_are_rejected(self):
        cases = [({"video_id": "other"}, "sequence_mismatch"),
                 ({"image_size": (1000, 1000)}, "image_size_changed"),
                 ({"checkpoint_sha256": "0" * 64}, "model_identity_changed"),
                 ({"frame_id": 3}, "frame_order_invalid"),
                 ({"source_frame_id": 1}, "timestamp_invalid"),
                 ({"source_frame_id": 3, "timestamp": 0.2, "source_timestamp": 0.2}, "timestamp_invalid"),
                 ({"timestamp": 0.0}, "timestamp_invalid"),
                 ({"timestamp": float("nan")}, "timestamp_invalid")]
        for changes, code in cases:
            with self.subTest(changes=changes):
                second = frame(2, **{"source_frame_id": 6, **changes})
                result, _ = run_frames([frame(1, source_frame_id=5), second])
                self.assertEqual(result.error_code, code)

    def test_wrong_profile_coco_nan_and_bad_mask_are_rejected(self):
        cases = [(frame(1, joint_schema="coco17"), "joint_schema_unsupported"),
                 (frame(1, checkpoint_sha256="0" * 64), "joint_profile_mismatch"),
                 (frame(1, keypoints=np.full((17, 2), np.nan, dtype=np.float32)), "invalid_keypoints"),
                 (frame(1, valid_mask=np.ones(17, dtype=np.int64)), "invalid_pose2d_frame")]
        for pose, code in cases:
            result, _ = run_frames([pose])
            self.assertEqual(result.error_code, code)

    def test_configuration_is_strict(self):
        config = load_pose_adapter_config(ROOT / "configs/pose_adapter.yaml")
        self.assertEqual(config.pose_adapter.clip_len, 243)
        with self.assertRaises(PoseAdapterError):
            PoseAdapterConfig(pipeline_mode="full")
        for changes in ({"clip_len": 81}, {"normalization": "bbox_center"}):
            with self.assertRaises(ValueError):
                AdapterSettings(**changes)

    def test_adapter_import_does_not_load_gpu_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c", "import sys; import video_to_3d_motion.pose_adapter; "
             "assert 'torch' not in sys.modules; assert 'mmpose' not in sys.modules"],
            cwd=ROOT, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class PoseAdapterQueueTests(unittest.TestCase):
    def test_small_queue_continuous_consumption_and_task_accounting(self):
        incoming = queue.Queue(maxsize=2)
        outgoing = queue.Queue(maxsize=1)
        stop = threading.Event()
        results = []
        observed = []
        worker = PoseAdapterWorker(PoseAdapterConfig(), record_frame=lambda pose: observed.append(pose.frame_id))
        thread = threading.Thread(target=lambda: results.append(worker.run(incoming, outgoing, stop)), daemon=True)
        thread.start()
        try:
            for n in range(1, 244):
                incoming.put(frame(n), timeout=2)
            incoming.put(terminal(243), timeout=2)
            self.assertIsInstance(outgoing.get(timeout=2), Pose2DWindow)
            outgoing.task_done()
            self.assertIsInstance(outgoing.get(timeout=2), EndOfPoseAdapter)
            outgoing.task_done()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(incoming.unfinished_tasks, 0)
            self.assertEqual(results[0].status, "completed")
            self.assertEqual(observed, list(range(1, 244)))
        finally:
            stop.set()
            thread.join(2)

    def test_cancel_releases_blocked_output_and_acknowledges_terminal(self):
        incoming = queue.Queue()
        outgoing = queue.Queue(maxsize=1)
        outgoing.put("occupied")
        for n in range(1, 244):
            incoming.put(frame(n))
        incoming.put(terminal(243))
        stop = threading.Event()
        results = []
        config = PoseAdapterConfig(queue=AdapterQueueSettings(put_timeout_seconds=0.01))
        reached_terminal = threading.Event()
        worker = PoseAdapterWorker(config, record_frame=lambda p: reached_terminal.set() if p.frame_id == 243 else None)
        thread = threading.Thread(target=lambda: results.append(worker.run(incoming, outgoing, stop)), daemon=True)
        thread.start()
        try:
            self.assertTrue(reached_terminal.wait(2))
            stop.set()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(results[0].status, "cancelled")
            self.assertEqual(results[0].metrics.windows_emitted, 0)
        finally:
            stop.set()
            thread.join(2)

    def test_cancel_releases_empty_input_and_upstream_failure_is_not_eof(self):
        stop = threading.Event()
        stop.set()
        result = PoseAdapterWorker(PoseAdapterConfig()).run(queue.Queue(), queue.Queue(), stop)
        self.assertEqual(result.status, "cancelled")
        result, output = run_frames([frame(1), Pose2DFailure("one-sequence", 2, "person_not_found", "missing", False)])
        self.assertEqual(result.error_code, "upstream_failed")
        self.assertFalse(any(isinstance(x, Pose2DWindow) for x in output))

    def test_recording_failure_is_observable(self):
        def fail_record(pose):
            raise OSError("disk full")
        result = PoseAdapterWorker(PoseAdapterConfig(), record_frame=fail_record).run_stream([frame(1)], lambda x: None)
        self.assertEqual(result.error_code, "archive_write_failed")


class PoseAdapterReplayTests(unittest.TestCase):
    def _online_cli(self, *, backend_failure=False):
        from video_to_3d_motion.media.models import DecodedFrame, VideoInfo
        from video_to_3d_motion.media.producer import FrameTaskProducer
        from video_to_3d_motion.media.config import load_config
        from video_to_3d_motion.pose2d import cli
        from video_to_3d_motion.pose2d.config import BatchConfig, ModelConfig, Pose2DConfig, VisualizationConfig
        from video_to_3d_motion.pose2d.worker import Pose2DWorker

        class Decoder:
            def probe(self, path):
                return VideoInfo(Path(path), 96, 64, 50, None)

            def iter_frames(self, info):
                for n in range(1, 261):
                    yield DecodedFrame(np.zeros((64, 96, 3), dtype=np.uint8), n, (n - 1) / 50, n, "pts")

        config = Pose2DConfig(model=ModelConfig(name="rtmpose_l_h36m", joint_schema="h36m17"),
                              batch=BatchConfig(max_size=4, output_queue_capacity=3),
                              visualization=VisualizationConfig(enabled=False))
        detector = SimpleNamespace(detect_batch=lambda images: [
            [PersonBBox(np.array([0, 0, 96, 64], dtype=np.float32), 0.9, "rtmdet")] for image in images])
        estimator = SimpleNamespace(joint_schema="h36m17", score_source="rtmpose_simcc", infer_batch=lambda images, boxes: [
            PosePrediction(frame(1).keypoints.copy(), frame(1).keypoint_scores_raw.copy()) for image in images])
        pose_worker = Pose2DWorker(config, detector=detector, estimator=estimator)
        pose_worker._checkpoint_sha256 = VERIFIED_CHECKPOINT_SHA256
        if backend_failure:
            def failing_inference(images, boxes):
                raise RuntimeError("backend crashed")
            estimator.infer_batch = failing_inference
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            args = Namespace(source=directory / "video.mp4", media_config=ROOT / "configs/media_decode.yaml",
                             pose2d_config=ROOT / "configs/pose2d.yaml", video_id="one-sequence",
                             person_selection=None, frames_output=directory / "frames.jsonl",
                             summary_output=directory / "summary.json",
                             pose_adapter_config=ROOT / "configs/pose_adapter.yaml", pose_adapter_output=directory / "adapter")
            producer = FrameTaskProducer(load_config(args.media_config), Decoder())
            with patch.object(cli, "FrameTaskProducer", return_value=producer), \
                    patch.object(cli, "load_pose2d_config", return_value=config), \
                    patch.object(cli, "create_gpu_worker", return_value=pose_worker):
                self.assertEqual(cli._run(args, image_sequence=False), 1 if backend_failure else 0)
            if backend_failure:
                summary = json.loads(args.summary_output.read_text())
                self.assertEqual(summary["pose_adapter"]["error_code"], "upstream_failed")
                self.assertEqual(summary["status"], "failed")
                self.assertFalse(list((directory / "adapter/windows").glob("*.npz")))
                return
            records = [json.loads(line) for line in args.frames_output.read_text().splitlines()]
            self.assertEqual([r["frame_id"] for r in records], list(range(1, 261)))
            summary = json.loads(args.summary_output.read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["pose_adapter"]["status"], "completed")
            with np.load(directory / "adapter/windows/window_000001.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["frame_ids"], np.arange(18, 261))

    def test_online_cli_records_and_adapts_the_same_frame_stream(self):
        self._online_cli()

    def test_online_backend_crash_stops_pipeline_without_waiting_for_terminal(self):
        self._online_cli(backend_failure=True)

    def test_cli_archive_and_csv_replay_use_existing_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            input_path, summary_path, summary = replay_files(directory, 472)
            csv_path = directory / "summary.csv"
            flat = {k: v for k, v in summary.items() if k != "pose2d_metrics"}
            flat.update({"pose2d_metrics." + k: v for k, v in summary["pose2d_metrics"].items()})
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=flat)
                writer.writeheader()
                writer.writerow(flat)
            for path in (summary_path, csv_path):
                out = directory / path.suffix[1:]
                self.assertEqual(main(["--input", str(input_path), "--input-summary", str(path),
                                       "--config", str(ROOT / "configs/pose_adapter.yaml"), "--output-dir", str(out)]), 0)
                with np.load(out / "windows/window_000001.npz", allow_pickle=False) as saved:
                    self.assertEqual(saved["keypoints_norm"].shape, (1, 243, 17, 2))
                    np.testing.assert_array_equal(saved["frame_ids"], np.arange(230, 473))
                    self.assertTrue(all(saved[key].dtype != object for key in saved.files))
                payload = json.loads((out / "adapter_summary.json").read_text())
                self.assertEqual(payload["metrics"]["frames_received"], 472)
                self.assertEqual(payload["status"], "completed")
                before = (out / "adapter_summary.json").read_bytes()
                with self.assertRaises(PoseAdapterError):
                    WindowArchive(out, PoseAdapterConfig())
                self.assertEqual((out / "adapter_summary.json").read_bytes(), before)

    def test_partial_failed_mismatched_and_normalized_inputs_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            input_path, summary_path, summary = replay_files(directory)
            for changes in ({"status": "failed"}, {"records_written": 244}, {"video_id": "other"}):
                summary_path.write_text(json.dumps(dict(summary, **changes)), encoding="utf-8")
                result, output = run_frames(iter_pose2d_jsonl(input_path, summary_path))
                self.assertEqual(result.error_code, "input_summary_invalid")
                self.assertFalse(any(isinstance(x, Pose2DWindow) for x in output))
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            records = input_path.read_text().splitlines()
            input_path.write_text("\n".join(records[:-1]) + "\n", encoding="utf-8")
            result, _ = run_frames(iter_pose2d_jsonl(input_path, summary_path))
            self.assertEqual(result.error_code, "input_summary_invalid")
            record = json.loads(records[0])
            record["coordinate_space"] = "h36m_screen_normalized"
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            result, _ = run_frames(iter_pose2d_jsonl(input_path, summary_path))
            self.assertEqual(result.error_code, "coordinate_space_mismatch")


if __name__ == "__main__":
    unittest.main()
