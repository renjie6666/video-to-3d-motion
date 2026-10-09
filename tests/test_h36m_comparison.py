from __future__ import annotations

import csv
import io
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_to_3d_motion.media.image_sequence_decoder import decode_image_bgr
from video_to_3d_motion.pose2d.comparison import COCO_TO_H36M, compare_record, run_comparison
from video_to_3d_motion.pose2d.config import VisualizationConfig
from video_to_3d_motion.pose2d.h36m_annotations import _SequenceUnpickler, load_sequence_annotations
from video_to_3d_motion.pose2d.visualization import _write_image

SEQUENCE = "s_01_act_02_subact_01_ca_01"


def gt():
    return {"keypoints": np.tile([12.0, 12.0], (17, 1)).tolist(), "visible": [True] * 17}


def prediction(number=1):
    return dict(frame_id=number, source_frame_id=number, joint_schema="coco17",
                keypoints=np.tile([15.0, 16.0], (17, 1)).tolist(),
                keypoint_scores_raw=[0.9] * 17, valid_mask=[True] * 17,
                image_size=[32, 24], bbox_source="rtmdet")


def annotation(number=1, *, camera=None):
    return dict(image=f"{SEQUENCE}/{SEQUENCE}_{number:06d}.jpg",
                joints_2d=np.full((17, 2), 12.0, dtype=np.float32),
                joints_vis=np.ones((17, 3)), camera=camera or {"R": np.eye(3)})


class H36MComparisonTests(unittest.TestCase):
    def test_correct_joint_mapping_and_no_fabricated_head(self):
        record = prediction()
        truth = gt()
        truth["keypoints"] = np.column_stack((np.arange(17), np.arange(17))).tolist()
        for pred_index, gt_index in COCO_TO_H36M.items():
            record["keypoints"][pred_index] = [gt_index + 3, gt_index + 4]
        rows = compare_record(record, truth, "image.jpg")
        self.assertEqual(sum(row["status"] == "ok" for row in rows), 12)
        self.assertEqual(rows[5]["gt_joint_index"], 11)
        self.assertEqual(rows[12]["gt_joint_index"], 1)
        self.assertEqual(rows[0]["status"], "unmapped_joint")
        self.assertIsNone(rows[0]["error_px"])
        self.assertEqual(rows[15]["mapping"], "ankle_foot_proxy")
        self.assertTrue(all(row["error_px"] == 5.0 for row in rows[5:]))

    def test_confidence_visibility_and_missing_gt_are_not_zero_errors(self):
        record = prediction()
        record["valid_mask"][5] = False
        truth = gt()
        truth["visible"][14] = False
        rows = compare_record(record, truth, "image.jpg")
        self.assertEqual(rows[5]["status"], "low_confidence")
        self.assertEqual(rows[5]["error_px"], 5.0)
        self.assertEqual(rows[6]["status"], "gt_invisible")
        self.assertIsNone(rows[6]["error_px"])
        missing = compare_record(record, None, "missing.jpg")
        self.assertTrue(all(row["status"] == "missing_gt" and row["error_px"] is None for row in missing))

    def test_h36m_schema_and_nonfinite_predictions(self):
        record = prediction()
        record["joint_schema"] = "h36m17"
        record["keypoints"][0][0] = float("nan")
        rows = compare_record(record, gt(), "image.jpg")
        self.assertEqual(rows[0]["status"], "invalid_prediction")
        self.assertIsNone(rows[0]["pred_x"])
        self.assertEqual(rows[16]["gt_joint_index"], 16)

    def test_streaming_pickle_crosses_batches_and_preserves_shared_camera(self):
        camera = {"R": np.eye(3), "T": np.ones((3, 1))}
        records = [annotation(index, camera=camera) for index in range(1, 2002)]
        payload = pickle.dumps(records, protocol=4)
        found = []
        reader = _SequenceUnpickler(io.BytesIO(payload), lambda row: found.append(row["image"]))
        root = reader.load()
        self.assertIs(root, reader.root)
        self.assertEqual(root, [])
        self.assertEqual(len(found), 2001)
        # The memo keeps calibration and constants, rather than every ndarray.
        self.assertLess(sum(isinstance(v, np.ndarray) for v in reader.memo.values()), 10)

    def test_arbitrary_pickle_globals_are_rejected(self):
        import datetime
        reader = _SequenceUnpickler(io.BytesIO(pickle.dumps([datetime.date(2020, 1, 1)], protocol=4)), lambda _: None)
        with self.assertRaisesRegex(pickle.UnpicklingError, "unsupported annotation global"):
            reader.load()

    def test_reused_frame_arrays_fail_explicitly(self):
        records = [annotation(index) for index in range(1, 1002)]
        records[-1]["joints_2d"] = records[0]["joints_2d"]
        reader = _SequenceUnpickler(io.BytesIO(pickle.dumps(records, protocol=4)), lambda _: None)
        with self.assertRaises(pickle.UnpicklingError):
            reader.load()

    def test_csv_summary_missing_gt_and_original_size_overlays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / SEQUENCE
            source.mkdir()
            for index in (1, 2):
                _write_image(np.zeros((24, 32, 3), dtype=np.uint8),
                             source / f"{SEQUENCE}_{index:06d}.jpg", VisualizationConfig())
            annotation_path = root / "train.pkl"
            annotation_path.write_bytes(pickle.dumps([annotation()], protocol=4))
            frames = root / "frames.jsonl"
            first = prediction()
            first["valid_mask"][5] = False
            frames.write_text(json.dumps(first) + "\n" + json.dumps(prediction(2)) + "\n", encoding="utf-8")
            summary = run_comparison(source, frames, annotation_path, root / "comparison")
            self.assertEqual(summary["counts"]["missing_gt_frames"], 1)
            self.assertEqual(summary["all_visible_pairs"]["count"], 12)
            self.assertEqual(summary["confident_pairs"]["count"], 11)
            self.assertEqual(summary["all_visible_pairs"]["mean_px"], 5.0)
            self.assertEqual(summary["direct_pairs"]["count"], 10)
            self.assertEqual(summary["common10_pairs"]["count"], 10)
            self.assertEqual(summary["common10_pairs"]["mean_px"], 5.0)
            self.assertEqual(summary["ankle_foot_proxy_pairs"]["count"], 2)
            with (root / "comparison/joint_comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 34)
            self.assertEqual(rows[17]["status"], "missing_gt")
            self.assertEqual(rows[17]["error_px"], "")
            self.assertEqual(decode_image_bgr(root / "comparison/overlays/frame_000001.jpg").shape, (24, 32, 3))
            cached = load_sequence_annotations(annotation_path, source)
            self.assertEqual(len(cached), 1)

    def test_wrong_annotation_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / SEQUENCE
            source.mkdir()
            (source / f"{SEQUENCE}_000001.jpg").touch()
            annotations = root / "validation.pkl"
            annotations.write_bytes(pickle.dumps([], protocol=4))
            with self.assertRaisesRegex(ValueError, "no matching annotations"):
                run_comparison(source, root / "frames.jsonl", annotations, root / "comparison")

    def test_wrong_coordinate_space_or_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / SEQUENCE
            source.mkdir()
            (source / f"{SEQUENCE}_000001.jpg").touch()
            annotations = root / "train.pkl"
            annotations.write_bytes(pickle.dumps([annotation()], protocol=4))
            frames = root / "frames.jsonl"
            for changes, message in (
                ({"coordinate_space": "normalized"}, "pixel coordinates"),
                ({"source_path": str(root / "different_sequence")}, "source_path"),
            ):
                record = prediction()
                record.update(changes)
                frames.write_text(json.dumps(record) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    run_comparison(source, frames, annotations, root / "comparison")

    def test_image_cli_connects_queues_selection_and_comparison(self):
        from video_to_3d_motion.pose2d.cli import _parser, _run
        from video_to_3d_motion.pose2d.models import PersonBBox, PosePrediction
        from video_to_3d_motion.pose2d.worker import Pose2DWorker

        class Detector:
            def detect_batch(self, images):
                return [[PersonBBox(np.array([0, 0, 32, 24], dtype=np.float32), 0.9, "rtmdet"),
                         PersonBBox(np.array([10, 10, 30, 24], dtype=np.float32), 0.4, "rtmdet")]
                        for _ in images]

        class Estimator:
            joint_schema = "coco17"
            score_source = "test"

            def infer_batch(self, images, boxes):
                return [PosePrediction(np.tile([15, 16], (17, 1)).astype(np.float32),
                                       np.full(17, 0.9, dtype=np.float32)) for _ in images]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / SEQUENCE
            source.mkdir()
            _write_image(np.zeros((24, 32, 3), dtype=np.uint8), source / f"{SEQUENCE}_000001.jpg", VisualizationConfig())
            annotations = root / "train.pkl"
            annotations.write_bytes(pickle.dumps([annotation()], protocol=4))
            pose_config = root / "pose.yaml"
            pose_config.write_text("visualization:\n  output_root: overlays\n", encoding="utf-8")
            argv = [str(source), "--media-config", str(Path(__file__).resolve().parents[1] / "configs/media_decode.yaml"),
                    "--pose2d-config", str(pose_config), "--frames-output", str(root / "frames.jsonl"),
                    "--summary-output", str(root / "summary.json"), "--annotations", str(annotations),
                    "--comparison-output", str(root / "comparison")]
            factory = lambda config: Pose2DWorker(config, detector=Detector(), estimator=Estimator())
            with patch("video_to_3d_motion.pose2d.cli.create_gpu_worker", side_effect=factory), patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
                strict = _parser("test", "source", image_sequence=True).parse_args(argv)
                self.assertEqual(_run(strict, image_sequence=True), 1)
                summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["error_code"], "multiple_persons")
                selected = _parser("test", "source", image_sequence=True).parse_args(argv + ["--person-selection", "highest_score"])
                self.assertEqual(_run(selected, image_sequence=True), 0)
            record = json.loads((root / "frames.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["bbox"], [0, 0, 32, 24])
            self.assertEqual(record["bbox_selection"], "highest_score")
            self.assertEqual(record["coordinate_space"], "original_image_pixels")
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["producer_status"], "completed")
            self.assertEqual(summary["comparison"]["counts"]["matched_frames"], 1)


if __name__ == "__main__":
    unittest.main()
