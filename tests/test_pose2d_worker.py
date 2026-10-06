from __future__ import annotations

import json
import queue
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path

import av
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from video_to_3d_motion.media.models import EndOfVideo, FrameTask
from video_to_3d_motion.pose2d.config import (
    BatchConfig,
    DetectionConfig,
    Pose2DConfig,
    VisualizationConfig,
)
from video_to_3d_motion.pose2d.models import (
    EndOfPose2D,
    PersonBBox,
    Pose2DFailure,
    Pose2DFrame,
    PosePrediction,
)
from video_to_3d_motion.pose2d.worker import Pose2DWorker


def make_task(frame_id: int, image: np.ndarray | None = None) -> FrameTask:
    frame = image if image is not None else np.zeros((64, 96, 3), dtype=np.uint8)
    return FrameTask(
        image=frame,
        video_id="test-video",
        frame_id=frame_id,
        source_frame_id=frame_id,
        timestamp=(frame_id - 1) / 50,
        source_timestamp=(frame_id - 1) / 50,
        timestamp_source="filename",
        image_size=(96, 64),
        fps=50,
    )


def read_image(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        frame = next(container.decode(video=0))
        return frame.to_ndarray(format="bgr24")


class StubDetector:
    """Test-only deterministic detector; it is not shipped as a runtime backend."""

    def detect_batch(self, images: list[np.ndarray]) -> list[list[PersonBBox]]:
        return [
            [
                PersonBBox(
                    bbox=np.asarray([8, 4, image.shape[1] - 8, image.shape[0] - 4], dtype=np.float32),
                    score=0.99,
                    source="rtmdet",
                )
            ]
            for image in images
        ]


class StubEstimator:
    """Test-only deterministic pose estimator."""

    joint_schema = "coco17"
    score_source = "test_stub"

    def infer_batch(
        self, images: list[np.ndarray], bboxes: list[np.ndarray]
    ) -> list[PosePrediction]:
        predictions = []
        for bbox in bboxes:
            xs = np.linspace(bbox[0], bbox[2], 17, dtype=np.float32)
            ys = np.linspace(bbox[1], bbox[3], 17, dtype=np.float32)
            predictions.append(
                PosePrediction(
                    keypoints=np.column_stack((xs, ys)).astype(np.float32),
                    keypoint_scores_raw=np.full(17, 0.9, dtype=np.float32),
                )
            )
        return predictions


def make_worker(config: Pose2DConfig) -> Pose2DWorker:
    return Pose2DWorker(
        config, detector=StubDetector(), estimator=StubEstimator()
    )


class Pose2DWorkerTests(unittest.TestCase):
    def test_batches_tail_and_writes_one_overlay_per_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "visualizations"
            config = Pose2DConfig(
                detection=DetectionConfig(min_bbox_area=10),
                batch=BatchConfig(max_size=2, input_queue_capacity=4),
                visualization=VisualizationConfig(output_root=output_root),
            )
            worker = make_worker(config)
            input_queue = queue.Queue()
            output_queue = queue.Queue()
            originals = []
            for frame_id in range(1, 4):
                image = np.zeros((64, 96, 3), dtype=np.uint8)
                originals.append(image.copy())
                input_queue.put(make_task(frame_id, image))
            input_queue.put(EndOfVideo("test-video", 3))

            result = worker.run(input_queue, output_queue, threading.Event())
            output = []
            while not output_queue.empty():
                output.append(output_queue.get_nowait())

            poses = [item for item in output if isinstance(item, Pose2DFrame)]
            terminal = [item for item in output if isinstance(item, EndOfPose2D)]
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.metrics.batches_processed, 2)
            self.assertEqual(result.metrics.visualization_frames_written, 3)
            self.assertEqual([pose.frame_id for pose in poses], [1, 2, 3])
            self.assertEqual(len(terminal), 1)
            self.assertEqual(input_queue.unfinished_tasks, 0)

            video_dir = output_root / "test-video"
            image_paths = sorted(video_dir.glob("frame_*.jpg"))
            self.assertEqual(len(image_paths), 3)
            rendered = read_image(image_paths[0])
            self.assertEqual(rendered.shape, (64, 96, 3))
            self.assertTrue(np.any(rendered != 0))
            for original in originals:
                self.assertFalse(np.any(original))

            records = [
                json.loads(line)
                for line in (video_dir / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([record["frame_id"] for record in records], [1, 2, 3])
            self.assertEqual(
                [path.name for path in image_paths],
                [f"frame_{index:06d}.jpg" for index in range(1, 4)],
            )

    def test_missing_required_external_bbox_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Pose2DConfig(
                detection=DetectionConfig(
                    bbox_source="external_only", min_bbox_area=10
                ),
                batch=BatchConfig(max_size=1),
                visualization=VisualizationConfig(
                    enabled=False, output_root=Path(temp_dir)
                ),
            )
            worker = make_worker(config)
            input_queue = queue.Queue()
            output_queue = queue.Queue()
            input_queue.put(make_task(1))
            input_queue.put(EndOfVideo("test-video", 1))

            result = worker.run(input_queue, output_queue, threading.Event())

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.error_code, "external_bbox_missing")
            self.assertIsInstance(output_queue.get_nowait(), Pose2DFailure)

    def test_every_n_frames_only_writes_selected_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Pose2DConfig(
                detection=DetectionConfig(min_bbox_area=10),
                batch=BatchConfig(max_size=4),
            )
            config = replace(
                base,
                visualization=VisualizationConfig(
                    output_root=Path(temp_dir), every_n_frames=2
                ),
            )
            input_queue = queue.Queue()
            output_queue = queue.Queue()
            for frame_id in range(1, 5):
                input_queue.put(make_task(frame_id))
            input_queue.put(EndOfVideo("test-video", 4))

            result = make_worker(config).run(
                input_queue, output_queue, threading.Event()
            )

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.metrics.visualization_frames_written, 2)
            names = sorted(path.name for path in Path(temp_dir).rglob("*.jpg"))
            self.assertEqual(names, ["frame_000001.jpg", "frame_000003.jpg"])


if __name__ == "__main__":
    unittest.main()
