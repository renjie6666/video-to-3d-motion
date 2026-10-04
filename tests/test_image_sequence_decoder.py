from __future__ import annotations

import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from video_to_3d_motion.media.config import ImageSequenceConfig
from video_to_3d_motion.media.config import AppConfig, MediaConfig, QueueConfig
from video_to_3d_motion.media.image_sequence_decoder import ImageSequenceDecoder
from video_to_3d_motion.media.exceptions import MediaPipelineError
from video_to_3d_motion.media.models import EndOfVideo, FrameTask
from video_to_3d_motion.media.producer import FrameTaskProducer


class FakeImageLoader:
    def __init__(self, sizes: dict[str, tuple[int, int]] | None = None) -> None:
        self.sizes = sizes or {}

    def __call__(self, path: Path) -> np.ndarray:
        width, height = self.sizes.get(path.name, (6, 4))
        return np.zeros((height, width, 3), dtype=np.uint8)


def touch_sequence(directory: Path, names: list[str]) -> None:
    for name in names:
        (directory / name).touch()


class ImageSequenceDecoderTests(unittest.TestCase):
    def test_numeric_sort_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            touch_sequence(
                root,
                ["seq_000003.jpg", "seq_000001.jpg", "seq_000002.jpg"],
            )
            decoder = ImageSequenceDecoder(image_loader=FakeImageLoader())
            info = decoder.probe(root)
            frames = list(decoder.iter_frames(info))

            self.assertEqual([frame.source_frame_id for frame in frames], [1, 2, 3])
            self.assertEqual(
                [frame.source_timestamp for frame in frames], [0.0, 0.02, 0.04]
            )
            self.assertEqual((info.width, info.height), (6, 4))

    def test_sequence_must_start_at_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            touch_sequence(root, ["seq_000000.jpg", "seq_000001.jpg"])
            decoder = ImageSequenceDecoder(image_loader=FakeImageLoader())

            with self.assertRaises(MediaPipelineError) as context:
                decoder.probe(root)
            self.assertEqual(context.exception.code, "frame_start_invalid")
    def test_missing_frame_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            touch_sequence(root, ["seq_000001.jpg", "seq_000003.jpg"])
            decoder = ImageSequenceDecoder(image_loader=FakeImageLoader())

            with self.assertRaises(MediaPipelineError) as context:
                decoder.probe(root)
            self.assertEqual(context.exception.code, "frame_missing")

    def test_duplicate_frame_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            touch_sequence(root, ["seq_000001.jpg", "seq_000001.png"])
            decoder = ImageSequenceDecoder(image_loader=FakeImageLoader())

            with self.assertRaises(MediaPipelineError) as context:
                decoder.probe(root)
            self.assertEqual(context.exception.code, "duplicate_frame")

    def test_image_size_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            touch_sequence(root, ["seq_000001.jpg", "seq_000002.jpg"])
            loader = FakeImageLoader({"seq_000002.jpg": (8, 4)})
            decoder = ImageSequenceDecoder(image_loader=loader)
            info = decoder.probe(root)

            with self.assertRaises(MediaPipelineError) as context:
                list(decoder.iter_frames(info))
            self.assertEqual(context.exception.code, "image_size_changed")

    def test_existing_producer_emits_frame_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            touch_sequence(
                root,
                ["seq_000001.jpg", "seq_000002.jpg", "seq_000003.jpg"],
            )
            decoder = ImageSequenceDecoder(image_loader=FakeImageLoader())
            config = AppConfig(
                media=MediaConfig(target_fps=50),
                queue=QueueConfig(capacity=8),
            )
            producer = FrameTaskProducer(config, decoder=decoder)
            output_queue = queue.Queue(maxsize=8)
            result = producer.run(root, output_queue, threading.Event(), "seq")

            output = []
            while not output_queue.empty():
                output.append(output_queue.get_nowait())
            tasks = [item for item in output if isinstance(item, FrameTask)]

            self.assertEqual(result.status, "completed")
            self.assertEqual([task.frame_id for task in tasks], [1, 2, 3])
            self.assertEqual([task.source_frame_id for task in tasks], [1, 2, 3])
            self.assertEqual(
                [task.timestamp_source for task in tasks],
                ["filename", "filename", "filename"],
            )
            self.assertIsInstance(output[-1], EndOfVideo)


if __name__ == "__main__":
    unittest.main()
