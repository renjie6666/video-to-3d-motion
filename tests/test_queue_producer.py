from __future__ import annotations

import queue
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_to_3d_motion.media.config import AppConfig, MediaConfig, QueueConfig
from video_to_3d_motion.media.models import DecodedFrame, EndOfVideo, FrameTask, VideoInfo
from video_to_3d_motion.media.producer import FrameTaskProducer


class FakeDecoder:
    def __init__(self, frame_count: int, source_fps: float = 50.0) -> None:
        self.frame_count = frame_count
        self.source_fps = source_fps

    def probe(self, video_path: str | Path) -> VideoInfo:
        return VideoInfo(Path(video_path), 6, 4, self.source_fps, None)

    def iter_frames(self, video_info: VideoInfo):
        for index in range(self.frame_count):
            yield DecodedFrame(
                image=np.zeros((4, 6, 3), dtype=np.uint8),
                source_frame_id=index + 1,
                source_timestamp=index / self.source_fps,
                pts=index,
                timestamp_source="pts",
            )


class FrameTaskProducerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            media=MediaConfig(target_fps=50),
            queue=QueueConfig(
                batch_size=2,
                prefetch_batches=1,
                capacity=2,
                put_timeout_seconds=0.01,
            ),
        )

    def test_backpressure_blocks_without_dropping_frames(self) -> None:
        output_queue = queue.Queue(maxsize=2)
        stop_event = threading.Event()
        producer = FrameTaskProducer(self.config, decoder=FakeDecoder(6))
        result_holder = []

        worker = threading.Thread(
            target=lambda: result_holder.append(
                producer.run("fake.mp4", output_queue, stop_event, "test-video")
            )
        )
        worker.start()
        time.sleep(0.05)

        received = []
        while True:
            item = output_queue.get(timeout=1.0)
            if isinstance(item, FrameTask):
                received.append(item)
            elif isinstance(item, EndOfVideo):
                break

        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual([item.frame_id for item in received], list(range(1, 7)))
        self.assertEqual(result_holder[0].status, "completed")
        self.assertGreater(result_holder[0].metrics.queue_full_count, 0)

    def test_stop_event_releases_a_blocked_producer(self) -> None:
        output_queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        producer = FrameTaskProducer(self.config, decoder=FakeDecoder(20))
        result_holder = []

        worker = threading.Thread(
            target=lambda: result_holder.append(
                producer.run("fake.mp4", output_queue, stop_event, "test-video")
            )
        )
        worker.start()
        time.sleep(0.05)
        stop_event.set()
        worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result_holder[0].status, "cancelled")


if __name__ == "__main__":
    unittest.main()
