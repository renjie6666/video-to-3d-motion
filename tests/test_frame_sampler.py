from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_to_3d_motion.media.exceptions import MediaPipelineError
from video_to_3d_motion.media.frame_sampler import FrameSampler
from video_to_3d_motion.media.models import DecodedFrame


def make_frame(frame_id: int, timestamp: float) -> DecodedFrame:
    return DecodedFrame(
        image=np.zeros((4, 6, 3), dtype=np.uint8),
        source_frame_id=frame_id + 1,
        source_timestamp=timestamp,
        pts=frame_id,
        timestamp_source="pts",
    )


class FrameSamplerTests(unittest.TestCase):
    def test_50_fps_keeps_every_frame(self) -> None:
        sampler = FrameSampler(target_fps=50, tolerance_ms=10.0)
        output = []
        for index in range(10):
            output.extend(sampler.push(make_frame(index, index / 50)))
        output.extend(sampler.flush())

        self.assertEqual([item.frame_id for item in output], list(range(1, 11)))
        self.assertEqual(
            [item.decoded.source_frame_id for item in output], list(range(1, 11))
        )

    def test_60_fps_maps_to_50_fps_without_duplicates(self) -> None:
        sampler = FrameSampler(target_fps=50, tolerance_ms=10.0)
        output = []
        for index in range(61):
            output.extend(sampler.push(make_frame(index, index / 60)))
        output.extend(sampler.flush())

        self.assertEqual(len(output), 51)
        self.assertEqual([item.frame_id for item in output], list(range(1, 52)))
        source_ids = [item.decoded.source_frame_id for item in output]
        self.assertEqual(len(source_ids), len(set(source_ids)))

    def test_timestamp_gap_is_rejected(self) -> None:
        sampler = FrameSampler(target_fps=50, tolerance_ms=10.0)
        sampler.push(make_frame(0, 0.0))
        with self.assertRaises(MediaPipelineError) as context:
            sampler.push(make_frame(1, 0.04))
        self.assertEqual(context.exception.code, "timestamp_gap")


if __name__ == "__main__":
    unittest.main()
