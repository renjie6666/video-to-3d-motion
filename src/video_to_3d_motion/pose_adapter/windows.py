"""Offline full-sequence window preparation using the S1-03 cache rules."""

from dataclasses import replace
from pathlib import Path

from video_to_3d_motion.pose2d.models import EndOfPose2D

from .config import PoseAdapterConfig
from .exceptions import PoseAdapterError
from .jsonl_reader import iter_pose2d_jsonl
from .window_buffer import WindowBuffer


def prepare_sequence_windows(input_path: str | Path, summary_path: str | Path,
                             *, mode: str = "baseline", stride: int = 81):
    """Return windows only after upstream EOF/counts have been verified.

    Stored data are numeric poses and metadata, never decoded source images.
    Full-mode result memory grows with output length; the historical cache stays
    bounded to 243. A failed/partial replay cannot reach the inference backend.
    """
    if mode not in ("baseline", "full") or type(stride) is not int or stride != 81:
        raise ValueError("expected baseline/full with stride=81")
    buffer = WindowBuffer(PoseAdapterConfig())
    windows = []
    tail = None
    for item in iter_pose2d_jsonl(input_path, summary_path):
        if isinstance(item, EndOfPose2D):
            tail = buffer.finish(item)
            break
        buffer.push(item)
        count = buffer.metrics.frames_received
        if mode == "full" and count >= 243 and (count - 243) % stride == 0:
            windows.append(buffer.make_window(window_index=len(windows) + 1, pipeline_mode=mode))
    if tail is None:
        raise PoseAdapterError("metadata_mismatch", "missing upstream terminal message")
    if not windows or windows[-1].window_id != tail.window_id:
        windows.append(replace(tail, window_index=len(windows) + 1, pipeline_mode=mode))
    return windows, buffer.metrics
