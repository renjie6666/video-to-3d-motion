"""Bounded historical cache, independent from the input queue capacity."""

from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from video_to_3d_motion.pose2d.models import EndOfPose2D, Pose2DFrame

from .config import PoseAdapterConfig
from .exceptions import PoseAdapterError
from .models import Pose2DWindow, PoseAdapterMetrics
from .normalization import normalize_h36m
from .validation import validate_frame


class WindowBuffer:
    def __init__(self, config: PoseAdapterConfig) -> None:
        self.config = config
        self.frames: deque[tuple[Pose2DFrame, np.ndarray]] = deque(maxlen=config.pose_adapter.clip_len)
        self.metrics = PoseAdapterMetrics()
        self.last: Pose2DFrame | None = None
        self.finished = False

    @property
    def video_id(self) -> str | None:
        return self.last.video_id if self.last else None

    def push(self, pose: Pose2DFrame) -> None:
        if self.finished:
            raise PoseAdapterError("metadata_mismatch", "cannot append after terminal message")
        interval = validate_frame(pose, self.last)
        snapshot = replace(pose, keypoints=pose.keypoints.copy(),
                           keypoint_scores_raw=pose.keypoint_scores_raw.copy(), valid_mask=pose.valid_mask.copy())
        self.frames.append((snapshot, normalize_h36m(snapshot.keypoints, snapshot.image_size)))
        self.last = snapshot
        self.metrics.frames_received += 1
        self.metrics.low_confidence_points += int(np.count_nonzero(~snapshot.valid_mask))
        self.metrics.max_buffer_size = max(self.metrics.max_buffer_size, len(self.frames))
        if interval is not None:
            minimum = self.metrics.interval_min_seconds
            maximum = self.metrics.interval_max_seconds
            self.metrics.interval_min_seconds = interval if minimum is None else min(minimum, interval)
            self.metrics.interval_max_seconds = interval if maximum is None else max(maximum, interval)

    def finish(self, terminal: EndOfPose2D) -> Pose2DWindow:
        if self.finished:
            raise PoseAdapterError("metadata_mismatch", "duplicate terminal message")
        count = self.metrics.frames_received
        if (not isinstance(terminal.video_id, str) or not terminal.video_id
                or any(type(getattr(terminal, name)) is not int or getattr(terminal, name) < 0
                       for name in ("frames_received", "frames_succeeded", "frames_failed", "batches_processed"))
                or terminal.video_id != (self.video_id or terminal.video_id)
                or terminal.frames_succeeded != count or terminal.frames_received != count
                or terminal.frames_failed != 0):
            raise PoseAdapterError("metadata_mismatch", "upstream terminal identity or counts do not match")
        if len(self.frames) < self.config.pose_adapter.clip_len:
            raise PoseAdapterError("insufficient_frames", f"need 243 frames, received {count}")
        self.finished = True
        return self.make_window()

    def make_window(self, *, window_index: int = 1, pipeline_mode: str = "baseline") -> Pose2DWindow:
        """Snapshot the current verified cache for offline baseline/full callers."""
        if len(self.frames) < self.config.pose_adapter.clip_len:
            raise PoseAdapterError("insufficient_frames", "need 243 frames to snapshot a window")
        poses, normalized = zip(*self.frames)
        last = poses[-1]
        return Pose2DWindow(
            window_id=f"{last.video_id}:{poses[0].frame_id}-{last.frame_id}",
            video_id=last.video_id, window_index=window_index, pipeline_mode=pipeline_mode,
            keypoints_norm=np.ascontiguousarray(np.stack(normalized)[None], dtype=np.float32),
            frame_ids=np.array([p.frame_id for p in poses], dtype=np.int64),
            source_frame_ids=np.array([p.source_frame_id for p in poses], dtype=np.int64),
            timestamps=np.array([p.timestamp for p in poses], dtype=np.float64),
            source_timestamps=np.array([p.source_timestamp for p in poses], dtype=np.float64),
            timestamp_sources=tuple(p.timestamp_source for p in poses),
            keypoint_scores_raw=np.stack([p.keypoint_scores_raw for p in poses]),
            valid_mask=np.stack([p.valid_mask for p in poses]),
            image_size=last.image_size, fps=last.fps, model=last.model,
            model_config=last.model_config, checkpoint_sha256=last.checkpoint_sha256,
            score_source=last.score_source, cutoff_timestamp=last.timestamp,
        )
