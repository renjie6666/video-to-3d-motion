"""CPU window and control-message contracts for S1-03/S1-04."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from video_to_3d_motion.pose2d.models import Pose2DOutputItem

PoseAdapterInputItem: TypeAlias = Pose2DOutputItem


@dataclass(frozen=True, slots=True)
class Pose2DWindow:
    window_id: str
    video_id: str
    window_index: int
    keypoints_norm: NDArray[np.float32]
    frame_ids: NDArray[np.int64]
    source_frame_ids: NDArray[np.int64]
    timestamps: NDArray[np.float64]
    source_timestamps: NDArray[np.float64]
    timestamp_sources: tuple[str, ...]
    keypoint_scores_raw: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    image_size: tuple[int, int]
    fps: int  # Upstream nominal FPS; timestamps retain the actual sampling times.
    model: str
    model_config: str
    checkpoint_sha256: str
    score_source: str
    cutoff_timestamp: float
    coordinate_space: str = "h36m_screen_normalized"
    normalization_id: str = "h36m_width_center_v1"
    joint_schema: str = "h36m17"
    joint_order_id: str = "h36m17_rtmpose_h36m_v1"
    joint_mapping: str = "identity"
    pipeline_mode: str = "baseline"


@dataclass(frozen=True, slots=True)
class EndOfPoseAdapter:
    video_id: str
    frames_received: int
    windows_emitted: int
    first_frame_id: int
    last_frame_id: int


@dataclass(frozen=True, slots=True)
class PoseAdapterFailure:
    video_id: str | None
    frame_id: int | None
    error_code: str
    message: str
    recoverable: bool = False


PoseAdapterOutputItem: TypeAlias = Pose2DWindow | EndOfPoseAdapter | PoseAdapterFailure


@dataclass(slots=True)
class PoseAdapterMetrics:
    frames_received: int = 0
    windows_emitted: int = 0
    low_confidence_points: int = 0
    max_buffer_size: int = 0
    publish_block_seconds: float = 0.0
    interval_min_seconds: float | None = None
    interval_max_seconds: float | None = None


@dataclass(slots=True)
class PoseAdapterResult:
    video_id: str | None
    status: Literal["completed", "failed", "cancelled"]
    metrics: PoseAdapterMetrics = field(default_factory=PoseAdapterMetrics)
    error_code: str | None = None
    message: str | None = None
