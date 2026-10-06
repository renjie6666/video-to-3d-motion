"""Data contracts for S1-02 detection, pose estimation, and output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

Float32Array: TypeAlias = NDArray[np.float32]
BoolArray: TypeAlias = NDArray[np.bool_]
JointSchema: TypeAlias = Literal["coco17", "h36m17"]


@dataclass(frozen=True, slots=True)
class ExternalBBox:
    video_id: str
    frame_id: int
    bbox: Float32Array
    score: float | None = None
    source: str = "external"


@dataclass(frozen=True, slots=True)
class PersonBBox:
    bbox: Float32Array
    score: float | None
    source: Literal["external", "rtmdet"]


@dataclass(frozen=True, slots=True)
class PosePrediction:
    keypoints: Float32Array
    keypoint_scores_raw: Float32Array


@dataclass(slots=True)
class Pose2DFrame:
    keypoints: Float32Array
    keypoint_scores_raw: Float32Array
    valid_mask: BoolArray
    joint_schema: JointSchema
    score_source: str
    bbox: Float32Array
    bbox_score: float | None
    bbox_source: str
    video_id: str
    frame_id: int
    source_frame_id: int
    timestamp: float
    source_timestamp: float
    timestamp_source: str
    image_size: tuple[int, int]
    fps: int
    model: str
    model_config: str
    checkpoint_sha256: str
    visualization_path: Path | None = None


@dataclass(frozen=True, slots=True)
class EndOfPose2D:
    video_id: str
    frames_received: int
    frames_succeeded: int
    frames_failed: int
    batches_processed: int


@dataclass(frozen=True, slots=True)
class Pose2DFailure:
    video_id: str | None
    frame_id: int | None
    error_code: str
    message: str
    recoverable: bool


Pose2DOutputItem: TypeAlias = Pose2DFrame | EndOfPose2D | Pose2DFailure


@dataclass(slots=True)
class Pose2DMetrics:
    frames_received: int = 0
    frames_succeeded: int = 0
    frames_failed: int = 0
    batches_processed: int = 0
    visualization_frames_written: int = 0


@dataclass(slots=True)
class Pose2DResult:
    video_id: str | None
    status: Literal["completed", "cancelled", "failed"]
    metrics: Pose2DMetrics
    error_code: str | None = None
    message: str | None = None
