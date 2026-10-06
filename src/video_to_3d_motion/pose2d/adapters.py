"""Validation and normalization at the S1-01/S1-02 boundary."""

from __future__ import annotations

import math

import numpy as np

from video_to_3d_motion.media.models import FrameTask

from .exceptions import Pose2DError
from .models import PersonBBox, PosePrediction


def validate_frame_task(task: FrameTask) -> None:
    if task.image.dtype != np.uint8 or task.image.ndim != 3:
        raise Pose2DError("invalid_frame_task", "image must be uint8[H,W,3]")
    if task.image.shape[2] != 3 or not task.image.flags.c_contiguous:
        raise Pose2DError(
            "invalid_frame_task", "image must be a contiguous BGR array"
        )
    height, width = task.image.shape[:2]
    if task.image_size != (width, height):
        raise Pose2DError("invalid_frame_task", "image_size does not match image")
    if task.frame_id < 1 or not math.isfinite(task.timestamp):
        raise Pose2DError("invalid_frame_task", "invalid frame id or timestamp")


def normalize_bbox(
    bbox: np.ndarray,
    *,
    width: int,
    height: int,
    min_area: float,
    score: float | None,
    source: str,
) -> PersonBBox:
    values = np.asarray(bbox, dtype=np.float32)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise Pose2DError("invalid_bbox", "bbox must be finite float32[4]")
    x1, y1, x2, y2 = (float(value) for value in values)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise Pose2DError("invalid_bbox", "bbox is outside the image or empty")
    if (x2 - x1) * (y2 - y1) < min_area:
        raise Pose2DError("invalid_bbox", "bbox area is below min_bbox_area")
    if score is not None and not math.isfinite(score):
        raise Pose2DError("invalid_bbox", "bbox score must be finite")
    return PersonBBox(values, score, source)  # type: ignore[arg-type]


def validate_prediction(prediction: PosePrediction) -> PosePrediction:
    keypoints = np.asarray(prediction.keypoints, dtype=np.float32)
    scores = np.asarray(prediction.keypoint_scores_raw, dtype=np.float32)
    if keypoints.shape != (17, 2):
        raise Pose2DError("joint_count_mismatch", "expected keypoints[17,2]")
    if scores.shape != (17,):
        raise Pose2DError("joint_count_mismatch", "expected scores[17]")
    if not np.isfinite(keypoints).all() or not np.isfinite(scores).all():
        raise Pose2DError("invalid_keypoints", "keypoints and scores must be finite")
    return PosePrediction(keypoints=keypoints, keypoint_scores_raw=scores)
