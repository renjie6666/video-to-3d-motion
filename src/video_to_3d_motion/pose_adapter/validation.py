"""Single-sequence metadata and frame-order validation."""

from __future__ import annotations

import math

import numpy as np

from video_to_3d_motion.pose2d.models import Pose2DFrame

from .exceptions import PoseAdapterError
from .skeleton import validate_joint_profile


def validate_frame(pose: Pose2DFrame, previous: Pose2DFrame | None) -> float | None:
    if not isinstance(pose, Pose2DFrame):
        raise PoseAdapterError("invalid_pose2d_frame", "expected Pose2DFrame")
    for name in ("keypoints", "keypoint_scores_raw", "valid_mask"):
        if not isinstance(getattr(pose, name), np.ndarray):
            raise PoseAdapterError("invalid_pose2d_frame", f"{name} must be a numpy array")
    if (pose.keypoints.shape != (17, 2) or pose.keypoints.dtype != np.float32
            or pose.keypoint_scores_raw.shape != (17,) or pose.keypoint_scores_raw.dtype != np.float32
            or pose.valid_mask.shape != (17,) or pose.valid_mask.dtype != np.bool_):
        raise PoseAdapterError("invalid_pose2d_frame", "expected float32[17,2], float32[17], bool[17]")
    if not np.isfinite(pose.keypoints).all() or not np.isfinite(pose.keypoint_scores_raw).all():
        raise PoseAdapterError("invalid_keypoints", "coordinates and scores must be finite")
    if (not isinstance(pose.image_size, tuple) or len(pose.image_size) != 2
            or any(type(v) is not int or v <= 0 for v in pose.image_size)):
        raise PoseAdapterError("image_size_changed", "image_size must be positive integer (W,H)")
    for name in ("video_id", "model", "model_config", "score_source", "timestamp_source"):
        if not isinstance(getattr(pose, name), str) or not getattr(pose, name):
            raise PoseAdapterError("invalid_pose2d_frame", f"{name} must be a nonempty string")
    if type(pose.frame_id) is not int or pose.frame_id != (previous.frame_id + 1 if previous else 1):
        raise PoseAdapterError("frame_order_invalid", f"unexpected target frame_id {pose.frame_id}")
    if type(pose.source_frame_id) is not int or pose.source_frame_id < 1:
        raise PoseAdapterError("timestamp_invalid", "invalid source frame_id")
    for value in (pose.timestamp, pose.source_timestamp):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise PoseAdapterError("timestamp_invalid", "timestamps must be finite numbers")
    if type(pose.fps) is not int or pose.fps <= 0:
        raise PoseAdapterError("invalid_pose2d_frame", "fps metadata must be a positive integer")
    interval = None
    if previous is not None:
        if pose.video_id != previous.video_id:
            raise PoseAdapterError("sequence_mismatch", "cannot mix video IDs")
        if pose.image_size != previous.image_size:
            raise PoseAdapterError("image_size_changed", "image dimensions changed")
        for name in ("model", "model_config", "checkpoint_sha256", "score_source", "joint_schema"):
            if getattr(pose, name) != getattr(previous, name):
                raise PoseAdapterError("model_identity_changed", f"{name} changed within sequence")
        interval = pose.timestamp - previous.timestamp
        if (interval <= 0 or pose.source_timestamp <= previous.source_timestamp
                or pose.source_frame_id <= previous.source_frame_id):
            raise PoseAdapterError("timestamp_invalid", "timestamps and source frame IDs must increase")
    validate_joint_profile(pose)
    return interval
