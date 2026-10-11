"""CPU contracts between inference and sequence output."""

from dataclasses import dataclass

import numpy as np

from video_to_3d_motion.pose_adapter.models import Pose2DWindow


@dataclass(frozen=True, slots=True)
class Pose3DWindow:
    pose_3d: np.ndarray
    input_window: Pose2DWindow
    model_info: dict
    inference_seconds: float


@dataclass(frozen=True, slots=True)
class Pose3DSequence:
    P_3d: np.ndarray
    P_3d_raw: np.ndarray
    frame_ids: np.ndarray
    source_frame_ids: np.ndarray
    timestamps: np.ndarray
    source_timestamps: np.ndarray
    keypoint_scores_raw: np.ndarray
    valid_mask: np.ndarray
    contribution_count: np.ndarray
    metadata: dict
