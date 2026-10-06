"""MMPose top-down adapter for RTMPose and ViTPose."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..exceptions import Pose2DError
from ..models import JointSchema, PosePrediction


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class MMPoseEstimator:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        *,
        joint_schema: JointSchema,
        score_source: str,
    ) -> None:
        try:
            from mmpose.apis import init_model
            from mmpose.utils import register_all_modules
        except ImportError as exc:
            raise Pose2DError(
                "openmmlab_import_failed", "MMPose is not installed"
            ) from exc
        register_all_modules()
        self.model = init_model(
            str(config_path), str(checkpoint_path), device="cuda:0"
        )
        self._joint_schema = joint_schema
        self._score_source = score_source

    @property
    def joint_schema(self) -> JointSchema:
        return self._joint_schema

    @property
    def score_source(self) -> str:
        return self._score_source

    def infer_batch(
        self,
        images: list[np.ndarray],
        bboxes: list[np.ndarray],
    ) -> list[PosePrediction]:
        try:
            from mmpose.apis import inference_topdown
        except ImportError as exc:
            raise Pose2DError(
                "openmmlab_import_failed", "MMPose is not installed"
            ) from exc
        predictions: list[PosePrediction] = []
        for image, bbox in zip(images, bboxes, strict=True):
            results = inference_topdown(
                self.model,
                image,
                bboxes=np.asarray(bbox, dtype=np.float32)[None, :],
                bbox_format="xyxy",
            )
            if len(results) != 1:
                raise Pose2DError(
                    "pose_inference_failed",
                    f"expected one pose result, got {len(results)}",
                )
            instances = results[0].pred_instances
            keypoints = _numpy(instances.keypoints)
            scores = _numpy(instances.keypoint_scores)
            if keypoints.ndim == 3:
                keypoints = keypoints[0]
            if scores.ndim == 2:
                scores = scores[0]
            predictions.append(
                PosePrediction(
                    keypoints=keypoints.astype(np.float32, copy=False),
                    keypoint_scores_raw=scores.astype(np.float32, copy=False),
                )
            )
        return predictions
