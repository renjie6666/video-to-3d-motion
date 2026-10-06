"""MMDetection adapter for the RTMDet person detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..exceptions import Pose2DError
from ..models import PersonBBox


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class MMDetPersonDetector:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        *,
        person_class_id: int = 0,
    ) -> None:
        try:
            from mmdet.apis import init_detector
            from mmdet.utils import register_all_modules
        except ImportError as exc:
            raise Pose2DError(
                "openmmlab_import_failed", "MMDetection is not installed"
            ) from exc
        register_all_modules()
        self.model = init_detector(
            str(config_path), str(checkpoint_path), device="cuda:0"
        )
        self.person_class_id = person_class_id

    def detect_batch(self, images: list[np.ndarray]) -> list[list[PersonBBox]]:
        try:
            from mmdet.apis import inference_detector
        except ImportError as exc:
            raise Pose2DError(
                "openmmlab_import_failed", "MMDetection is not installed"
            ) from exc
        output: list[list[PersonBBox]] = []
        for image in images:
            result = inference_detector(self.model, image)
            instances = result.pred_instances
            bboxes = _numpy(instances.bboxes).astype(np.float32, copy=False)
            scores = _numpy(instances.scores).astype(np.float32, copy=False)
            labels = _numpy(instances.labels).astype(np.int64, copy=False)
            output.append(
                [
                    PersonBBox(bbox=bbox, score=float(score), source="rtmdet")
                    for bbox, score, label in zip(
                        bboxes, scores, labels, strict=True
                    )
                    if int(label) == self.person_class_id
                ]
            )
        return output
