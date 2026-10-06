"""Stable interfaces that isolate S1-02 from model framework internals."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..models import JointSchema, PersonBBox, PosePrediction


class PersonDetector(Protocol):
    def detect_batch(
        self, images: list[np.ndarray]
    ) -> list[list[PersonBBox]]: ...


class PoseEstimator(Protocol):
    @property
    def joint_schema(self) -> JointSchema: ...

    @property
    def score_source(self) -> str: ...

    def infer_batch(
        self,
        images: list[np.ndarray],
        bboxes: list[np.ndarray],
    ) -> list[PosePrediction]: ...
