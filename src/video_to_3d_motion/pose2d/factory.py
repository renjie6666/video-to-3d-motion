"""Build the GPU-only S1-02 runtime."""

from __future__ import annotations

import torch

from .backends import MMDetPersonDetector, MMPoseEstimator
from .config import Pose2DConfig
from .exceptions import Pose2DError
from .worker import Pose2DWorker


def check_gpu_environment() -> dict[str, str]:
    if not torch.cuda.is_available():
        raise Pose2DError(
            "cuda_unavailable", "S1-02 requires CUDA but PyTorch cannot access it"
        )
    if torch.cuda.device_count() < 1:
        raise Pose2DError("cuda_device_not_found", "CUDA device 0 does not exist")
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
    }


def create_gpu_worker(config: Pose2DConfig) -> Pose2DWorker:
    check_gpu_environment()
    model = config.model
    detection = config.detection
    if model.config_path is None or not model.config_path.is_file():
        raise Pose2DError("model_config_not_found", "MMPose config does not exist")
    if model.checkpoint_path is None or not model.checkpoint_path.is_file():
        raise Pose2DError("checkpoint_not_found", "MMPose checkpoint does not exist")
    if detection.detector_config is None or not detection.detector_config.is_file():
        raise Pose2DError("model_config_not_found", "RTMDet config does not exist")
    if (
        detection.detector_checkpoint is None
        or not detection.detector_checkpoint.is_file()
    ):
        raise Pose2DError("checkpoint_not_found", "RTMDet checkpoint does not exist")

    detector = MMDetPersonDetector(
        detection.detector_config,
        detection.detector_checkpoint,
        person_class_id=detection.person_class_id,
    )
    estimator = MMPoseEstimator(
        model.config_path,
        model.checkpoint_path,
        joint_schema=model.joint_schema,
        score_source=model.score_source,
    )
    return Pose2DWorker(
        config,
        detector=detector,
        estimator=estimator,
    )
