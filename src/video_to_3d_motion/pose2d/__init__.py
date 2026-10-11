"""S1-02 single-person detection and 2D pose estimation."""

from .config import Pose2DConfig, load_pose2d_config
from .models import EndOfPose2D, Pose2DFailure, Pose2DFrame
from .worker import Pose2DWorker


def create_gpu_worker(config: Pose2DConfig) -> Pose2DWorker:
    """Load the GPU factory only when a runtime, rather than a record, is needed."""
    from .factory import create_gpu_worker as create

    return create(config)

__all__ = [
    "EndOfPose2D",
    "Pose2DConfig",
    "Pose2DFailure",
    "Pose2DFrame",
    "Pose2DWorker",
    "create_gpu_worker",
    "load_pose2d_config",
]
