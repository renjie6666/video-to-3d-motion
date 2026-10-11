"""Width-based screen normalization used by PoseMamba's H36M reader."""

import numpy as np

from .exceptions import PoseAdapterError


def normalize_h36m(keypoints: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    if (len(image_size) != 2 or any(type(v) is not int or v <= 0 for v in image_size)):
        raise PoseAdapterError("image_size_changed", "image dimensions must be positive integers")
    values = np.asarray(keypoints)
    if values.shape != (17, 2) or not np.issubdtype(values.dtype, np.floating):
        raise PoseAdapterError("invalid_pose2d_frame", "expected floating keypoints[17,2]")
    if not np.isfinite(values).all():
        raise PoseAdapterError("invalid_keypoints", "keypoints must be finite")
    width, height = image_size
    output = np.array(values, dtype=np.float32, order="C", copy=True)
    output[:, 0] = 2 * output[:, 0] / width - 1
    output[:, 1] = 2 * output[:, 1] / width - height / width
    if not np.isfinite(output).all():
        raise PoseAdapterError("invalid_keypoints", "normalization produced nonfinite coordinates")
    return output
