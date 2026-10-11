"""Explicit GT-root-depth camera reconstruction, separate from official P1.

Intrinsics: facebookresearch/VideoPose3D/common/h36m_dataset.py.
Image-depth convention: CHUNYUWANG/lcn-pose/tools/gendb.py, 2000 mm box.
This diagnostic requires GT root depth; it is not monocular metric-scale inference.
"""

import numpy as np


CAMERA_54138969 = dict(
    id="54138969", focal=[1145.0494384765625, 1143.7811279296875],
    center=[512.54150390625, 515.4514770507812],
    radial=[-.20709891617298126, .24777518212795258, -.0030751503072679043],
    tangential=[-.0009756988729350269, -.00142447161488235],
)


def project_camera(points, *, distorted=True):
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3 or not np.isfinite(points).all() or np.any(points[..., 2] <= 0):
        raise ValueError("finite camera coordinates with positive depth required")
    xy = points[..., :2] / points[..., 2:]
    if distorted:
        r2 = np.sum(xy * xy, axis=-1, keepdims=True)
        radial = 1 + np.sum(np.concatenate((r2, r2**2, r2**3), axis=-1)
                            * CAMERA_54138969["radial"], axis=-1, keepdims=True)
        tangent = np.sum(xy * CAMERA_54138969["tangential"], axis=-1, keepdims=True)
        xy = xy * (radial + tangent) + r2 * CAMERA_54138969["tangential"]
    return xy * CAMERA_54138969["focal"] + CAMERA_54138969["center"]


def undistort_pixels(pixels):
    observed = (np.asarray(pixels, dtype=np.float64) - CAMERA_54138969["center"]) / CAMERA_54138969["focal"]
    xy = observed.copy()
    for _ in range(30):
        r2 = np.sum(xy * xy, axis=-1, keepdims=True)
        radial = 1 + np.sum(np.concatenate((r2, r2**2, r2**3), axis=-1)
                            * CAMERA_54138969["radial"], axis=-1, keepdims=True)
        tangent = np.sum(xy * CAMERA_54138969["tangential"], axis=-1, keepdims=True)
        xy = (observed - r2 * CAMERA_54138969["tangential"]) / (radial + tangent)
    if not np.isfinite(xy).all():
        raise ValueError("camera undistortion failed")
    return xy * CAMERA_54138969["focal"] + CAMERA_54138969["center"]


def verify_camera_labels(camera_points, pixels):
    residual = np.linalg.norm(project_camera(camera_points) - pixels, axis=-1)
    if float(residual.max()) > .01:
        raise ValueError("camera intrinsics do not reproduce these 2D/3D labels")
    return dict(mean_projection_error_px=float(residual.mean()),
                max_projection_error_px=float(residual.max()))


def reconstruct_camera(poses_normalized, camera_gt, image_width):
    """Invert LCN image-3D with known intrinsics and GT absolute root anchor.

    Predicted XY remains unchanged; no optimal scale or rotation is fitted.
    The image plane here is pinhole, as in LCN's _weak_project function.
    """
    poses = np.asarray(poses_normalized, dtype=np.float64)
    target = np.asarray(camera_gt, dtype=np.float64)
    if poses.shape != target.shape or poses.ndim != 3 or poses.shape[1:] != (17, 3):
        raise ValueError("expected matching [N,17,3] poses")
    if not np.isfinite(poses).all() or not np.isfinite(target).all() or image_width <= 0:
        raise ValueError("invalid camera reconstruction inputs")
    root_depth = target[:, :1, 2:]
    if np.any(root_depth <= 0):
        raise ValueError("positive absolute GT root depth is required")
    # LCN: ratio=(box_right-box_left+1)/2000, using a 2000 mm box.
    ratio = CAMERA_54138969["focal"][0] / root_depth + 1 / 2000
    depth = root_depth + poses[..., 2:] * (image_width / 2) / ratio
    root_pixel = project_camera(target[:, :1], distorted=False)
    pixels = root_pixel + poses[..., :2] * (image_width / 2)
    camera_xy = (pixels - CAMERA_54138969["center"]) / CAMERA_54138969["focal"] * depth
    result = np.concatenate((camera_xy, depth), axis=-1)
    return result - result[:, :1]
