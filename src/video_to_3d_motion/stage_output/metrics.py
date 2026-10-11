"""Root-relative single-sequence metrics with explicitly labeled GT scale."""

import numpy as np

from video_to_3d_motion.pose3d.exceptions import Pose3DError
from .calibration import CAMERA_54138969, reconstruct_camera, verify_camera_labels


def align_scale(predicted, target):
    denominator = np.sum(predicted * predicted, axis=(-2, -1))
    if np.any(denominator <= 1e-14):
        raise Pose3DError("evaluation_invalid", "cannot fit scale to a degenerate predicted skeleton")
    scale = np.sum(predicted * target, axis=(-2, -1)) / denominator
    # A negative scale is a global reflection, not valid scale alignment.
    scale = np.maximum(scale, 0)
    return predicted * scale[:, None, None], scale


def align_procrustes(predicted, target):
    center_p = predicted.mean(axis=1, keepdims=True)
    center_t = target.mean(axis=1, keepdims=True)
    p, t = predicted - center_p, target - center_t
    norm_p = np.sqrt(np.sum(p * p, axis=(-2, -1), keepdims=True))
    norm_t = np.sqrt(np.sum(t * t, axis=(-2, -1), keepdims=True))
    if np.any(norm_p <= 1e-10) or np.any(norm_t <= 1e-10):
        raise Pose3DError("evaluation_invalid", "cannot align a degenerate skeleton")
    pn, tn = p / norm_p, t / norm_t
    covariance = np.swapaxes(pn, 1, 2) @ tn
    u, singular, vh = np.linalg.svd(covariance)
    sign = np.where(np.linalg.det(u @ vh) < 0, -1, 1)
    u[:, :, -1] *= sign[:, None]
    singular[:, -1] *= sign
    rotation = u @ vh
    scale = singular.sum(axis=-1)[:, None, None] * norm_t / norm_p
    return scale * (p @ rotation) + center_t


def acceleration(positions, timestamps):
    dt = np.diff(timestamps)
    if not np.isfinite(dt).all() or not (dt > 0).all():
        raise Pose3DError("evaluation_invalid", "timestamps must increase for motion metrics")
    if len(positions) < 3:
        return np.empty((0, 17, 3)), np.empty(0, dtype=np.int64)
    velocity = np.diff(positions, axis=0) / dt[:, None, None]
    return np.diff(velocity, axis=0) / ((dt[:-1] + dt[1:]) / 2)[:, None, None], np.arange(1, len(positions) - 1)


def _statistics(values):
    return dict(mean=float(np.mean(values)), median=float(np.median(values)), p95=float(np.percentile(values, 95)))


def apply_official_25d(evaluation, sequence, labels_path):
    """Add official P1/P2 using an explicitly supplied, frame-matched NPZ subset.

    The subset must be extracted from the original preprocessing pickle. This
    does not infer a factor from prediction/GT errors or calibrate online output.
    """
    with np.load(labels_path, allow_pickle=False) as labels:
        if not np.array_equal(labels["source_frame_ids"], sequence.source_frame_ids):
            raise Pose3DError("evaluation_invalid", "official labels have different source frames")
        if str(labels["sequence"]) != evaluation[0]["sequence"]:
            raise Pose3DError("evaluation_invalid", "official labels have a different sequence")
        factor = labels["factors"].astype(np.float64)
        target = labels["target_25d_mm"].astype(np.float64)
        source_sha256 = str(labels["original_pickle_sha256"])
    if (factor.shape != (len(sequence.frame_ids),) or target.shape != sequence.P_3d.shape
            or not np.isfinite(factor).all() or not (factor > 0).all()
            or not np.isfinite(target).all() or len(source_sha256) != 64):
        raise Pose3DError("evaluation_invalid", "invalid official 2.5D labels")
    target -= target[:, :1]
    metrics, rows, joints, arrays = evaluation
    metrics["official_25d"] = dict(labels=str(labels_path), original_pickle_sha256=source_sha256,
                                  scope="official P1/P2 formula on these source frames; not a full benchmark")
    metrics["mpjpe_mm_unavailable_reason"] = None
    arrays["gt_official_25d_mm"] = target
    for variant, poses in (("raw", sequence.P_3d_raw), ("output", sequence.P_3d)):
        pred = poses.astype(np.float64) * (sequence.metadata["image_size"][0] / 2) * factor[:, None, None]
        pred -= pred[:, :1]
        errors = np.linalg.norm(pred - target, axis=-1)
        metrics["variants"][variant]["mpjpe_official_25d_mm"] = _statistics(errors)
        metrics["variants"][variant]["pa_mpjpe_official_25d_mm"] = _statistics(
            np.linalg.norm(align_procrustes(pred, target) - target, axis=-1))
        arrays[f"prediction_{variant}_official_25d_mm"] = pred
        for row in rows:
            if row["variant"] == variant:
                index = int(np.flatnonzero(sequence.frame_ids == row["frame_id"])[0])
                row["mpjpe_official_25d_mm"] = float(errors[index].mean())
        for row in joints:
            if row["variant"] == variant:
                row["mpjpe_official_25d_mm"] = float(errors[:, row["joint_index"]].mean())
    metrics["mpjpe_mm"] = metrics["variants"]["output"]["mpjpe_official_25d_mm"]["mean"]
    return evaluation


def evaluate_sequence(sequence, annotations):
    matched = annotations.matched
    source_ids = sequence.source_frame_ids[matched]
    target = annotations.joints_camera_mm[matched]
    target = target - target[:, :1]
    pixels = annotations.joints_pixel[matched]
    pixels = pixels - pixels[:, :1]
    denom = np.sum(pixels * pixels, axis=(-2, -1))
    if np.any(denom <= 1e-10):
        raise Pose3DError("evaluation_invalid", "cannot estimate GT image scale from degenerate 2D labels")
    factor = np.sum(pixels * target[..., :2], axis=(-2, -1)) / denom
    if not np.isfinite(factor).all() or not (factor > 0).all():
        raise Pose3DError("evaluation_invalid", "invalid GT camera-to-image scale")
    model_scale_mm = sequence.metadata["image_size"][0] / 2 * factor
    width = sequence.metadata["image_size"][0]
    output = dict(
        sequence=annotations.provenance["sequence"], matched_frames=int(matched.sum()),
        total_output_frames=len(sequence.frame_ids), source_frame_ids=source_ids.tolist(),
        mpjpe_mm=None,
        mpjpe_mm_unavailable_reason="normalized image-3D output; official 2.5d_factor/joints_2.5d_image are absent from this H5",
        evaluation_scope="one sparse single-action single-camera sequence; not the official full H36M benchmark",
        joint_count=17, root_included=True,
        definitions={
            "n_mpjpe_mm": "root-relative camera GT; per-frame positive least-squares scale alignment only",
            "pa_mpjpe_mm": "per-frame scale/rotation/translation Procrustes; reflections forbidden",
            "mpjpe_gt_scale_mm": "approximate camera scale from GT 2D/3D XY; not uncalibrated P1 MPJPE",
            "acceleration_error_gt_scale_mm_s2": "central differences with real timestamps, using the same GT-aided approximate scale",
        },
        gt_scale_mm_per_pixel=_statistics(factor),
        gt_scale_projection_residual_px=float(np.mean(np.linalg.norm(target[..., :2] / factor[:, None, None] - pixels, axis=-1))),
        annotations=annotations.provenance,
        variants={},
    )
    temporal_dt = np.diff(sequence.timestamps)
    output["sampling"] = dict(interval_min_seconds=float(temporal_dt.min()),
                              interval_max_seconds=float(temporal_dt.max()),
                              median_effective_fps=float(1 / np.median(temporal_dt)),
                              nominal_fps=sequence.metadata["fps"], image_width=width)
    camera_verified = False
    if annotations.provenance["sequence"].endswith("." + CAMERA_54138969["id"]):
        try:
            calibration = verify_camera_labels(annotations.joints_camera_mm[matched], annotations.joints_pixel[matched])
            camera_verified = True
            output["camera_root_depth_calibration"] = dict(
                status="verified", camera=CAMERA_54138969, **calibration,
                depth_convention="LCN ICCV2019 2000mm box +1 pixel",
                uses_gt=["absolute root depth", "absolute root XY anchor"],
                prediction_alignment="root subtraction only; no fitted scale or rotation",
                scope="camera-space diagnostic; not official 2.5D P1 benchmark",
            )
            output["definitions"]["mpjpe_camera_root_depth_mm"] = "camera reconstruction using known intrinsics and GT root anchor; root-relative MPJPE, no fitted scale/rotation; not official 2.5D P1"
        except ValueError as exc:
            output["camera_root_depth_calibration"] = dict(status="unavailable", reason=str(exc))
    rows = []
    joint_rows = []
    arrays = dict(gt_camera_root_mm=np.full_like(annotations.joints_camera_mm, np.nan),
                  annotation_matched=matched.copy(), gt_scale_mm_per_normalized_unit=np.full(len(matched), np.nan))
    arrays["gt_camera_root_mm"][matched] = target
    arrays["gt_scale_mm_per_normalized_unit"][matched] = model_scale_mm
    for variant, all_poses in (("raw", sequence.P_3d_raw), ("output", sequence.P_3d)):
        predicted = all_poses[matched].astype(np.float64)
        scaled, fitted_scale = align_scale(predicted, target)
        aligned = align_procrustes(predicted, target)
        gt_scaled = predicted * model_scale_mm[:, None, None]
        errors = dict(n_mpjpe_mm=np.linalg.norm(scaled - target, axis=-1),
                      pa_mpjpe_mm=np.linalg.norm(aligned - target, axis=-1),
                      mpjpe_gt_scale_mm=np.linalg.norm(gt_scaled - target, axis=-1))
        if camera_verified:
            camera_prediction = reconstruct_camera(predicted, annotations.joints_camera_mm[matched], width)
            errors["mpjpe_camera_root_depth_mm"] = np.linalg.norm(camera_prediction - target, axis=-1)
            arrays[f"prediction_{variant}_camera_root_depth_mm"] = np.full_like(arrays["gt_camera_root_mm"], np.nan)
            arrays[f"prediction_{variant}_camera_root_depth_mm"][matched] = camera_prediction
        stats = {name: dict(_statistics(values.mean(axis=1)), non_root16_mean=float(values[:, 1:].mean()))
                 for name, values in errors.items()}
        # Use only triples of adjacent output samples that all have labels.
        full_gt_scaled = all_poses.astype(np.float64) * arrays["gt_scale_mm_per_normalized_unit"][:, None, None]
        pred_acc, centers = acceleration(full_gt_scaled, sequence.timestamps)
        gt_acc, _ = acceleration(arrays["gt_camera_root_mm"], sequence.timestamps)
        triples = matched[:-2] & matched[1:-1] & matched[2:]
        acceleration_error = np.linalg.norm(pred_acc[triples] - gt_acc[triples], axis=-1).mean(axis=1)
        stats["acceleration_error_gt_scale_mm_s2"] = _statistics(acceleration_error) if len(acceleration_error) else None
        raw_acc, _ = acceleration(all_poses, sequence.timestamps)
        stats["acceleration_normalized_s2"] = _statistics(np.linalg.norm(raw_acc, axis=-1).mean(axis=1))
        # Standard 16 bone segments, measured before any GT fitting.
        parents = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
        lengths = np.linalg.norm(all_poses[:, 1:] - all_poses[:, parents], axis=-1)
        means = lengths.mean(axis=0)
        stats["bone_length_cv_mean"] = float(np.mean(lengths.std(axis=0) / np.maximum(means, 1e-12)))
        output["variants"][variant] = stats
        arrays[f"prediction_{variant}_gt_scale_mm"] = full_gt_scaled
        arrays[f"prediction_{variant}_pa_mm"] = np.full_like(arrays["gt_camera_root_mm"], np.nan)
        arrays[f"prediction_{variant}_pa_mm"][matched] = aligned
        matched_positions = np.flatnonzero(matched)
        for local, position in enumerate(matched_positions):
            rows.append(dict(variant=variant, frame_id=int(sequence.frame_ids[position]),
                             source_frame_id=int(source_ids[local]), timestamp=float(sequence.timestamps[position]),
                             **{name: float(values[local].mean()) for name, values in errors.items()}))
        for joint in range(17):
            joint_rows.append(dict(variant=variant, joint_index=joint,
                                   **{name: float(values[:, joint].mean()) for name, values in errors.items()}))
    return output, rows, joint_rows, arrays
