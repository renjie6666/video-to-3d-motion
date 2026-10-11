"""Load one H36M H5 sequence, joining names by row and source frame ID."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from video_to_3d_motion.pose3d.backend import file_sha256
from video_to_3d_motion.pose3d.exceptions import Pose3DError

# This permutation is for the user's H5 labels, never for model predictions.
H5_TO_H36M17 = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


@dataclass(frozen=True, slots=True)
class SequenceAnnotations:
    joints_camera_mm: np.ndarray
    joints_pixel: np.ndarray
    matched: np.ndarray
    source_frame_ids: np.ndarray
    row_indices: np.ndarray
    provenance: dict


def load_annotations(h5_path, names_path, sequence, source_frame_ids):
    try:
        import h5py
    except ImportError as exc:
        raise Pose3DError("dependency_missing", "h5py is required for h36m_annot evaluation") from exc
    h5_path, names_path = Path(h5_path), Path(names_path)
    requested = np.asarray(source_frame_ids, dtype=np.int64)
    names = names_path.read_text(encoding="utf-8-sig").splitlines()
    by_id = {}
    pattern = re.compile(re.escape(sequence) + r"_(\d{6})\.(?:jpg|jpeg|png)$", re.IGNORECASE)
    for row, name in enumerate(names):
        match = pattern.fullmatch(name.replace("\\", "/").split("/")[-1])
        if match:
            source_id = int(match[1])
            if source_id in by_id:
                raise Pose3DError("annotation_invalid", "duplicate source frame in annotation names")
            by_id[source_id] = row
    if not by_id:
        raise Pose3DError("annotation_mismatch", f"sequence {sequence!r} not found in {names_path}")
    rows = np.array([by_id.get(int(source_id), -1) for source_id in requested], dtype=np.int64)
    matched = rows >= 0
    if not matched.any():
        raise Pose3DError("annotation_mismatch", "no output source frames match annotations")
    joints3d = np.full((len(requested), 17, 3), np.nan, dtype=np.float64)
    joints2d = np.full((len(requested), 17, 2), np.nan, dtype=np.float64)
    try:
        with h5py.File(h5_path, "r") as dataset:
            if (dataset["S"].shape != (len(names), 17, 3)
                    or dataset["part"].shape != (len(names), 17, 2)):
                raise Pose3DError("annotation_invalid", "H5 arrays and names.txt rows do not match")
            # h5py fancy indexing requires increasing row indices, irrespective
            # of how the names file orders sequences or frames.
            positions = np.flatnonzero(matched)
            order = np.argsort(rows[matched])
            selected = rows[matched][order]
            if np.any(dataset["normalize"][selected] != 0):
                raise Pose3DError("annotation_invalid", "expected original pixel annotations, normalize=0")
            joints3d[positions[order]] = dataset["S"][selected][:, H5_TO_H36M17]
            joints2d[positions[order]] = dataset["part"][selected][:, H5_TO_H36M17]
    except (KeyError, OSError, ValueError) as exc:
        raise Pose3DError("annotation_invalid", str(exc)) from exc
    if not np.isfinite(joints3d[matched]).all() or not np.isfinite(joints2d[matched]).all():
        raise Pose3DError("annotation_invalid", "GT contains nonfinite coordinates")
    # Root-to-knee distance guards accidental meter/normalized H5 inputs.
    leg_scale = np.linalg.norm(joints3d[matched, 2] - joints3d[matched, 1], axis=-1)
    if not 50 < np.median(leg_scale) < 1000:
        raise Pose3DError("annotation_invalid", "H5 S is not consistent with camera-space millimeter labels")
    return SequenceAnnotations(joints3d, joints2d, matched, requested, rows, dict(
        sequence=sequence, annotation_h5=str(h5_path.resolve()), annotation_names=str(names_path.resolve()),
        annotation_h5_sha256=file_sha256(h5_path), annotation_names_sha256=file_sha256(names_path),
        h5_to_h36m17=H5_TO_H36M17, coordinate_space="camera", units="mm",
        matched_frames=int(matched.sum()), missing_source_frame_ids=requested[~matched].tolist(),
        source_frame_ids=requested.tolist(), row_indices=rows.tolist(),
        visibility="all labeled joints; no per-joint visibility labels in this H5",
        alignment="sequence filename prefix + source_frame_id; names.txt row selects H5 row",
    ))
