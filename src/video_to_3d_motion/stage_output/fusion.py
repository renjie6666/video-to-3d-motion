"""Validated equal-weight overlap fusion and time-aware causal smoothing."""

from dataclasses import asdict

import numpy as np

from video_to_3d_motion.pose3d.config import OutputSettings
from video_to_3d_motion.pose3d.exceptions import Pose3DError
from video_to_3d_motion.pose3d.models import Pose3DSequence


def smooth_sequence(poses, timestamps, settings: OutputSettings):
    result = np.array(poses, dtype=np.float32, copy=True)
    if settings.smoothing == "ema":
        for index in range(1, len(result)):
            alpha = -np.expm1(-(timestamps[index] - timestamps[index - 1]) / settings.tau_seconds)
            result[index] = alpha * poses[index] + (1 - alpha) * result[index - 1]
    result[:, 0] = 0
    return result


def fuse_windows(windows, expected_frame_ids, settings: OutputSettings = OutputSettings()):
    expected = np.asarray(expected_frame_ids)
    if (expected.ndim != 1 or expected.dtype.kind not in "iu" or len(expected) < settings.obs_len
            or expected[0] < 1 or not (np.diff(expected) == 1).all()):
        raise Pose3DError("coverage_mismatch", "expected frame IDs must be an ordered contiguous sample sequence")
    expected = expected.astype(np.int64)
    count = len(expected)
    index_by_id = {int(frame_id): index for index, frame_id in enumerate(expected)}
    sums = np.zeros((count, 17, 3), dtype=np.float64)
    contributions = np.zeros(count, dtype=np.int64)
    metadata_arrays = dict(source_frame_ids=np.empty(count, dtype=np.int64),
                           timestamps=np.empty(count, dtype=np.float64), source_timestamps=np.empty(count, dtype=np.float64),
                           keypoint_scores_raw=np.empty((count, 17), dtype=np.float32),
                           valid_mask=np.empty((count, 17), dtype=np.bool_))
    timestamp_sources = [None] * count
    seen_windows = set()
    identity = None
    sequence_metadata = None
    for window in windows:
        inp = window.input_window
        if inp.window_id in seen_windows:
            raise Pose3DError("duplicate_window", f"duplicate window {inp.window_id}")
        seen_windows.add(inp.window_id)
        current_identity = (inp.video_id, inp.image_size, inp.model, inp.model_config, inp.checkpoint_sha256,
                            inp.score_source, inp.fps, inp.pipeline_mode,
                            window.model_info.get("checkpoint_sha256"), window.model_info.get("backend"),
                            window.model_info.get("flip_test"), window.model_info.get("units"))
        if identity is None:
            identity = current_identity
            sequence_metadata = dict(video_id=inp.video_id, image_size=list(inp.image_size),
                                     fps=inp.fps, fps_meaning="upstream_nominal; use timestamps for actual intervals",
                                     pipeline_mode=inp.pipeline_mode, joint_schema=inp.joint_schema,
                                     joint_order_id=inp.joint_order_id, joint_mapping=inp.joint_mapping,
                                     model_info=window.model_info, pose2d_model=inp.model,
                                     pose2d_checkpoint_sha256=inp.checkpoint_sha256,
                                     pose2d_model_config=inp.model_config, score_source=inp.score_source,
                                     coord_system="root_relative_image_3d", units="h36m_image_normalized",
                                     output_settings=asdict(settings))
        elif current_identity != identity:
            raise Pose3DError("sequence_mismatch", "cannot fuse different sequences or model settings")
        if (window.pose_3d.shape != (243, 17, 3) or window.pose_3d.dtype != np.float32
                or not np.isfinite(window.pose_3d).all() or not (window.pose_3d[:, 0] == 0).all()
                or inp.frame_ids.shape != (243,) or not (np.diff(inp.frame_ids) == 1).all()):
            raise Pose3DError("invalid_model_output", "expected finite root-relative float32[243,17,3]")
        try:
            indices = np.array([index_by_id[int(frame_id)] for frame_id in inp.frame_ids])
        except KeyError as exc:
            raise Pose3DError("coverage_mismatch", "a window lies outside the expected sample sequence") from exc
        for local, global_index in enumerate(indices):
            for name, target in metadata_arrays.items():
                value = getattr(inp, name)[local]
                if contributions[global_index] and not np.array_equal(target[global_index], value):
                    raise Pose3DError("metadata_mismatch", f"overlapping frame metadata disagree: {name}")
                target[global_index] = value
            source = inp.timestamp_sources[local]
            if contributions[global_index] and timestamp_sources[global_index] != source:
                raise Pose3DError("metadata_mismatch", "overlapping timestamp sources disagree")
            timestamp_sources[global_index] = source
        sums[indices] += window.pose_3d
        contributions[indices] += 1
    if (contributions == 0).any():
        raise Pose3DError("coverage_mismatch", "windows do not cover every expected sample")
    for name in ("source_frame_ids", "timestamps", "source_timestamps"):
        values = metadata_arrays[name]
        if not np.isfinite(values).all() or not (np.diff(values) > 0).all():
            raise Pose3DError("metadata_mismatch", f"fused {name} must increase")
    raw = (sums / contributions[:, None, None]).astype(np.float32)
    smoothed = smooth_sequence(raw, metadata_arrays["timestamps"], settings)
    sequence_metadata.update(window_count=len(seen_windows), fusion="equal_weight_mean",
                             timestamp_sources=timestamp_sources,
                             frame_count=count, first_frame_id=int(expected[0]), last_frame_id=int(expected[-1]),
                             first_source_frame_id=int(metadata_arrays["source_frame_ids"][0]),
                             last_source_frame_id=int(metadata_arrays["source_frame_ids"][-1]))
    return Pose3DSequence(smoothed, raw, expected, contribution_count=contributions,
                          metadata=sequence_metadata, **metadata_arrays)
