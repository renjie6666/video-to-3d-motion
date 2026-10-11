"""Build a local, dependency-free 3D skeleton viewer with embedded data."""

import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from video_to_3d_motion.pose_adapter.skeleton import H36M17_JOINT_NAMES as JOINT_NAMES


def _json_array(array):
    # Missing GT is null rather than JavaScript NaN or fabricated coordinates.
    values = np.asarray(array)
    if not np.isfinite(values).all():
        return json.loads(json.dumps(values.tolist()).replace("NaN", "null"))
    return values.round(6).tolist()


def create_viewer(directory, sequence, *, evaluation=None, image_dir=None, sequence_prefix=None):
    directory = Path(directory)
    thumbnails = [None] * len(sequence.frame_ids)
    if image_dir is not None:
        from PIL import Image
        image_dir = Path(image_dir)
        # Only inspect this sequence's filenames; the dataset root may contain
        # hundreds of thousands of unrelated images.
        for index, source_id in enumerate(sequence.source_frame_ids):
            path = next((candidate for extension in ("jpg", "jpeg", "png")
                         if (candidate := image_dir / f"{sequence_prefix}_{source_id:06d}.{extension}").is_file()), None)
            if path is not None:
                with Image.open(path) as image:
                    image.thumbnail((400, 400))
                    buffer = BytesIO()
                    image.convert("RGB").save(buffer, format="JPEG", quality=72)
                    thumbnails[index] = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    payload = dict(meta=sequence.metadata, frameIds=sequence.frame_ids.tolist(),
                   sourceIds=sequence.source_frame_ids.tolist(), timestamps=sequence.timestamps.tolist(),
                   sourceTimestamps=sequence.source_timestamps.tolist(), jointNames=JOINT_NAMES,
                   contributions=sequence.contribution_count.tolist(), images=thumbnails,
                   raw=_json_array(sequence.P_3d_raw), output=_json_array(sequence.P_3d),
                   metrics=None, frameMetrics=[])
    if evaluation is not None:
        metrics, rows, _, arrays = evaluation
        payload.update(metrics=metrics, frameMetrics=rows,
                       gt=_json_array(arrays["gt_camera_root_mm"]),
                       rawMM=_json_array(arrays["prediction_raw_gt_scale_mm"]),
                       outputMM=_json_array(arrays["prediction_output_gt_scale_mm"]),
                       rawPA=_json_array(arrays["prediction_raw_pa_mm"]),
                       outputPA=_json_array(arrays["prediction_output_pa_mm"]))
        if "prediction_output_camera_root_depth_mm" in arrays:
            payload.update(rawCameraMM=_json_array(arrays["prediction_raw_camera_root_depth_mm"]),
                           outputCameraMM=_json_array(arrays["prediction_output_camera_root_depth_mm"]))
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    (directory / "viewer.html").write_text(template.replace("/*__POSE_DATA__*/", encoded), encoding="utf-8")
    return directory / "viewer.html"
