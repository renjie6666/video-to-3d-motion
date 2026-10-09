"""Dependency-light pose overlay rendering using NumPy and PyAV."""

from __future__ import annotations

import json
import re
from pathlib import Path

import av
import numpy as np

from video_to_3d_motion.media.models import FrameTask

from .config import VisualizationConfig
from .exceptions import Pose2DError
from .models import Pose2DFrame

COCO17_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)

H36M17_EDGES = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7),
    (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
)

_DIGITS = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "F": ("111", "100", "110", "100", "100"),
    "G": ("111", "100", "101", "101", "111"),
    "T": ("111", "010", "010", "010", "010"),
    "P": ("110", "101", "110", "100", "100"),
    "R": ("110", "101", "110", "101", "101"),
    "E": ("111", "100", "110", "100", "111"),
    "D": ("110", "101", "101", "101", "110"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("111", "101", "101", "101", "111"),
    ":": ("0", "1", "0", "1", "0"),
}


def _paint_disk(
    image: np.ndarray, x: int, y: int, radius: int, color: tuple[int, int, int]
) -> None:
    height, width = image.shape[:2]
    x1, x2 = max(0, x - radius), min(width - 1, x + radius)
    y1, y2 = max(0, y - radius), min(height - 1, y + radius)
    if x1 > x2 or y1 > y2:
        return
    yy, xx = np.ogrid[y1 : y2 + 1, x1 : x2 + 1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    region = image[y1 : y2 + 1, x1 : x2 + 1]
    region[mask] = color


def _draw_line(
    image: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1 = start
    x2, y2 = end
    steps = max(1, int(max(abs(x2 - x1), abs(y2 - y1))) + 1)
    xs = np.rint(np.linspace(x1, x2, steps)).astype(np.int32)
    ys = np.rint(np.linspace(y1, y2, steps)).astype(np.int32)
    radius = max(0, thickness // 2)
    for x, y in zip(xs, ys, strict=True):
        _paint_disk(image, int(x), int(y), radius, color)


def _draw_rectangle(
    image: np.ndarray, bbox: np.ndarray, color: tuple[int, int, int]
) -> None:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    _draw_line(image, (x1, y1), (x2, y1), color)
    _draw_line(image, (x2, y1), (x2, y2), color)
    _draw_line(image, (x2, y2), (x1, y2), color)
    _draw_line(image, (x1, y2), (x1, y1), color)


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: int = 2,
) -> None:
    x_cursor, y_origin = origin
    for character in text:
        pattern = _DIGITS.get(character)
        if pattern is None:
            x_cursor += 4 * scale
            continue
        for row, line in enumerate(pattern):
            for column, value in enumerate(line):
                if value == "1":
                    x1 = x_cursor + column * scale
                    y1 = y_origin + row * scale
                    image[y1 : y1 + scale, x1 : x1 + scale] = color
        x_cursor += (len(pattern[0]) + 1) * scale


def draw_pose2d(
    image: np.ndarray,
    pose: Pose2DFrame,
    config: VisualizationConfig,
) -> np.ndarray:
    """Draw a pose on a caller-owned BGR image and return that image."""
    edges = {
        "coco17": COCO17_EDGES,
        "h36m17": H36M17_EDGES,
    }.get(pose.joint_schema)
    if edges is None:
        raise Pose2DError(
            "visualization_schema_missing",
            f"no skeleton edges for {pose.joint_schema}",
        )

    if config.draw_bbox:
        _draw_rectangle(image, pose.bbox, (255, 160, 0))
    if config.draw_skeleton:
        for first, second in edges:
            if pose.valid_mask[first] and pose.valid_mask[second]:
                _draw_line(
                    image,
                    tuple(pose.keypoints[first]),
                    tuple(pose.keypoints[second]),
                    (255, 255, 0),
                    thickness=2,
                )
    for index, (x, y) in enumerate(pose.keypoints):
        color = (0, 220, 0) if pose.valid_mask[index] else (0, 0, 255)
        _paint_disk(image, int(round(float(x))), int(round(float(y))), 3, color)

    if config.draw_frame_info:
        label = f"F:{pose.frame_id:06d}"
        banner_width = len(label) * 8 + 6
        image[2:16, 2 : min(image.shape[1], banner_width)] = (0, 0, 0)
        _draw_text(image, label, (5, 4), scale=2)
    return image


def _write_image(image: np.ndarray, path: Path, config: VisualizationConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    codec = "mjpeg" if config.image_format == "jpg" else "png"
    pixel_format = "yuvj420p" if codec == "mjpeg" else "rgb24"
    try:
        container = av.open(str(temporary), mode="w", format="image2")
        stream = container.add_stream(codec, rate=1)
        stream.width = int(image.shape[1])
        stream.height = int(image.shape[0])
        stream.pix_fmt = pixel_format
        if codec == "mjpeg":
            qscale = max(2, min(31, round((101 - config.jpeg_quality) / 3.3)))
            stream.codec_context.options = {"qscale": str(qscale)}
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        temporary.replace(path)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        raise Pose2DError(
            "visualization_write_failed", f"cannot write {path}: {exc}"
        ) from exc


def _safe_video_id(video_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", video_id).strip("._")
    return cleaned or "video"


class Pose2DVisualizer:
    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config

    def should_render(self, frame_id: int) -> bool:
        return self.config.enabled and (
            (frame_id - 1) % self.config.every_n_frames == 0
        )

    def write(self, task: FrameTask, pose: Pose2DFrame) -> Path | None:
        if not self.should_render(task.frame_id):
            return None
        if task.video_id != pose.video_id or task.frame_id != pose.frame_id:
            raise Pose2DError(
                "metadata_mismatch", "visualization input and pose do not match"
            )
        video_dir = self.config.output_root / _safe_video_id(task.video_id)
        suffix = ".jpg" if self.config.image_format == "jpg" else ".png"
        image_path = video_dir / f"frame_{task.frame_id:06d}{suffix}"
        canvas = np.ascontiguousarray(task.image.copy())
        draw_pose2d(canvas, pose, self.config)
        _write_image(canvas, image_path, self.config)

        record = {
            "video_id": pose.video_id,
            "frame_id": pose.frame_id,
            "source_frame_id": pose.source_frame_id,
            "image_size": list(pose.image_size),
            "path": image_path.name,
            "bbox": pose.bbox.tolist(),
            "joint_schema": pose.joint_schema,
            "model": pose.model,
        }
        manifest = video_dir / "manifest.jsonl"
        with manifest.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        pose.visualization_path = image_path
        return image_path
