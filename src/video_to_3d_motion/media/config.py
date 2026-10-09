"""Configuration models for media decoding and queue backpressure."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MediaConfig:
    target_fps: int = 50
    pixel_format: str = "bgr24"
    fps_tolerance: float = 0.5
    timestamp_tolerance_ms: float = 10.0
    fail_on_low_fps: bool = True
    fallback_when_pts_missing: bool = True

    def __post_init__(self) -> None:
        if self.target_fps <= 0:
            raise ValueError("media.target_fps must be positive")
        if self.fps_tolerance < 0:
            raise ValueError("media.fps_tolerance cannot be negative")
        if self.timestamp_tolerance_ms < 0:
            raise ValueError("media.timestamp_tolerance_ms cannot be negative")
        if self.pixel_format != "bgr24":
            raise ValueError("the current MMPose boundary requires pixel_format=bgr24")


@dataclass(frozen=True, slots=True)
class ImageSequenceConfig:
    source_fps: float = 50.0
    frame_pattern: str = r"_(?P<frame_id>\d{6})\.(?:jpg|jpeg|png)$"
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    require_contiguous: bool = True

    def __post_init__(self) -> None:
        if self.source_fps <= 0:
            raise ValueError("image_sequence.source_fps must be positive")
        if "frame_id" not in re.compile(self.frame_pattern).groupindex:
            raise ValueError(
                "image_sequence.frame_pattern must contain a named frame_id group"
            )
        normalized = tuple(
            suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
            for suffix in self.extensions
        )
        if not normalized:
            raise ValueError("image_sequence.extensions cannot be empty")
        object.__setattr__(self, "extensions", normalized)


@dataclass(frozen=True, slots=True)
class QueueConfig:
    batch_size: int = 27
    prefetch_batches: int = 4
    capacity: int | None = None
    put_timeout_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("queue.batch_size must be positive")
        if self.prefetch_batches <= 0:
            raise ValueError("queue.prefetch_batches must be positive")
        if self.put_timeout_seconds <= 0:
            raise ValueError("queue.put_timeout_seconds must be positive")
        if self.capacity is None:
            object.__setattr__(self, "capacity", self.batch_size * self.prefetch_batches)
        elif self.capacity <= 0:
            raise ValueError("queue.capacity must be positive")


@dataclass(frozen=True, slots=True)
class AppConfig:
    media: MediaConfig = MediaConfig()
    image_sequence: ImageSequenceConfig = ImageSequenceConfig()
    queue: QueueConfig = QueueConfig()


def load_config(path: str | Path) -> AppConfig:
    """Load and validate the YAML configuration file."""
    try:
        import yaml
    except ImportError as exc:  # keep core tests independent of YAML
        raise RuntimeError("PyYAML is required to load a YAML configuration file") from exc
    with Path(path).open("r", encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream) or {}

    image_sequence = dict(raw.get("image_sequence", {}))
    if "extensions" in image_sequence:
        image_sequence["extensions"] = tuple(image_sequence["extensions"])

    return AppConfig(
        media=MediaConfig(**raw.get("media", {})),
        image_sequence=ImageSequenceConfig(**image_sequence),
        queue=QueueConfig(**raw.get("queue", {})),
    )
