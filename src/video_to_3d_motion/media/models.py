"""Data contracts shared by decoders and the downstream consumer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

UInt8Image: TypeAlias = NDArray[np.uint8]


@dataclass(slots=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    source_fps: float
    duration_seconds: float | None


@dataclass(slots=True)
class DecodedFrame:
    image: UInt8Image
    source_frame_id: int
    source_timestamp: float
    pts: int | None
    timestamp_source: Literal["pts", "fps_fallback", "filename"]


@dataclass(slots=True)
class SampledFrame:
    decoded: DecodedFrame
    frame_id: int
    timestamp: float


@dataclass(slots=True)
class FrameTask:
    image: UInt8Image
    video_id: str
    frame_id: int
    source_frame_id: int
    timestamp: float
    source_timestamp: float
    timestamp_source: Literal["pts", "fps_fallback", "filename"]
    image_size: tuple[int, int]
    fps: int = 50


@dataclass(frozen=True, slots=True)
class EndOfVideo:
    video_id: str
    frame_count: int


@dataclass(frozen=True, slots=True)
class DecodeFailure:
    video_id: str
    error_code: str
    message: str


QueueItem: TypeAlias = FrameTask | EndOfVideo | DecodeFailure


@dataclass(slots=True)
class ProducerMetrics:
    decoded_frames: int = 0
    emitted_frames: int = 0
    pts_fallback_frames: int = 0
    queue_full_count: int = 0
    producer_block_seconds: float = 0.0
    max_queue_size: int = 0


@dataclass(slots=True)
class ProducerResult:
    video_id: str
    status: Literal["completed", "cancelled", "failed"]
    metrics: ProducerMetrics
    error_code: str | None = None
    message: str | None = None
