"""PyAV-backed video probing and presentation-order frame decoding."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .config import MediaConfig
from .exceptions import MediaPipelineError
from .models import DecodedFrame, VideoInfo


def _import_av():
    try:
        import av
    except ImportError as exc:  # pragma: no cover - depends on local runtime
        raise MediaPipelineError(
            "dependency_missing",
            "PyAV is not installed; install the project dependencies first",
        ) from exc
    return av


def _stream_rate(stream: object) -> float:
    for name in ("average_rate", "base_rate", "guessed_rate"):
        value = getattr(stream, name, None)
        if value is not None:
            rate = float(value)
            if rate > 0:
                return rate
    raise MediaPipelineError("fps_unavailable", "video stream FPS is unavailable")


class PyAVVideoDecoder:
    def __init__(self, config: MediaConfig) -> None:
        self.config = config

    def probe(self, video_path: str | Path) -> VideoInfo:
        path = Path(video_path)
        if not path.is_file():
            raise MediaPipelineError("video_not_found", f"video does not exist: {path}")

        av = _import_av()
        try:
            with av.open(str(path)) as container:
                stream = next(iter(container.streams.video), None)
                if stream is None:
                    raise MediaPipelineError("video_stream_missing", "no video stream found")
                source_fps = _stream_rate(stream)
                duration = (
                    float(stream.duration * stream.time_base)
                    if stream.duration is not None and stream.time_base is not None
                    else None
                )
                return VideoInfo(
                    path=path,
                    width=int(stream.codec_context.width),
                    height=int(stream.codec_context.height),
                    source_fps=source_fps,
                    duration_seconds=duration,
                )
        except MediaPipelineError:
            raise
        except Exception as exc:
            raise MediaPipelineError("decode_failed", f"cannot open video: {exc}") from exc

    def iter_frames(self, video_info: VideoInfo) -> Iterator[DecodedFrame]:
        av = _import_av()
        first_pts_offset: float | None = None
        previous_timestamp: float | None = None

        try:
            with av.open(str(video_info.path)) as container:
                stream = next(iter(container.streams.video), None)
                if stream is None:
                    raise MediaPipelineError("video_stream_missing", "no video stream found")

                for source_frame_id, frame in enumerate(container.decode(stream), start=1):
                    timestamp_source = "pts"
                    if frame.pts is not None and stream.time_base is not None:
                        raw_pts_seconds = float(frame.pts * stream.time_base)
                        if first_pts_offset is None:
                            first_pts_offset = raw_pts_seconds - (
                                (source_frame_id - 1) / video_info.source_fps
                            )
                        source_timestamp = raw_pts_seconds - first_pts_offset
                    elif self.config.fallback_when_pts_missing:
                        source_timestamp = (source_frame_id - 1) / video_info.source_fps
                        timestamp_source = "fps_fallback"
                    else:
                        raise MediaPipelineError(
                            "pts_missing", f"PTS missing at source frame {source_frame_id}"
                        )

                    if previous_timestamp is not None and source_timestamp <= previous_timestamp:
                        raise MediaPipelineError(
                            "non_monotonic_pts",
                            f"timestamp is not increasing at source frame {source_frame_id}",
                        )
                    previous_timestamp = source_timestamp

                    image = np.ascontiguousarray(
                        frame.to_ndarray(format=self.config.pixel_format), dtype=np.uint8
                    )
                    yield DecodedFrame(
                        image=image,
                        source_frame_id=source_frame_id,
                        source_timestamp=source_timestamp,
                        pts=frame.pts,
                        timestamp_source=timestamp_source,
                    )
        except MediaPipelineError:
            raise
        except Exception as exc:
            raise MediaPipelineError("decode_failed", f"video decoding failed: {exc}") from exc
