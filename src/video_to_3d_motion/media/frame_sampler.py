"""Streaming nearest-timestamp sampling onto a fixed target FPS timeline."""

from __future__ import annotations

from .exceptions import MediaPipelineError
from .models import DecodedFrame, SampledFrame


class FrameSampler:
    """Map decoded frames to a fixed timeline without duplicating source frames."""

    def __init__(self, target_fps: int, tolerance_ms: float) -> None:
        self.target_fps = target_fps
        self.tolerance_seconds = tolerance_ms / 1000.0
        self._previous: DecodedFrame | None = None
        self._next_frame_id = 1
        self._last_source_frame_id = -1

    def push(self, current: DecodedFrame) -> list[SampledFrame]:
        if self._previous is None:
            self._previous = current
            return []

        if current.source_timestamp <= self._previous.source_timestamp:
            raise MediaPipelineError(
                "non_monotonic_pts", "decoded timestamps must be strictly increasing"
            )

        emitted: list[SampledFrame] = []
        while self._target_timestamp <= current.source_timestamp:
            target = self._target_timestamp
            previous_distance = abs(self._previous.source_timestamp - target)
            current_distance = abs(current.source_timestamp - target)
            candidate = self._previous if previous_distance <= current_distance else current

            if candidate.source_frame_id <= self._last_source_frame_id:
                raise MediaPipelineError(
                    "timestamp_gap",
                    "a source frame would need to be duplicated to maintain target FPS",
                )
            distance = abs(candidate.source_timestamp - target)
            if distance > self.tolerance_seconds:
                raise MediaPipelineError(
                    "timestamp_gap",
                    f"nearest source frame is {distance * 1000:.3f} ms from target",
                )

            emitted.append(
                SampledFrame(
                    decoded=candidate,
                    frame_id=self._next_frame_id,
                    timestamp=target,
                )
            )
            self._last_source_frame_id = candidate.source_frame_id
            self._next_frame_id += 1

        self._previous = current
        return emitted

    def flush(self) -> list[SampledFrame]:
        if self._previous is None:
            return []
        target = self._target_timestamp
        distance = abs(self._previous.source_timestamp - target)
        if (
            target <= self._previous.source_timestamp
            and distance <= self.tolerance_seconds
            and self._previous.source_frame_id > self._last_source_frame_id
        ):
            sampled = SampledFrame(
                decoded=self._previous,
                frame_id=self._next_frame_id,
                timestamp=target,
            )
            self._last_source_frame_id = self._previous.source_frame_id
            self._next_frame_id += 1
            return [sampled]
        return []

    @property
    def _target_timestamp(self) -> float:
        return (self._next_frame_id - 1) / self.target_fps
