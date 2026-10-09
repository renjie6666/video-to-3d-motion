"""Decode a video and publish FrameTask records to a bounded FIFO queue."""

from __future__ import annotations

import hashlib
import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from .config import AppConfig
from .exceptions import MediaPipelineError, ProducerCancelled
from .frame_sampler import FrameSampler
from .models import (
    DecodeFailure,
    DecodedFrame,
    EndOfVideo,
    FrameTask,
    ProducerMetrics,
    ProducerResult,
    QueueItem,
    VideoInfo,
)
from .video_decoder import PyAVVideoDecoder


class Decoder(Protocol):
    def probe(self, video_path: str | Path) -> VideoInfo: ...

    def iter_frames(self, video_info: VideoInfo) -> Iterator[DecodedFrame]: ...


def make_video_id(video_path: str | Path) -> str:
    path = Path(video_path).resolve()
    digest = hashlib.sha1(str(path).lower().encode("utf-8")).hexdigest()[:8]
    return f"{path.stem}-{digest}"


class FrameTaskProducer:
    def __init__(self, config: AppConfig, decoder: Decoder | None = None) -> None:
        self.config = config
        self.decoder = decoder or PyAVVideoDecoder(config.media)

    def run(
        self,
        video_path: str | Path,
        output_queue: queue.Queue[QueueItem],
        stop_event: threading.Event,
        video_id: str | None = None,
    ) -> ProducerResult:
        resolved_video_id = video_id or make_video_id(video_path)
        metrics = ProducerMetrics()

        try:
            info = self.decoder.probe(video_path)
            sparse_images = (
                info.path.is_dir()
                and not self.config.image_sequence.require_contiguous
            )
            minimum_fps = self.config.media.target_fps - self.config.media.fps_tolerance
            if (
                not sparse_images
                and self.config.media.fail_on_low_fps
                and info.source_fps < minimum_fps
            ):
                raise MediaPipelineError(
                    "fps_too_low",
                    f"source FPS {info.source_fps:.3f} is below target "
                    f"{self.config.media.target_fps}",
                )

            sampler = None if sparse_images else FrameSampler(
                target_fps=self.config.media.target_fps,
                tolerance_ms=self.config.media.timestamp_tolerance_ms,
            )
            for decoded in self.decoder.iter_frames(info):
                self._raise_if_cancelled(stop_event)
                metrics.decoded_frames += 1
                if decoded.timestamp_source == "fps_fallback":
                    metrics.pts_fallback_frames += 1
                if sampler is None:
                    # Evaluate each supplied image once, keeping its actual
                    # source time instead of filling a continuous FPS timeline.
                    task = self._to_task(
                        decoded, metrics.emitted_frames + 1,
                        decoded.source_timestamp, resolved_video_id,
                    )
                    self._publish(task, output_queue, stop_event, metrics)
                    metrics.emitted_frames += 1
                    continue
                for sampled in sampler.push(decoded):
                    task = self._to_task(
                        sampled.decoded,
                        sampled.frame_id,
                        sampled.timestamp,
                        resolved_video_id,
                    )
                    self._publish(task, output_queue, stop_event, metrics)
                    metrics.emitted_frames += 1

            if sampler is not None:
                for sampled in sampler.flush():
                    task = self._to_task(
                        sampled.decoded,
                        sampled.frame_id,
                        sampled.timestamp,
                        resolved_video_id,
                    )
                    self._publish(task, output_queue, stop_event, metrics)
                    metrics.emitted_frames += 1

            self._publish(
                EndOfVideo(resolved_video_id, metrics.emitted_frames),
                output_queue,
                stop_event,
                metrics,
            )
            return ProducerResult(resolved_video_id, "completed", metrics)
        except ProducerCancelled as exc:
            return ProducerResult(
                resolved_video_id,
                "cancelled",
                metrics,
                error_code=exc.code,
                message=str(exc),
            )
        except MediaPipelineError as exc:
            self._publish_failure_if_possible(
                DecodeFailure(resolved_video_id, exc.code, str(exc)),
                output_queue,
                stop_event,
                metrics,
            )
            return ProducerResult(
                resolved_video_id,
                "failed",
                metrics,
                error_code=exc.code,
                message=str(exc),
            )
        except Exception as exc:
            self._publish_failure_if_possible(
                DecodeFailure(resolved_video_id, "decode_failed", str(exc)),
                output_queue,
                stop_event,
                metrics,
            )
            return ProducerResult(
                resolved_video_id,
                "failed",
                metrics,
                error_code="decode_failed",
                message=str(exc),
            )

    def _publish(
        self,
        item: QueueItem,
        output_queue: queue.Queue[QueueItem],
        stop_event: threading.Event,
        metrics: ProducerMetrics,
    ) -> None:
        while not stop_event.is_set():
            started = time.perf_counter()
            try:
                output_queue.put(item, timeout=self.config.queue.put_timeout_seconds)
                metrics.producer_block_seconds += time.perf_counter() - started
                metrics.max_queue_size = max(metrics.max_queue_size, output_queue.qsize())
                return
            except queue.Full:
                metrics.producer_block_seconds += time.perf_counter() - started
                metrics.queue_full_count += 1
        raise ProducerCancelled()

    def _publish_failure_if_possible(
        self,
        failure: DecodeFailure,
        output_queue: queue.Queue[QueueItem],
        stop_event: threading.Event,
        metrics: ProducerMetrics,
    ) -> None:
        if stop_event.is_set():
            return
        try:
            self._publish(failure, output_queue, stop_event, metrics)
        except ProducerCancelled:
            pass

    @staticmethod
    def _raise_if_cancelled(stop_event: threading.Event) -> None:
        if stop_event.is_set():
            raise ProducerCancelled()

    def _to_task(
        self,
        decoded: DecodedFrame,
        frame_id: int,
        timestamp: float,
        video_id: str,
    ) -> FrameTask:
        height, width = decoded.image.shape[:2]
        return FrameTask(
            image=decoded.image,
            video_id=video_id,
            frame_id=frame_id,
            source_frame_id=decoded.source_frame_id,
            timestamp=timestamp,
            source_timestamp=decoded.source_timestamp,
            timestamp_source=decoded.timestamp_source,
            image_size=(width, height),
            fps=self.config.media.target_fps,
        )
