"""Queue worker with bounded history, backpressure and observable failure."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterable

from video_to_3d_motion.pose2d.models import EndOfPose2D, Pose2DFailure, Pose2DFrame

from .config import PoseAdapterConfig
from .exceptions import PoseAdapterCancelled, PoseAdapterError
from .models import EndOfPoseAdapter, PoseAdapterFailure, PoseAdapterOutputItem, PoseAdapterResult
from .window_buffer import WindowBuffer


class PoseAdapterWorker:
    def __init__(self, config: PoseAdapterConfig, *,
                 record_frame: Callable[[Pose2DFrame], None] | None = None) -> None:
        self.config = config
        self.record_frame = record_frame

    def run(self, input_queue: queue.Queue, output_queue: queue.Queue,
            stop_event: threading.Event) -> PoseAdapterResult:
        def incoming():
            while not stop_event.is_set():
                try:
                    item = input_queue.get(timeout=self.config.queue.get_timeout_seconds)
                except queue.Empty:
                    continue
                try:
                    yield item
                finally:
                    input_queue.task_done()
        return self.run_stream(incoming(), lambda item: self._publish(item, output_queue, stop_event), stop_event)

    def run_stream(self, items: Iterable, emit: Callable[[PoseAdapterOutputItem], None],
                   stop_event: threading.Event | None = None) -> PoseAdapterResult:
        stop = stop_event if stop_event is not None else threading.Event()
        buffer = WindowBuffer(self.config)
        current_frame_id = None
        current_video_id = None
        iterator = iter(items)
        try:
            for item in iterator:
                current_video_id = getattr(item, "video_id", current_video_id)
                if stop.is_set():
                    raise PoseAdapterCancelled()
                if isinstance(item, Pose2DFrame):
                    current_frame_id = item.frame_id
                    buffer.push(item)
                    if self.record_frame:
                        try:
                            self.record_frame(item)
                        except Exception as exc:
                            raise PoseAdapterError("archive_write_failed", str(exc)) from exc
                elif isinstance(item, Pose2DFailure):
                    raise PoseAdapterError("upstream_failed", f"{item.error_code}: {item.message}")
                elif isinstance(item, EndOfPose2D):
                    current_frame_id = None
                    window = buffer.finish(item)
                    started = time.perf_counter()
                    emit(window)
                    buffer.metrics.publish_block_seconds += time.perf_counter() - started
                    buffer.metrics.windows_emitted += 1
                    emit(EndOfPoseAdapter(item.video_id, buffer.metrics.frames_received, 1, 1,
                                          int(window.frame_ids[-1])))
                    return PoseAdapterResult(item.video_id, "completed", buffer.metrics)
                else:
                    raise PoseAdapterError("invalid_pose2d_frame", "unknown input message")
            if stop.is_set():
                raise PoseAdapterCancelled()
            raise PoseAdapterError("metadata_mismatch", "input ended without EndOfPose2D")
        except Exception as exc:
            error = exc if isinstance(exc, PoseAdapterError) else PoseAdapterError("invalid_pose2d_frame", str(exc))
            stop.set()
            video_id = buffer.video_id or current_video_id
            failure = PoseAdapterFailure(video_id, current_frame_id, error.code, str(error))
            # The return value remains authoritative if an output queue is full.
            try:
                emit(failure)
            except Exception:
                pass
            return PoseAdapterResult(video_id, "cancelled" if error.code == "cancelled" else "failed",
                                     buffer.metrics, error.code, str(error))
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def _publish(self, item: PoseAdapterOutputItem, output: queue.Queue,
                 stop: threading.Event) -> None:
        if isinstance(item, PoseAdapterFailure):
            output.put_nowait(item)
            return
        while not stop.is_set():
            try:
                output.put(item, timeout=self.config.queue.put_timeout_seconds)
                return
            except queue.Full:
                continue
        raise PoseAdapterCancelled()
