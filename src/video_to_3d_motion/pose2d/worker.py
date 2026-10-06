"""Bounded-queue S1-02 worker with deterministic batching and metadata flow."""

from __future__ import annotations

import hashlib
import queue
import threading
from pathlib import Path

import numpy as np

from video_to_3d_motion.media.models import (
    DecodeFailure,
    EndOfVideo,
    FrameTask,
    QueueItem,
)

from .adapters import normalize_bbox, validate_frame_task, validate_prediction
from .backends.protocols import PersonDetector, PoseEstimator
from .config import Pose2DConfig
from .exceptions import Pose2DCancelled, Pose2DError
from .models import (
    EndOfPose2D,
    ExternalBBox,
    PersonBBox,
    Pose2DFailure,
    Pose2DFrame,
    Pose2DMetrics,
    Pose2DOutputItem,
    Pose2DResult,
)
from .visualization import Pose2DVisualizer


def _sha256(path: Path | None) -> str:
    if path is None:
        return "not_provided"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Pose2DWorker:
    """Consume one video's FrameTask stream and publish Pose2DFrame records."""

    def __init__(
        self,
        config: Pose2DConfig,
        *,
        detector: PersonDetector,
        estimator: PoseEstimator,
        external_bboxes: dict[tuple[str, int], ExternalBBox] | None = None,
        visualizer: Pose2DVisualizer | None = None,
    ) -> None:
        self.config = config
        self.external_bboxes = external_bboxes or {}
        self.detector = detector
        self.estimator = estimator

        if self.estimator.joint_schema != config.model.joint_schema:
            raise Pose2DError(
                "joint_schema_mismatch",
                "estimator joint schema does not match the selected model config",
            )
        self.visualizer = visualizer or (
            Pose2DVisualizer(config.visualization)
            if config.visualization.enabled
            else None
        )
        self._checkpoint_sha256 = _sha256(config.model.checkpoint_path)

    def run(
        self,
        input_queue: queue.Queue[QueueItem],
        output_queue: queue.Queue[Pose2DOutputItem],
        stop_event: threading.Event,
    ) -> Pose2DResult:
        metrics = Pose2DMetrics()
        pending: list[FrameTask] = []
        video_id: str | None = None
        last_frame_id = 0
        current_frame_id: int | None = None

        try:
            while True:
                if stop_event.is_set():
                    raise Pose2DCancelled()
                try:
                    item = input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if isinstance(item, FrameTask):
                    current_frame_id = item.frame_id
                    try:
                        validate_frame_task(item)
                        if video_id is None:
                            video_id = item.video_id
                        if item.video_id != video_id:
                            raise Pose2DError(
                                "metadata_mismatch",
                                "one worker cannot mix multiple video_id values",
                            )
                        if item.frame_id != last_frame_id + 1:
                            raise Pose2DError(
                                "frame_order_invalid",
                                f"expected frame {last_frame_id + 1}, got {item.frame_id}",
                            )
                        last_frame_id = item.frame_id
                        metrics.frames_received += 1
                        pending.append(item)
                    except Exception:
                        input_queue.task_done()
                        raise
                    if len(pending) >= self.config.batch.max_size:
                        self._flush_batch(
                            pending, input_queue, output_queue, stop_event, metrics
                        )
                    continue

                current_frame_id = None
                if isinstance(item, DecodeFailure):
                    self._discard_pending(pending, input_queue)
                    input_queue.task_done()
                    failure = Pose2DFailure(
                        video_id=item.video_id,
                        frame_id=None,
                        error_code=item.error_code,
                        message=item.message,
                        recoverable=False,
                    )
                    self._publish(failure, output_queue, stop_event)
                    return Pose2DResult(
                        item.video_id,
                        "failed",
                        metrics,
                        error_code=item.error_code,
                        message=item.message,
                    )

                if not isinstance(item, EndOfVideo):
                    input_queue.task_done()
                    raise Pose2DError("invalid_frame_task", "unknown queue item")

                try:
                    if video_id is None:
                        video_id = item.video_id
                    if item.video_id != video_id:
                        raise Pose2DError(
                            "metadata_mismatch", "EndOfVideo video_id does not match"
                        )
                    self._flush_batch(
                        pending, input_queue, output_queue, stop_event, metrics
                    )
                    if item.frame_count != metrics.frames_received:
                        raise Pose2DError(
                            "metadata_mismatch",
                            "EndOfVideo frame_count does not match received frames",
                        )
                    terminal = EndOfPose2D(
                        video_id=video_id,
                        frames_received=metrics.frames_received,
                        frames_succeeded=metrics.frames_succeeded,
                        frames_failed=metrics.frames_failed,
                        batches_processed=metrics.batches_processed,
                    )
                    self._publish(terminal, output_queue, stop_event)
                finally:
                    input_queue.task_done()
                return Pose2DResult(
                    video_id, "completed", metrics
                )
        except Pose2DCancelled as exc:
            self._discard_pending(pending, input_queue)
            return Pose2DResult(
                video_id,
                "cancelled",
                metrics,
                error_code=exc.code,
                message=str(exc),
            )
        except Pose2DError as exc:
            self._discard_pending(pending, input_queue)
            stop_event.set()
            metrics.frames_failed = max(
                metrics.frames_failed,
                metrics.frames_received - metrics.frames_succeeded,
                1,
            )
            failure = Pose2DFailure(
                video_id=video_id,
                frame_id=current_frame_id,
                error_code=exc.code,
                message=str(exc),
                recoverable=False,
            )
            self._publish_without_stop_check(failure, output_queue)
            return Pose2DResult(
                video_id,
                "failed",
                metrics,
                error_code=exc.code,
                message=str(exc),
            )
        except Exception as exc:
            self._discard_pending(pending, input_queue)
            stop_event.set()
            metrics.frames_failed = max(
                metrics.frames_failed,
                metrics.frames_received - metrics.frames_succeeded,
                1,
            )
            failure = Pose2DFailure(
                video_id=video_id,
                frame_id=current_frame_id,
                error_code="pose_inference_failed",
                message=str(exc),
                recoverable=False,
            )
            self._publish_without_stop_check(failure, output_queue)
            return Pose2DResult(
                video_id,
                "failed",
                metrics,
                error_code="pose_inference_failed",
                message=str(exc),
            )

    def _flush_batch(
        self,
        pending: list[FrameTask],
        input_queue: queue.Queue[QueueItem],
        output_queue: queue.Queue[Pose2DOutputItem],
        stop_event: threading.Event,
        metrics: Pose2DMetrics,
    ) -> None:
        if not pending:
            return
        batch = list(pending)
        pending.clear()
        try:
            boxes = self._resolve_bboxes(batch)
            predictions = self.estimator.infer_batch(
                [task.image for task in batch], [box.bbox for box in boxes]
            )
            if len(predictions) != len(batch):
                raise Pose2DError(
                    "metadata_mismatch", "pose result count does not match batch"
                )
            for task, box, raw_prediction in zip(
                batch, boxes, predictions, strict=True
            ):
                prediction = validate_prediction(raw_prediction)
                pose = self._to_pose_frame(task, box, prediction)
                if self.visualizer is not None:
                    try:
                        if self.visualizer.write(task, pose) is not None:
                            metrics.visualization_frames_written += 1
                    except Pose2DError:
                        if self.config.visualization.strict:
                            raise
                self._publish(pose, output_queue, stop_event)
                metrics.frames_succeeded += 1
            metrics.batches_processed += 1
        finally:
            for _ in batch:
                input_queue.task_done()

    def _resolve_bboxes(self, tasks: list[FrameTask]) -> list[PersonBBox]:
        output: list[PersonBBox | None] = [None] * len(tasks)
        detection_indices: list[int] = []
        for index, task in enumerate(tasks):
            external = self.external_bboxes.get((task.video_id, task.frame_id))
            if (
                self.config.detection.bbox_source != "detector_only"
                and external is not None
            ):
                output[index] = normalize_bbox(
                    external.bbox,
                    width=task.image_size[0],
                    height=task.image_size[1],
                    min_area=self.config.detection.min_bbox_area,
                    score=external.score,
                    source="external",
                )
            elif self.config.detection.bbox_source == "external_only":
                raise Pose2DError(
                    "external_bbox_missing",
                    f"missing bbox for {task.video_id}/{task.frame_id}",
                )
            else:
                detection_indices.append(index)

        if detection_indices:
            candidates_by_image = self.detector.detect_batch(
                [tasks[index].image for index in detection_indices]
            )
            if len(candidates_by_image) != len(detection_indices):
                raise Pose2DError(
                    "metadata_mismatch", "detector result count does not match input"
                )
            for index, candidates in zip(
                detection_indices, candidates_by_image, strict=True
            ):
                task = tasks[index]
                filtered = [
                    candidate
                    for candidate in candidates
                    if candidate.score is None
                    or candidate.score >= self.config.detection.score_threshold
                ]
                if not filtered:
                    raise Pose2DError("person_not_found", "no valid person bbox")
                if self.config.detection.strict_single_person and len(filtered) > 1:
                    raise Pose2DError(
                        "multiple_persons", "multiple person bboxes were detected"
                    )
                selected = max(filtered, key=lambda box: box.score or 0.0)
                output[index] = normalize_bbox(
                    selected.bbox,
                    width=task.image_size[0],
                    height=task.image_size[1],
                    min_area=self.config.detection.min_bbox_area,
                    score=selected.score,
                    source=selected.source,
                )
        return [box for box in output if box is not None]

    def _to_pose_frame(self, task, box, prediction) -> Pose2DFrame:
        return Pose2DFrame(
            keypoints=prediction.keypoints,
            keypoint_scores_raw=prediction.keypoint_scores_raw,
            valid_mask=(
                prediction.keypoint_scores_raw
                >= self.config.runtime.keypoint_score_threshold
            ),
            joint_schema=self.estimator.joint_schema,
            score_source=self.estimator.score_source,
            bbox=box.bbox,
            bbox_score=box.score,
            bbox_source=box.source,
            video_id=task.video_id,
            frame_id=task.frame_id,
            source_frame_id=task.source_frame_id,
            timestamp=task.timestamp,
            source_timestamp=task.source_timestamp,
            timestamp_source=task.timestamp_source,
            image_size=task.image_size,
            fps=task.fps,
            model=self.config.model.name,
            model_config=str(self.config.model.config_path),
            checkpoint_sha256=self._checkpoint_sha256,
        )

    def _publish(
        self,
        item: Pose2DOutputItem,
        output_queue: queue.Queue[Pose2DOutputItem],
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                output_queue.put(
                    item, timeout=self.config.batch.put_timeout_seconds
                )
                return
            except queue.Full:
                continue
        raise Pose2DCancelled()

    @staticmethod
    def _publish_without_stop_check(
        item: Pose2DOutputItem,
        output_queue: queue.Queue[Pose2DOutputItem],
    ) -> None:
        try:
            output_queue.put_nowait(item)
        except queue.Full:
            pass

    @staticmethod
    def _discard_pending(
        pending: list[FrameTask], input_queue: queue.Queue[QueueItem]
    ) -> None:
        while pending:
            pending.pop()
            input_queue.task_done()
