"""Development CLI connecting S1-01 input to the S1-02 worker."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from dataclasses import asdict, replace
from pathlib import Path

from video_to_3d_motion.media.config import load_config
from video_to_3d_motion.media.image_sequence_decoder import ImageSequenceDecoder
from video_to_3d_motion.media.models import QueueItem
from video_to_3d_motion.media.producer import FrameTaskProducer

from .config import load_pose2d_config
from .comparison import run_comparison
from .h36m_annotations import load_sequence_annotations
from .factory import create_gpu_worker
from .models import EndOfPose2D, Pose2DFailure, Pose2DFrame, Pose2DOutputItem
from .worker import Pose2DWorker


def _parser(description: str, source_help: str, *, image_sequence: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("source", type=Path, help=source_help)
    parser.add_argument("--media-config", type=Path, required=True)
    parser.add_argument("--pose2d-config", type=Path, required=True)
    parser.add_argument("--video-id")
    parser.add_argument("--frames-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--person-selection", choices=("strict", "highest_score"),
                        help="override YAML single-person policy; highest_score is for known single-person sequences")
    if image_sequence:
        parser.add_argument("--max-frames", type=int, help="process only the first N images for a smoke test")
        parser.add_argument("--annotations", type=Path, help="H36M train/validation pickle for comparison side output")
        parser.add_argument("--comparison-output", type=Path, help="comparison directory (default: summary directory/h36m_comparison)")
        parser.add_argument("--comparison-every-n-frames", type=int, default=1,
                            help="write one comparison overlay per N frames; all frames still appear in CSV")
    return parser


def _record(pose: Pose2DFrame, source: Path | None = None) -> dict[str, object]:
    return {
        "video_id": pose.video_id,
        "frame_id": pose.frame_id,
        "source_frame_id": pose.source_frame_id,
        "timestamp": pose.timestamp,
        "source_timestamp": pose.source_timestamp,
        "timestamp_source": pose.timestamp_source,
        "image_size": list(pose.image_size),
        "coordinate_space": "original_image_pixels",
        "source_path": str(source.resolve()) if source else None,
        "fps": pose.fps,
        "keypoints": pose.keypoints.tolist(),
        "keypoint_scores_raw": pose.keypoint_scores_raw.tolist(),
        "valid_mask": pose.valid_mask.tolist(),
        "joint_schema": pose.joint_schema,
        "score_source": pose.score_source,
        "bbox": pose.bbox.tolist(),
        "bbox_score": pose.bbox_score,
        "bbox_source": pose.bbox_source,
        "model": pose.model,
        "model_config": pose.model_config,
        "checkpoint_sha256": pose.checkpoint_sha256,
        "visualization_path": (
            str(pose.visualization_path) if pose.visualization_path else None
        ),
    }


def _run(args, *, image_sequence: bool) -> int:
    media_config = load_config(args.media_config)
    pose_config = load_pose2d_config(args.pose2d_config)
    if args.person_selection:
        pose_config = replace(pose_config, detection=replace(
            pose_config.detection, strict_single_person=args.person_selection == "strict"
        ))
    selection = "strict" if pose_config.detection.strict_single_person else "highest_score"
    if image_sequence:
        if args.comparison_output and not args.annotations:
            raise ValueError("--comparison-output requires --annotations")
        if args.comparison_every_n_frames <= 0:
            raise ValueError("--comparison-every-n-frames must be positive")
        if args.annotations and not load_sequence_annotations(args.annotations, args.source):
            raise ValueError("no matching annotations; check the sequence and train/validation split")
    decoder = (
        ImageSequenceDecoder(media_config.image_sequence, max_frames=args.max_frames)
        if image_sequence
        else None
    )
    sequence_id = args.video_id or (args.source.name if image_sequence else None)
    frame_queue: queue.Queue[QueueItem] = queue.Queue(
        maxsize=int(media_config.queue.capacity or 0)
    )
    pose_queue: queue.Queue[Pose2DOutputItem] = queue.Queue(
        maxsize=pose_config.batch.output_queue_capacity
    )
    stop_event = threading.Event()
    producer = FrameTaskProducer(media_config, decoder=decoder)
    pose_worker = create_gpu_worker(pose_config)
    producer_results = []
    pose_results = []
    producer_thread = threading.Thread(
        target=lambda: producer_results.append(
            producer.run(args.source, frame_queue, stop_event, sequence_id)
        ),
        name="frame-task-producer",
    )
    pose_thread = threading.Thread(
        target=lambda: pose_results.append(
            pose_worker.run(frame_queue, pose_queue, stop_event)
        ),
        name="pose2d-worker",
    )

    args.frames_output.parent.mkdir(parents=True, exist_ok=True)
    terminal: EndOfPose2D | Pose2DFailure | None = None
    records_written = 0
    pose_thread.start()
    producer_thread.start()
    try:
        with args.frames_output.open("w", encoding="utf-8", newline="\n") as stream:
            while terminal is None:
                item = pose_queue.get()
                try:
                    if isinstance(item, Pose2DFrame):
                        record = _record(item, args.source)
                        record["bbox_selection"] = selection
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        records_written += 1
                        if records_written == 1 or records_written % 50 == 0:
                            print(f"Pose2D: {records_written} frames", file=sys.stderr)
                    else:
                        terminal = item
                finally:
                    pose_queue.task_done()
    except KeyboardInterrupt:
        stop_event.set()
    except BaseException:
        stop_event.set()
        raise
    finally:
        producer_thread.join()
        pose_thread.join()

    producer_result = producer_results[0] if producer_results else None
    pose_result = pose_results[0] if pose_results else None
    summary = {
        "status": pose_result.status if pose_result else "failed",
        "video_id": pose_result.video_id if pose_result else sequence_id,
        "error_code": pose_result.error_code if pose_result else "worker_missing",
        "message": pose_result.message if pose_result else "pose worker did not return",
        "producer_status": producer_result.status if producer_result else None,
        "producer_metrics": asdict(producer_result.metrics) if producer_result else None,
        "pose2d_metrics": asdict(pose_result.metrics) if pose_result else None,
        "model": pose_config.model.name,
        "detection": {
            "bbox_source": pose_config.detection.bbox_source,
            "person_selection": selection,
            "score_threshold": pose_config.detection.score_threshold,
            "detector_config": str(pose_config.detection.detector_config),
            "detector_checkpoint": str(pose_config.detection.detector_checkpoint),
        },
        "records_written": records_written,
        "frames_output": str(args.frames_output),
        "visualization_root": (
            str(pose_config.visualization.output_root)
            if pose_config.visualization.enabled
            else None
        ),
        "terminal_message": type(terminal).__name__ if terminal else None,
    }
    if image_sequence and args.annotations and summary["status"] == "completed":
        try:
            print("Pose2D: writing H36M comparison side output", file=sys.stderr)
            summary["comparison"] = run_comparison(
                args.source, args.frames_output, args.annotations,
                args.comparison_output or args.summary_output.parent / "h36m_comparison",
                every_n_frames=args.comparison_every_n_frames,
            )
        except Exception as exc:
            summary["inference_status"] = "completed"
            summary.update(status="failed", error_code="comparison_failed", message=str(exc))
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] == "completed":
        print(
            f"SUCCESS: wrote {records_written} Pose2D frames; "
            "backend=mmpose; "
            f"visualizations={summary['visualization_root']}",
            file=sys.stderr,
        )
        return 0
    print(f"FAILED: {summary['error_code']}: {summary['message']}", file=sys.stderr)
    return 1


def video_main() -> int:
    args = _parser(
        "Run the S1-01 to S1-02 development chain on a video",
        "path to an MP4/AVI video",
    ).parse_args()
    return _run(args, image_sequence=False)


def images_main() -> int:
    args = _parser(
        "Run the S1-01 to S1-02 development chain on an image sequence",
        "path to one ordered JPEG/PNG sequence directory",
        image_sequence=True,
    ).parse_args()
    return _run(args, image_sequence=True)


if __name__ == "__main__":
    raise SystemExit(images_main())
