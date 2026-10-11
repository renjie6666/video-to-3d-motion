"""Replay existing S1-02 JSONL with verified JSON or archived CSV summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from video_to_3d_motion.pose2d.models import EndOfPose2D, Pose2DFrame

from .exceptions import PoseAdapterError


def _count(value) -> int:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    raise ValueError(f"invalid frame count {value!r}")


def load_success_summary(path: str | Path, input_path: Path) -> dict:
    try:
        path = Path(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            if path.suffix.lower() == ".csv":
                rows = list(csv.DictReader(stream))
                if len(rows) != 1:
                    raise ValueError("use one experiment's detailed CSV summary")
                summary = rows[0]
            else:
                summary = json.load(stream)
        if summary.get("status") != "completed" or summary.get("producer_status") != "completed":
            raise ValueError("the upstream run did not complete successfully")
        if summary.get("terminal_message") != "EndOfPose2D":
            raise ValueError("upstream has no normal terminal message")
        metrics = summary.get("pose2d_metrics") or {
            name: summary.get("pose2d_metrics." + name)
            for name in ("frames_received", "frames_succeeded", "frames_failed", "batches_processed")
        }
        count = _count(summary.get("records_written"))
        if (_count(metrics.get("frames_received")) != count
                or _count(metrics.get("frames_succeeded")) != count
                or _count(metrics.get("frames_failed")) != 0):
            raise ValueError("upstream counts do not match")
        filename = str(summary["frames_output"]).replace("\\", "/").split("/")[-1]
        if filename != input_path.name:
            raise ValueError("summary belongs to a different input file")
        if not isinstance(summary["video_id"], str) or not summary["video_id"]:
            raise ValueError("missing video identity")
        return dict(summary, expected_count=count, batches=_count(metrics["batches_processed"]))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise PoseAdapterError("input_summary_invalid", str(exc)) from exc


def frame_from_record(record: dict) -> Pose2DFrame:
    if record.get("coordinate_space") != "original_image_pixels":
        raise PoseAdapterError("coordinate_space_mismatch", "expected original_image_pixels")
    try:
        raw_mask = np.asarray(record["valid_mask"])
        if raw_mask.dtype != np.bool_:
            raise ValueError("valid_mask values must be boolean")
        # Preserve scalar types so validation can reject strings and booleans.
        return Pose2DFrame(
            keypoints=np.asarray(record["keypoints"], dtype=np.float32),
            keypoint_scores_raw=np.asarray(record["keypoint_scores_raw"], dtype=np.float32),
            valid_mask=raw_mask, joint_schema=record["joint_schema"], score_source=record["score_source"],
            video_id=record["video_id"], frame_id=record["frame_id"], source_frame_id=record["source_frame_id"],
            timestamp=record["timestamp"], source_timestamp=record["source_timestamp"],
            timestamp_source=record["timestamp_source"], image_size=tuple(record["image_size"]),
            fps=record["fps"], model=record["model"], model_config=record["model_config"],
            checkpoint_sha256=record["checkpoint_sha256"], bbox=np.asarray(record["bbox"], dtype=np.float32),
            bbox_score=record.get("bbox_score"), bbox_source=record["bbox_source"],
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise PoseAdapterError("input_read_failed", str(exc)) from exc


def iter_pose2d_jsonl(input_path: str | Path, summary_path: str | Path):
    input_path = Path(input_path)
    summary = load_success_summary(summary_path, input_path)
    count = 0
    try:
        with input_path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise PoseAdapterError("input_read_failed", f"blank JSONL line {line_number}")
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("expected a JSON object")
                    pose = frame_from_record(record)
                except (ValueError, TypeError) as exc:
                    raise PoseAdapterError("input_read_failed", f"line {line_number}: {exc}") from exc
                if pose.video_id != summary["video_id"]:
                    raise PoseAdapterError("input_summary_invalid", "summary video ID does not match input")
                for name in ("model", "checkpoint_sha256"):
                    if summary.get(name) and summary[name] != getattr(pose, name):
                        raise PoseAdapterError("input_summary_invalid", f"summary {name} does not match input")
                count += 1
                if count > summary["expected_count"]:
                    raise PoseAdapterError("input_summary_invalid", "input exceeds summary frame count")
                yield pose
        if count != summary["expected_count"]:
            raise PoseAdapterError("input_summary_invalid", f"summary expects {summary['expected_count']} records, got {count}")
        yield EndOfPose2D(summary["video_id"], count, count, 0, summary["batches"])
    except OSError as exc:
        raise PoseAdapterError("input_read_failed", str(exc)) from exc
