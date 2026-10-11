"""Portable numeric exports, provenance and human-readable metric files."""

import csv
from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from video_to_3d_motion.pose_adapter.skeleton import H36M17_JOINT_NAMES as JOINT_NAMES
from video_to_3d_motion.pose3d.models import Pose3DSequence


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save_sequence(directory, sequence):
    directory = Path(directory)
    length = sequence.metadata["output_settings"]["obs_len"]
    arrays = {field.name: getattr(sequence, field.name) for field in fields(sequence)
              if isinstance(getattr(sequence, field.name), np.ndarray)}
    arrays["P_obs"] = sequence.P_3d[-length:].copy()
    for name in ("frame_ids", "source_frame_ids", "timestamps", "source_timestamps"):
        arrays["obs_" + name] = arrays[name][-length:].copy()
    np.savez_compressed(directory / "sequence_3d.npz", **arrays)
    metadata = dict(sequence.metadata, joint_names=JOINT_NAMES,
                    parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                    obs_len=length, obs_first_frame_id=int(arrays["obs_frame_ids"][0]),
                    obs_last_frame_id=int(arrays["obs_frame_ids"][-1]),
                    obs_time_span_seconds=float(arrays["obs_timestamps"][-1] - arrays["obs_timestamps"][0]))
    write_json(directory / "sequence_meta.json", metadata)


def load_sequence(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "sequence_meta.json").read_text(encoding="utf-8"))
    with np.load(directory / "sequence_3d.npz", allow_pickle=False) as saved:
        arrays = {field.name: saved[field.name].copy() for field in fields(Pose3DSequence) if field.name != "metadata"}
    return Pose3DSequence(metadata=metadata, **arrays)


def save_evaluation(directory, evaluation):
    metrics, rows, joint_rows, arrays = evaluation
    directory = Path(directory)
    write_json(directory / "metrics.json", metrics)
    write_csv(directory / "frame_metrics.csv", rows)
    for row in joint_rows:
        row["joint_name"] = JOINT_NAMES[row["joint_index"]]
    write_csv(directory / "joint_metrics.csv", joint_rows)
    np.savez_compressed(directory / "evaluation_3d.npz", **arrays)
    write_json(directory / "annotation_match.json", metrics["annotations"])
