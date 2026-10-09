"""H36M original-pixel comparison side output; does not change model predictions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from video_to_3d_motion.media.image_sequence_decoder import decode_image_bgr

from .config import VisualizationConfig
from .experiment_reports import COMMON10_H36M_INDICES
from .h36m_annotations import image_key, load_sequence_annotations
from .visualization import H36M17_EDGES, _draw_line, _draw_text, _paint_disk, _write_image

COCO_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)
H36M_NAMES = (
    "root", "right_hip", "right_knee", "right_foot", "left_hip", "left_knee",
    "left_foot", "spine", "thorax", "neck_base", "head", "left_shoulder",
    "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
)
# No fabricated pelvis/spine/head predictions are used for evaluation. H36M
# foot vs COCO ankle correspondences are explicitly labelled as proxies.
COCO_TO_H36M = {5: 11, 6: 14, 7: 12, 8: 15, 9: 13, 10: 16,
                11: 4, 12: 1, 13: 5, 14: 2, 15: 6, 16: 3}
JOINT_LABELS = dict(zip(
    COCO_NAMES + ("root", "spine", "thorax", "neck_base", "head", "left_foot", "right_foot"),
    ("鼻", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩", "左肘", "右肘", "左腕", "右腕",
     "左髋", "右髋", "左膝", "右膝", "左踝", "右踝", "骨盆根节点", "脊柱", "胸部", "颈部", "头", "左足", "右足")
))
FIELDS = (
    "image", "frame_id", "source_frame_id", "joint_schema", "joint_index",
    "joint_name", "joint_name_zh", "gt_joint_index", "gt_joint_name", "mapping", "pred_x",
    "pred_y", "gt_x", "gt_y", "error_px", "confidence", "prediction_valid",
    "gt_visible", "status", "bbox_source", "bbox_selection",
)


def compare_record(record: dict, annotation: dict | None, image: str) -> list[dict]:
    schema = record["joint_schema"]
    if schema not in {"coco17", "h36m17"}:
        raise ValueError(f"unsupported joint schema: {schema}")
    names = COCO_NAMES if schema == "coco17" else H36M_NAMES
    mapping = COCO_TO_H36M if schema == "coco17" else dict(enumerate(range(17)))
    points = np.asarray(record["keypoints"], dtype=float)
    scores = np.asarray(record["keypoint_scores_raw"], dtype=float)
    valid = np.asarray(record["valid_mask"], dtype=bool)
    if points.shape != (17, 2) or scores.shape != (17,) or valid.shape != (17,):
        raise ValueError("prediction must have 17 keypoints, scores and valid flags")
    gt = np.asarray(annotation["keypoints"], dtype=float) if annotation else None
    visible = np.asarray(annotation["visible"], dtype=bool) if annotation else None
    if annotation and (gt.shape != (17, 2) or visible.shape != (17,) or not np.isfinite(gt).all()):
        raise ValueError("invalid H36M annotation")
    rows = []
    for index, name in enumerate(names):
        gt_index = mapping.get(index)
        relationship = "unmapped" if gt_index is None else (
            "ankle_foot_proxy" if schema == "coco17" and index in {15, 16} else "direct"
        )
        finite = bool(np.isfinite(points[index]).all() and np.isfinite(scores[index]))
        row = dict(image=image, frame_id=record["frame_id"], source_frame_id=record["source_frame_id"],
                   joint_schema=schema, joint_index=index, joint_name=name,
                   joint_name_zh=JOINT_LABELS[name],
                   gt_joint_index=gt_index, gt_joint_name=H36M_NAMES[gt_index] if gt_index is not None else None,
                   mapping=relationship, pred_x=float(points[index, 0]) if finite else None,
                   pred_y=float(points[index, 1]) if finite else None,
                   gt_x=None, gt_y=None, error_px=None,
                   confidence=float(scores[index]) if np.isfinite(scores[index]) else None,
                   prediction_valid=bool(valid[index] and finite), gt_visible=None,
                   status="missing_gt", bbox_source=record.get("bbox_source"),
                   bbox_selection=record.get("bbox_selection", "unspecified"))
        if annotation is not None:
            if gt_index is None:
                row["status"] = "unmapped_joint"
            else:
                row.update(gt_x=float(gt[gt_index, 0]), gt_y=float(gt[gt_index, 1]),
                           gt_visible=bool(visible[gt_index]))
                if not visible[gt_index]:
                    row["status"] = "gt_invisible"
                elif not finite:
                    row["status"] = "invalid_prediction"
                else:
                    row["error_px"] = float(np.linalg.norm(points[index] - gt[gt_index]))
                    row["status"] = "ok" if valid[index] else "low_confidence"
        rows.append(row)
    return rows


def _stats(values) -> dict:
    if not values:
        return {"count": 0, "mean_px": None, "median_px": None, "p95_px": None}
    return {"count": len(values), "mean_px": float(np.mean(values)),
            "median_px": float(np.median(values)), "p95_px": float(np.percentile(values, 95))}


def _overlay(image_path: Path, record: dict, annotation: dict | None, rows, destination: Path):
    image = decode_image_bgr(image_path)
    if tuple(record["image_size"]) != (image.shape[1], image.shape[0]):
        raise ValueError(f"prediction/image dimensions differ: {image_path}")
    if annotation:
        gt = np.asarray(annotation["keypoints"])
        visible = annotation["visible"]
        for first, second in H36M17_EDGES:
            if visible[first] and visible[second]:
                _draw_line(image, tuple(gt[first]), tuple(gt[second]), (0, 255, 255), 1)
        for point, is_visible in zip(gt, visible):
            if is_visible:
                _paint_disk(image, int(round(point[0])), int(round(point[1])), 5, (0, 255, 255))
    for row in rows:
        if row["error_px"] is not None:
            _draw_line(image, (row["pred_x"], row["pred_y"]),
                       (row["gt_x"], row["gt_y"]), (170, 170, 170), 1)
        if row["pred_x"] is not None:
            color = (0, 220, 0) if row["prediction_valid"] else (0, 0, 255)
            _paint_disk(image, round(row["pred_x"]), round(row["pred_y"]), 3, color)
    image[2:22, 2:min(image.shape[1], 280)] = 0
    _draw_text(image, "PRED", (6, 6), (0, 220, 0))
    _draw_text(image, "GT" if annotation else "NO GT", (58, 6), (0, 255, 255))
    _draw_text(image, f"F:{record['frame_id']:06d}", (112, 6))
    _write_image(image, destination, VisualizationConfig())


def run_comparison(source: Path, frames_output: Path, annotations: Path, output: Path,
                   *, every_n_frames: int = 1) -> dict:
    if every_n_frames <= 0:
        raise ValueError("comparison interval must be positive")
    source, output = source.resolve(), output.resolve()
    image_paths = {}
    pattern = re.compile(r"_(\d{6})\.(?:jpg|jpeg|png)$", re.IGNORECASE)
    for path in source.iterdir():
        if path.is_file() and (match := pattern.search(path.name)):
            number = int(match.group(1))
            if number in image_paths:
                raise ValueError(f"duplicate source frame: {number}")
            image_paths[number] = path
    ground_truth = load_sequence_annotations(annotations, source)
    if not ground_truth:
        raise ValueError(f"no matching annotations for {source.name}; check the train/validation file")
    output.mkdir(parents=True, exist_ok=True)
    per_joint = defaultdict(lambda: {"all": [], "confident": [], "eligible": 0, "mapping": None})
    counts = Counter({key: 0 for key in (
        "frames", "matched_frames", "missing_gt_frames", "eligible_joint_pairs", "overlays",
        "ok", "low_confidence", "gt_invisible", "unmapped_joint", "missing_gt", "invalid_prediction"
    )})
    frame_ids = set()
    all_errors, confident_errors, direct_errors, proxy_errors, common10_errors = [], [], [], [], []
    matched_images = set()
    bbox_selections = set()
    with frames_output.open(encoding="utf-8") as stream, (output / "joint_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as table:
        writer = csv.DictWriter(table, fieldnames=FIELDS)
        writer.writeheader()
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            bbox_selections.add(record.get("bbox_selection", "unspecified"))
            if record.get("coordinate_space", "original_image_pixels") != "original_image_pixels":
                raise ValueError("comparison requires original-image pixel coordinates")
            if record.get("source_path") and Path(record["source_path"]).resolve() != source:
                raise ValueError("prediction source_path does not match the image sequence")
            source_id = int(record["source_frame_id"])
            if source_id in frame_ids:
                raise ValueError(f"duplicate prediction source frame: {source_id}")
            frame_ids.add(source_id)
            path = image_paths.get(source_id)
            if path is None:
                raise ValueError(f"prediction source frame has no image: {source_id}")
            key = image_key(f"{source.name}/{path.name}")
            annotation = ground_truth.get(key)
            counts["frames"] += 1
            counts["matched_frames" if annotation else "missing_gt_frames"] += 1
            if annotation:
                matched_images.add(key)
            rows = compare_record(record, annotation, key)
            writer.writerows(rows)
            for row in rows:
                counts[row["status"]] += 1
                stats = per_joint[(record["joint_schema"], row["joint_name"])]
                stats["mapping"] = row["mapping"]
                if row["gt_visible"]:
                    stats["eligible"] += 1
                    counts["eligible_joint_pairs"] += 1
                if row["error_px"] is not None:
                    error = row["error_px"]
                    all_errors.append(error)
                    stats["all"].append(error)
                    (proxy_errors if row["mapping"] == "ankle_foot_proxy" else direct_errors).append(error)
                    if row["mapping"] == "direct" and row["gt_joint_index"] in COMMON10_H36M_INDICES:
                        common10_errors.append(error)
                    if row["prediction_valid"]:
                        confident_errors.append(error)
                        stats["confident"].append(error)
            if (int(record["frame_id"]) - 1) % every_n_frames == 0:
                _overlay(path, record, annotation, rows, output / "overlays" / f"frame_{record['frame_id']:06d}.jpg")
                counts["overlays"] += 1
    if not frame_ids:
        raise ValueError("prediction JSONL is empty")
    fields = ("joint_schema", "joint_name", "joint_name_zh", "mapping", "eligible_pairs", "evaluated_pairs",
              "confident_pairs", "confidence_coverage", "mean_px", "median_px", "p95_px", "confident_mean_px")
    with (output / "joint_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (schema, name), stats in per_joint.items():
            errors = _stats(stats["all"])
            writer.writerow(dict(joint_schema=schema, joint_name=name, mapping=stats["mapping"],
                                 joint_name_zh=JOINT_LABELS[name],
                                 eligible_pairs=stats["eligible"], evaluated_pairs=errors["count"],
                                 confident_pairs=len(stats["confident"]),
                                 confidence_coverage=len(stats["confident"]) / stats["eligible"] if stats["eligible"] else None,
                                 mean_px=errors["mean_px"], median_px=errors["median_px"], p95_px=errors["p95_px"],
                                 confident_mean_px=_stats(stats["confident"])["mean_px"]))
    eligible = counts["eligible_joint_pairs"]
    summary = {
        "status": "completed", "source": str(source), "annotations": str(annotations.resolve()),
        "prediction_file": str(frames_output.resolve()), "counts": dict(counts),
        "bbox_selection_policies": sorted(bbox_selections),
        "unused_annotation_frames": len(set(ground_truth) - matched_images),
        "all_visible_pairs": _stats(all_errors), "confident_pairs": _stats(confident_errors),
        "direct_pairs": _stats(direct_errors), "ankle_foot_proxy_pairs": _stats(proxy_errors),
        "common10_pairs": _stats(common10_errors),
        "confidence_coverage": len(confident_errors) / eligible if eligible else None,
        "coordinate_space": "original_image_pixels", "gt_joint_schema": "h36m17_standard",
        "annotation_joint_order_assumption": "MMPose standard H36M17; verify against overlay and dataset producer",
        "accuracy_pass": None,
        "notes": ["No accuracy pass threshold has been specified.",
                  "Low-confidence predictions remain in all_visible_pairs; confident_pairs is reported separately.",
                  "COCO nose/eyes/ears are not compared; ankle-to-foot proxy pairs are reported separately.",
                  "S1 train-sequence results are a diagnostic, not a held-out validation benchmark."],
        "overlay_legend": {"green": "prediction", "red": "low-confidence prediction",
                           "yellow": "H36M ground truth", "gray": "paired-joint displacement"},
        "joint_comparison": str(output / "joint_comparison.csv"),
        "joint_summary": str(output / "joint_summary.csv"), "overlays": str(output / "overlays"),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare saved S1-02 predictions with H36M original-pixel GT")
    parser.add_argument("source", type=Path)
    parser.add_argument("--frames-output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--every-n-frames", type=int, default=1)
    args = parser.parse_args()
    summary = run_comparison(args.source, args.frames_output, args.annotations, args.output,
                             every_n_frames=args.every_n_frames)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
