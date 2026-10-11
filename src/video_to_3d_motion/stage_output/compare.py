"""Compare existing 3D runs on the intersection of source frame IDs."""

import argparse
from dataclasses import fields, replace
from html import escape
import json
from pathlib import Path
import sys

import numpy as np

from video_to_3d_motion.pose3d.exceptions import Pose3DError

from .annotations import load_annotations
from .archive import load_sequence, write_csv, write_json
from .metrics import evaluate_sequence


def compare_runs(directories, output_dir):
    directories = [Path(path).resolve() for path in directories]
    if len(directories) < 2 or len(set(directories)) != len(directories):
        raise Pose3DError("comparison_invalid", "provide at least two distinct completed runs")
    loaded = []
    common = None
    annotation_identity = None
    for directory in directories:
        summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "completed" or summary.get("evaluation_status") != "completed":
            raise Pose3DError("comparison_invalid", "all runs must have completed H36M evaluation")
        sequence = load_sequence(directory)
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        provenance = metrics["annotations"]
        identity = tuple(provenance[k] for k in ("sequence", "annotation_h5_sha256", "annotation_names_sha256", "units"))
        if annotation_identity is None:
            annotation_identity = identity
        elif annotation_identity != identity:
            raise Pose3DError("comparison_invalid", "runs use different ground truth or sequences")
        current = set(sequence.source_frame_ids.tolist()) & set(metrics["source_frame_ids"])
        common = current if common is None else common & current
        loaded.append((directory, sequence, provenance))
    if not common:
        raise Pose3DError("comparison_invalid", "runs have no common labeled source frames")
    rows = []
    details = []
    for directory, sequence, provenance in loaded:
        keep = np.array([int(frame_id) in common for frame_id in sequence.source_frame_ids])
        arrays = {field.name: getattr(sequence, field.name)[keep].copy()
                  for field in fields(sequence) if isinstance(getattr(sequence, field.name), np.ndarray)}
        selected = replace(sequence, **arrays)
        annotations = load_annotations(provenance["annotation_h5"], provenance["annotation_names"],
                                       provenance["sequence"], selected.source_frame_ids)
        # Check provenance after reading, guarding files modified since a run.
        for name in ("annotation_h5_sha256", "annotation_names_sha256"):
            if annotations.provenance[name] != provenance[name]:
                raise Pose3DError("comparison_invalid", "annotation files changed since the original run")
        metrics = evaluate_sequence(selected, annotations)[0]
        for variant, values in metrics["variants"].items():
            rows.append(dict(run=directory.name, pipeline_mode=sequence.metadata["pipeline_mode"],
                             smoothing=sequence.metadata["output_settings"]["smoothing"], variant=variant,
                             common_frames=len(common), first_source_frame_id=min(common), last_source_frame_id=max(common),
                             n_mpjpe_mm=values["n_mpjpe_mm"]["mean"], pa_mpjpe_mm=values["pa_mpjpe_mm"]["mean"],
                             mpjpe_gt_scale_mm=values["mpjpe_gt_scale_mm"]["mean"],
                             mpjpe_camera_root_depth_mm=values.get("mpjpe_camera_root_depth_mm", {}).get("mean"),
                             acceleration_error_gt_scale_mm_s2=(values["acceleration_error_gt_scale_mm_s2"] or {}).get("mean"),
                             bone_length_cv_mean=values["bone_length_cv_mean"]))
        details.append(dict(run=str(directory), metrics=metrics))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(sequence=annotation_identity[0], common_source_frame_ids=sorted(common),
                  scope="every row evaluated on exactly the same labeled source frames; no whole-dataset benchmark claim",
                  rows=rows, runs=details)
    write_csv(output_dir / "comparison.csv", rows)
    write_json(output_dir / "comparison.json", report)
    # A compact standalone page supplements Excel/JSON without third-party JS.
    sections = []
    for row in rows:
        directory = next(path for path in directories if path.name == row["run"])
        sections.append(f'<section><h2>{escape(row["run"])} · {escape(row["variant"])}</h2>'
                        f'<p>N-MPJPE <strong>{row["n_mpjpe_mm"]:.2f}</strong> mm · '
                        f'PA-MPJPE <strong>{row["pa_mpjpe_mm"]:.2f}</strong> mm · '
                        f'GT 尺度辅助 MPJPE <strong>{row["mpjpe_gt_scale_mm"]:.2f}</strong> mm</p>'
                        f'<p>{escape(row["pipeline_mode"])} / {escape(row["smoothing"])} · '
                        f'<a href="{escape((directory / "viewer.html").as_uri(), quote=True)}">打开骨架查看器</a></p></section>')
    html = '<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>三维结果同帧对比</title>'
    html += '<style>body{font:15px system-ui;max-width:1000px;margin:32px auto;padding:0 20px;background:#f8fafb;color:#15212b}h1{font-size:24px}h2{font-size:18px}section{padding:12px 0;border-top:1px solid #bdcbd5}a{color:#136a9e}strong{font-size:20px}</style>'
    html += f'<h1>{escape(annotation_identity[0])} · 同帧对比</h1><p>所有结果均在相同的 {len(common)} 个标注样本上评价；源帧 {min(common)}–{max(common)}。数值越小越好。</p>'
    html += ''.join(sections) + '<p>GT 尺度辅助结果依赖标注尺度，PA 消除了旋转和平移。不能当作官方完整 H36M 测试集复现。详细数据见 comparison.csv 和 comparison.json。</p></html>'
    (output_dir / "comparison.html").write_text(html, encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare completed 3D runs on identical labeled source frames.")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = compare_runs(args.runs, args.output_dir)
        print(f"COMPLETED: {args.output_dir.resolve() / 'comparison.html'}; common frames={len(report['common_source_frame_ids'])}")
        return 0
    except (Pose3DError, OSError, ValueError, KeyError) as exc:
        print(f"FAILED: {getattr(exc, 'code', 'comparison_invalid')}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
