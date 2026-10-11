"""Controlled single-sequence GT-2D versus detected-2D lifting experiment."""

import argparse
import csv
from dataclasses import asdict, replace
from html import escape
import json
from pathlib import Path
import re
from time import perf_counter

import h5py
import numpy as np
import yaml

from video_to_3d_motion.pose_adapter.normalization import normalize_h36m
from video_to_3d_motion.pose_adapter.windows import prepare_sequence_windows
from video_to_3d_motion.pose3d.backend import PoseMambaBackend, file_sha256
from video_to_3d_motion.pose3d.config import load_pose3d_config
from video_to_3d_motion.stage_output.annotations import H5_TO_H36M17, load_annotations
from video_to_3d_motion.stage_output.archive import (
    load_sequence, save_evaluation, save_sequence, write_csv, write_json,
)
from video_to_3d_motion.stage_output.compare import compare_runs
from video_to_3d_motion.stage_output.fusion import fuse_windows
from video_to_3d_motion.stage_output.metrics import evaluate_sequence
from video_to_3d_motion.stage_output.viewer import create_viewer


def read_gt_pixels(settings, source_ids):
    """Read only 2D labels before inference; camera-space 3D is not read here."""
    names = settings.annotation_names.read_text(encoding="utf-8-sig").splitlines()
    pattern = re.compile(re.escape(settings.sequence) + r"_(\d{6})\.(?:jpg|jpeg|png)$", re.I)
    by_id = {}
    for row, name in enumerate(names):
        match = pattern.fullmatch(name.replace("\\", "/").split("/")[-1])
        if match:
            source_id = int(match[1])
            if source_id in by_id:
                raise ValueError("duplicate annotation source frame")
            by_id[source_id] = row
    rows = np.array([by_id[int(source_id)] for source_id in source_ids], dtype=np.int64)
    order = np.argsort(rows)
    pixels = np.empty((len(rows), 17, 2), dtype=np.float64)
    with h5py.File(settings.annotation_h5, "r") as labels:
        if labels["part"].shape != (len(names), 17, 2):
            raise ValueError("annotation names/2D array mismatch")
        if np.any(labels["normalize"][rows[order]] != 0):
            raise ValueError("expected original pixel labels")
        pixels[order] = labels["part"][rows[order]][:, H5_TO_H36M17]
    if not np.isfinite(pixels).all():
        raise ValueError("nonfinite GT pixels")
    return pixels, rows


def render_existing(output_root, detected_dir):
    """Refresh presentation of completed results without re-running inference."""
    result = json.loads((output_root / "experiment.json").read_text(encoding="utf-8"))
    titles = {"n_mpjpe_mm": "N-MPJPE", "pa_mpjpe_mm": "PA-MPJPE",
              "mpjpe_gt_scale_mm": "GT 尺度辅助 MPJPE",
              "mpjpe_camera_root_depth_mm": "相机 MPJPE（GT 根深度辅助）"}
    rows = "".join(
        f'<tr><th>{titles[name]}</th><td>{values["detected_2d"]:.2f}</td>'
        f'<td>{values["gt_2d"]:.2f}</td><td>{values["reduction_mm"]:.2f}</td>'
        f'<td>{values["reduction_percent"]:.1f}%</td></tr>'
        for name, values in result["metrics"].items()
    )
    page = '''<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>二维输入对三维结果的影响</title>
<style>body{font:16px system-ui,"Microsoft YaHei";max-width:1000px;margin:32px auto;padding:0 24px;color:#15212b;background:#f8fafb}h1{font-size:24px}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{text-align:left;padding:14px 10px;border-bottom:1px solid #bdcbd5}p{line-height:1.8}a{color:#136a9e}.links{display:flex;gap:24px;flex-wrap:wrap}.muted{color:#526371}</style>
<h1>标注 2D 与图片检测 2D：三维指标对比</h1>'''
    page += f'<p>{escape(result["sequence"])} · 同一批 {result["frame_count"]} 帧 · 实际 {result["controls"]["effective_fps"]:.1f} FPS<br>相同模型和翻转测试；243 帧窗口、步长 81、重叠平均融合；无平滑。</p>'
    page += '<table><thead><tr><th>指标 · mm，越小越好</th><th>图片 → RTMPose 2D → 3D</th><th>标注 2D → 3D</th><th>误差减少 · mm</th><th>误差减少比例</th></tr></thead><tbody>' + rows + '</tbody></table>'
    page += f'<p>检测 2D 与标注 2D 的平均关节距离：{result["detected_2d_error_pixels_all17"]:.2f} 像素（17 个关节）。标注输入来自 H5 的 part 字段，先转换关节顺序，再做相同归一化。</p>'
    page += f'<div class="links"><a href="{escape((detected_dir / "viewer.html").as_uri(), quote=True)}">查看图片检测输入的三维骨架</a><a href="../gt2d_full/viewer.html">查看标注输入的三维骨架</a><a href="comparison.csv">下载指标 CSV</a><a href="joint_input_comparison.csv">下载逐关节对比 CSV</a></div>'
    page += '<p class="muted">标注输入实验用于分析二维检测误差的影响，实际使用仍需要检测器。PA 允许尺度、旋转和平移对齐；N 只允许尺度对齐。相机 MPJPE 通过已核验的相机参数和 GT 根节点位置、深度恢复毫米坐标，不进行误差拟合；它与论文的官方 2.5D P1 口径不同。GT 尺度辅助结果是近似诊断。两组差值反映模型对二维输入的敏感性，不能解释为可直接相加的二维、三维误差分解。本次只测试单动作单相机；S1 是训练主体，不能作为论文测试集成绩。</p></html>'
    (output_root / "comparison" / "comparison.html").write_text(page, encoding="utf-8")
    viewer = output_root / "gt2d_full" / "viewer.html"
    viewer_text = viewer.read_text(encoding="utf-8").replace(
        "<h1>三维姿态验证</h1>", "<h1>三维姿态验证 · 标注 2D 输入</h1>",
    ).replace("<title>三维姿态验证</title>", "<title>标注 2D 输入 · 三维姿态验证</title>")
    viewer.write_text(viewer_text, encoding="utf-8")


def run(detected_dir, output_root, image_dir=None):
    started = perf_counter()
    detected_dir, output_root = detected_dir.resolve(), output_root.resolve()
    original = json.loads((detected_dir / "run_summary.json").read_text(encoding="utf-8"))
    if original["status"] != "completed" or original["evaluation_status"] != "completed":
        raise ValueError("detected-2D reference must be completed and evaluated")
    config = load_pose3d_config(detected_dir / "config_snapshot.yaml")
    if config.pipeline_mode != "full" or config.output.smoothing != "none":
        raise ValueError("use a full-sequence unsmoothed detected-2D reference")
    for name in ("input", "input_summary"):
        if file_sha256(Path(original[name])) != original[name + "_sha256"]:
            raise ValueError("detected-2D reference input changed: " + name)
    windows, adapter = prepare_sequence_windows(
        original["input"], original["input_summary"], mode="full", stride=config.window_stride,
    )
    reference = load_sequence(detected_dir)
    frame_count = adapter.frames_received
    expected = np.arange(1, frame_count + 1, dtype=np.int64)
    if not np.array_equal(reference.frame_ids, expected):
        raise ValueError("reference frame coverage differs")
    detected_norm = np.empty((frame_count, 17, 2), dtype=np.float32)
    coverage = np.zeros(frame_count, dtype=np.int64)
    for window in windows:
        positions = window.frame_ids - 1
        for name in ("source_frame_ids", "timestamps", "source_timestamps"):
            if not np.array_equal(getattr(reference, name)[positions], getattr(window, name)):
                raise ValueError("reference metadata differs: " + name)
        existing = coverage[positions] > 0
        if not np.array_equal(detected_norm[positions[existing]], window.keypoints_norm[0, existing]):
            raise ValueError("inconsistent detected 2D overlap")
        detected_norm[positions] = window.keypoints_norm[0]
        coverage[positions] += 1
    if not (coverage > 0).all() or not np.array_equal(coverage, reference.contribution_count):
        raise ValueError("reference overlap coverage differs")
    pixels, annotation_rows = read_gt_pixels(config.evaluation, reference.source_frame_ids)
    image_size = windows[0].image_size
    gt_norm = np.stack([normalize_h36m(pose, image_size) for pose in pixels])
    width, height = image_size
    detected_pixels = (detected_norm.astype(np.float64) + [1, height / width]) * (width / 2)
    input_errors = np.linalg.norm(detected_pixels - pixels, axis=-1)
    provenance = dict(
        input_kind="h36m_annot_2d_ground_truth", annotation_h5=str(config.evaluation.annotation_h5),
        annotation_h5_sha256=file_sha256(config.evaluation.annotation_h5),
        annotation_names=str(config.evaluation.annotation_names),
        annotation_names_sha256=file_sha256(config.evaluation.annotation_names),
        label_to_model_permutation=H5_TO_H36M17, fields_used_for_model_input=["part"],
        sequence=config.evaluation.sequence, detected_reference=str(detected_dir),
        coordinate_space="original_pixels_then_h36m_width_center_v1",
    )
    output_root.mkdir(parents=True, exist_ok=False)
    gt_dir = output_root / "gt2d_full"
    gt_dir.mkdir()
    (gt_dir / "windows").mkdir()
    (gt_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(json.loads(json.dumps(asdict(config), default=str)), sort_keys=False), encoding="utf-8",
    )
    write_json(gt_dir / "input_provenance.json", provenance)
    np.savez_compressed(
        gt_dir / "input_2d.npz", gt_pixels=pixels, gt_normalized=gt_norm,
        detected_pixels=detected_pixels, detected_normalized=detected_norm,
        frame_ids=expected, source_frame_ids=reference.source_frame_ids,
        timestamps=reference.timestamps, annotation_row_indices=annotation_rows,
    )
    backend = PoseMambaBackend(config.model)
    current_info = backend.runtime_info()
    for name in ("checkpoint_sha256", "backend", "flip_test", "units", "upstream_revision", "torch_version", "device"):
        if current_info[name] != original["model_info"][name]:
            raise ValueError("3D model/reference settings differ: " + name)
    lifted = []
    for index, window in enumerate(windows, 1):
        gt_window = replace(
            window, keypoints_norm=gt_norm[window.frame_ids - 1][None].copy(),
            model="h36m_annot_2d_ground_truth", model_config=str(config.evaluation.annotation_h5),
            checkpoint_sha256=provenance["annotation_h5_sha256"], score_source="ground_truth_available",
            keypoint_scores_raw=np.ones((243, 17), dtype=np.float32),
            valid_mask=np.ones((243, 17), dtype=np.bool_),
        )
        prediction = backend.predict(gt_window)
        lifted.append(prediction)
        np.savez_compressed(
            gt_dir / "windows" / f"window_{index:06d}.npz", pose_3d=prediction.pose_3d,
            keypoints_norm=gt_window.keypoints_norm, frame_ids=window.frame_ids,
            source_frame_ids=window.source_frame_ids, timestamps=window.timestamps,
            source_timestamps=window.source_timestamps,
        )
        print(f"GT2D: window {index}/{len(windows)}, {prediction.inference_seconds:.2f}s", flush=True)
    sequence = fuse_windows(lifted, expected, config.output)
    sequence.metadata["input_provenance"] = provenance
    # Only now read 3D labels for evaluation.
    annotations = load_annotations(
        config.evaluation.annotation_h5, config.evaluation.annotation_names,
        config.evaluation.sequence, sequence.source_frame_ids,
    )
    np.testing.assert_array_equal(annotations.joints_pixel, pixels)
    for name in ("frame_ids", "source_frame_ids", "timestamps", "source_timestamps", "contribution_count"):
        np.testing.assert_array_equal(getattr(sequence, name), getattr(reference, name))
    evaluation = evaluate_sequence(sequence, annotations)
    save_evaluation(gt_dir, evaluation)
    save_sequence(gt_dir, sequence)
    viewer = create_viewer(gt_dir, sequence, evaluation=evaluation, image_dir=image_dir,
                           sequence_prefix=config.evaluation.sequence)
    write_json(gt_dir / "run_summary.json", dict(
        status="completed", evaluation_status="completed", input_kind=provenance["input_kind"],
        input_provenance=provenance, pipeline_mode="full", frame_count=frame_count,
        window_count=len(windows), model_info=backend.runtime_info(),
        inference_seconds=sum(prediction.inference_seconds for prediction in lifted),
        total_seconds=perf_counter() - started, P_3d_shape=list(sequence.P_3d.shape),
        P_obs_shape=[10, 17, 3], viewer=str(viewer), sequence_output=str(gt_dir / "sequence_3d.npz"),
    ))
    report = compare_runs([detected_dir, gt_dir], output_root / "comparison")
    rows = {row["run"]: row for row in report["rows"] if row["variant"] == "output"}
    measured, ideal = rows[detected_dir.name], rows[gt_dir.name]
    comparison = {}
    metric_names = ["n_mpjpe_mm", "pa_mpjpe_mm", "mpjpe_gt_scale_mm"]
    if measured.get("mpjpe_camera_root_depth_mm") is not None and ideal.get("mpjpe_camera_root_depth_mm") is not None:
        metric_names.append("mpjpe_camera_root_depth_mm")
    for name in metric_names:
        difference = measured[name] - ideal[name]
        comparison[name] = dict(detected_2d=measured[name], gt_2d=ideal[name],
                                reduction_mm=difference, reduction_percent=100 * difference / measured[name])
    result = dict(
        sequence=config.evaluation.sequence, frame_count=frame_count,
        detected_2d_error_pixels_all17=float(input_errors.mean()),
        detected_2d_error_pixels_nonroot16=float(input_errors[:, 1:].mean()),
        metrics=comparison,
        controls=dict(same_source_frames=True, same_timestamps=True, same_windows=True,
                      same_model=True, flip_test=config.model.flip_test, smoothing="none",
                      clip_len=243, stride=81, effective_fps=float(1 / np.median(np.diff(sequence.timestamps)))),
        interpretation="GT-2D input is an offline diagnostic; changes measure input sensitivity, not an additive error decomposition",
        verification=dict(gt_input_matches_annotation=True, same_output_metadata=True,
                          finite_output=bool(np.isfinite(sequence.P_3d).all()),
                          root_zero=bool((sequence.P_3d[:, 0] == 0).all())),
    )
    write_json(output_root / "experiment.json", result)
    with (detected_dir / "joint_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        detected_joints = [row for row in csv.DictReader(stream) if row["variant"] == "output"]
    gt_joints = [row for row in evaluation[2] if row["variant"] == "output"]
    joint_rows = []
    for detected, gt in zip(detected_joints, gt_joints):
        if int(detected["joint_index"]) != gt["joint_index"]:
            raise ValueError("joint report order differs")
        row = dict(joint_index=gt["joint_index"], joint_name=detected["joint_name"])
        for name in comparison:
            row[name + "_detected"] = float(detected[name])
            row[name + "_gt2d"] = gt[name]
            row[name + "_reduction"] = float(detected[name]) - gt[name]
        joint_rows.append(row)
    write_csv(output_root / "comparison" / "joint_input_comparison.csv", joint_rows)
    render_existing(output_root, detected_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("COMPARISON:", output_root / "comparison" / "comparison.html", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detected-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-dir", type=Path)
    args = parser.parse_args()
    run(args.detected_run, args.output_dir, args.image_dir)
