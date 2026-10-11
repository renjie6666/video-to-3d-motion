"""Single-action numerical, calibration and protocol audits; no full benchmark."""

import argparse
import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np

from video_to_3d_motion.pose_adapter.normalization import normalize_h36m
from video_to_3d_motion.pose_adapter.windows import prepare_sequence_windows
from video_to_3d_motion.pose3d.backend import PoseMambaBackend
from video_to_3d_motion.pose3d.config import ModelSettings
from video_to_3d_motion.stage_output.annotations import load_annotations
from video_to_3d_motion.stage_output.archive import load_sequence, save_evaluation, save_sequence, write_csv, write_json
from video_to_3d_motion.stage_output.calibration import CAMERA_54138969, project_camera, undistort_pixels, verify_camera_labels
from video_to_3d_motion.stage_output.fusion import fuse_windows
from video_to_3d_motion.pose3d.config import OutputSettings
from video_to_3d_motion.stage_output.metrics import align_procrustes, evaluate_sequence, apply_official_25d
from video_to_3d_motion.stage_output.viewer import create_viewer

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/reproduction_audit_2026-10-10"


def load_labels(directory, sequence):
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    info = metrics["annotations"]
    return load_annotations(info["annotation_h5"], info["annotation_names"], info["sequence"], sequence.source_frame_ids)


def audit_structure():
    def model_ast(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PoseMamba")
        for item in ast.walk(node):
            if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.body:
                if isinstance(item.body[0], ast.Expr) and isinstance(item.body[0].value, ast.Constant) and isinstance(item.body[0].value.value, str):
                    item.body.pop(0)
        return ast.dump(node, include_attributes=False)
    equal = model_ast(ROOT / "work/posemamba_source/lib/model/PoseMamba.py") == model_ast(ROOT / "src/video_to_3d_motion/pose3d/vendor/model.py")
    assert equal, "vendored model architecture differs from pinned original class"
    from video_to_3d_motion.pose3d.vendor.model import PoseMamba
    model = PoseMamba(num_frame=243, num_joints=17, in_chans=2, embed_dim_ratio=128, depth=20, mlp_ratio=2)
    full = sum(p.numel() for p in model.parameters())
    leaf = sum(sum(p.numel() for p in m.parameters(recurse=False)) for m in model.modules() if not list(m.children()))
    result = dict(status="passed", pinned_original_class_ast_equal=equal,
                  spatial_blocks=len(model.STEblocks), temporal_blocks=len(model.TTEblocks),
                  feature_dim=128, window=243, full_parameter_count=full,
                  leaf_parameter_count=leaf, direct_custom_parameter_count=full-leaf)
    write_json(AUDIT / "structure_audit.json", result)
    print(json.dumps(result), flush=True)


def audit_scan():
    import torch
    from video_to_3d_motion.pose3d.vendor import scan
    from video_to_3d_motion.pose3d.vendor.scan import CrossScan_plus_poselimbs, CrossMerge_plus_poselimbs
    original_tree = ast.parse((ROOT / "work/posemamba_source/lib/model/csms6s.py").read_text(encoding="utf-8"))
    selected = [n for n in original_tree.body if isinstance(n, ast.ClassDef) and n.name in {"CrossScan_plus_poselimbs", "CrossMerge_plus_poselimbs"}]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "pinned_original_cross_scan", "exec"), namespace)
    torch.manual_seed(19)
    x = torch.randn(1, 3, 243, 17)
    np.testing.assert_array_equal(CrossScan_plus_poselimbs.apply(x), namespace["CrossScan_plus_poselimbs"].apply(x))
    y = torch.randn(1, 4, 3, 243, 17)
    np.testing.assert_array_equal(CrossMerge_plus_poselimbs.apply(y), namespace["CrossMerge_plus_poselimbs"].apply(y))
    reference_dir = ROOT / "results/s1_04_05_acceptance_2026-10-10/s9_full"
    summary = json.loads((reference_dir / "run_summary.json").read_text())
    windows, _ = prepare_sequence_windows(summary["input"], summary["input_summary"], mode="full")
    backend = PoseMambaBackend(ModelSettings())
    original_scan = scan.selective_scan_torch
    baseline = backend.predict(windows[0])
    with np.load(reference_dir / "windows/window_000001.npz") as saved:
        np.testing.assert_array_equal(baseline.pose_3d, saved["pose_3d"])
    records = []
    def checked_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, chunk_size=256):
        actual = original_scan(u, delta, A, B, C, D, delta_bias, delta_softplus, chunk_size)
        groups = B.shape[1]
        per_group = u.shape[1] // groups
        indices = np.array([g * per_group + c for g in range(groups) for c in [0, per_group//3, 2*per_group//3, per_group-1]])
        def cpu(t):
            return t.detach().cpu().numpy().astype(np.float64)
        us, dt = cpu(u[:, indices]), cpu(delta[:, indices])
        if delta_bias is not None:
            dt += cpu(delta_bias)[indices][None, :, None]
        if delta_softplus:
            dt = np.logaddexp(0, dt)
        av, bv, cv = cpu(A)[indices], cpu(B), cpu(C)
        state = np.zeros((u.shape[0], len(indices), A.shape[-1]))
        expected = np.empty_like(us)
        group_ids = indices // per_group
        dv = cpu(D)[indices] if D is not None else None
        for t in range(u.shape[-1]):
            state = np.exp(dt[..., t, None] * av) * state + dt[..., t, None] * bv[..., t][:, group_ids] * us[..., t, None]
            expected[..., t] = (cv[..., t][:, group_ids] * state).sum(axis=-1)
            if dv is not None:
                expected[..., t] += dv * us[..., t]
        observed = cpu(actual[:, indices])
        records.append(dict(call=len(records)+1, channels_checked=len(indices), length=u.shape[-1],
                            max_absolute_error=float(np.abs(observed-expected).max()),
                            rmse=float(np.sqrt(np.mean((observed-expected)**2))),
                            within_initial_2e5_tolerance=bool(np.allclose(observed, expected, atol=2e-5, rtol=2e-5))))
        if len(records) % 10 == 0:
            print(f"Scan audit: {len(records)}/80 learned scan calls verified", flush=True)
        # Inject the independent float64 recurrence on the checked channels.
        # Final 3D output sensitivity is more informative than a fixed tolerance
        # on intermediate activations whose magnitudes vary across blocks.
        corrected = actual.clone()
        corrected[:, indices] = torch.from_numpy(expected.astype(np.float32)).to(actual.device)
        return corrected
    scan.selective_scan_torch = checked_scan
    try:
        result = backend.predict(windows[0])
    finally:
        scan.selective_scan_torch = original_scan
    assert len(records) == 80
    difference = np.linalg.norm(result.pose_3d - baseline.pose_3d, axis=-1)
    conservative_mm = difference * 2500  # width/2 * conservative 5 mm/pixel scale
    assert float(conservative_mm.max()) < .1, "scan numerical errors materially change 3D predictions"
    scan.selective_scan_torch = lambda *args, **kwargs: original_scan(*args, **dict(kwargs, chunk_size=64))
    try:
        alternate = backend.predict(windows[0])
    finally:
        scan.selective_scan_torch = original_scan
    chunk_difference_mm = np.linalg.norm(alternate.pose_3d-baseline.pose_3d, axis=-1)*2500
    assert float(chunk_difference_mm.max()) < .1, "scan accumulation sensitivity exceeds 0.1 mm"
    report = dict(status="passed", original_cross_scan_merge_exact=True,
                  learned_blocks_checked=40, flip_passes=2, calls=records,
                  max_absolute_error=max(r["max_absolute_error"] for r in records),
                  archived_prediction_exact=True,
                  independent_reference_injected_max_3d_difference_mm=float(conservative_mm.max()),
                  independent_reference_injected_mean_3d_difference_mm=float(conservative_mm.mean()),
                  chunk64_vs256_max_3d_difference_mm=float(chunk_difference_mm.max()),
                  reference="independent NumPy float64 serial recurrence, 16 stratified channels per scan, full 4131 steps",
                  limitation="not a direct comparison against the original compiled CUDA kernel")
    write_json(AUDIT / "all_blocks_scan_audit.json", report)
    print(json.dumps({k:v for k,v in report.items() if k != "calls"}), flush=True)


def source_cases():
    return [
        ("S9 detected 2D", ROOT / "results/s1_04_05_acceptance_2026-10-10/s9_full", AUDIT / "s9_detected_full"),
        ("S9 GT 2D", ROOT / "results/gt2d_vs_detected2d_s9_2026-10-10/gt2d_full", AUDIT / "s9_gt2d_full"),
        ("S1 detected 2D", AUDIT / "s1_detected_full", AUDIT / "s1_detected_full"),
        ("S1 GT 2D", AUDIT / "s1_input_comparison/gt2d_full", AUDIT / "s1_input_comparison/gt2d_full"),
    ]


def audit_evaluation():
    rows = []
    for label, source, destination in source_cases():
        sequence = load_sequence(source)
        annotations = load_labels(source, sequence)
        destination.mkdir(parents=True, exist_ok=True)
        if source != destination:
            for name in ["run_summary.json", "config_snapshot.yaml"]:
                shutil.copy2(source / name, destination / name)
        verify_camera_labels(annotations.joints_camera_mm, annotations.joints_pixel)
        np.testing.assert_allclose(undistort_pixels(annotations.joints_pixel),
                                   project_camera(annotations.joints_camera_mm, distorted=False), atol=.0001)
        evaluation = evaluate_sequence(sequence, annotations)
        if (destination / "official_25d_labels.npz").is_file():
            evaluation = apply_official_25d(evaluation, sequence, destination / "official_25d_labels.npz")
        save_sequence(destination, sequence)
        save_evaluation(destination, evaluation)
        image_dir = (AUDIT / "input/S1_Directions.54138969" if label.startswith("S1")
                     else ROOT / "results/s9_generalization_directions_1_54138969/input/S9_Directions_1.54138969")
        viewer = create_viewer(destination, sequence, evaluation=evaluation, image_dir=image_dir,
                               sequence_prefix=annotations.provenance["sequence"])
        # Reconstruct an explicitly labeled LCN/PAMI-style 2.5D protocol.
        # The original MotionBERT pickle is absent, so do not label this official P1.
        camera = annotations.joints_camera_mm
        camera_relative = camera - camera[:, :1]
        pinhole = project_camera(camera, distorted=False)
        ratio = CAMERA_54138969["focal"][0] / camera[:, :1, 2:] + 1/2000
        pp = np.concatenate((pinhole-pinhole[:, :1], camera_relative[..., 2:] * ratio), axis=-1)
        world_to_pixel = np.sum(pp*camera_relative, axis=(1,2)) / np.sum(camera_relative**2, axis=(1,2))
        factor = 1/world_to_pixel
        gt25 = pp * factor[:,None,None]
        model25 = sequence.P_3d.astype(np.float64) * (sequence.metadata["image_size"][0]/2) * factor[:,None,None]
        candidates = [(label, sequence.P_3d, evaluation)]
        if "GT 2D" in label:
            xy = np.stack([normalize_h36m(p, tuple(sequence.metadata["image_size"])) for p in annotations.joints_pixel])
            passthrough = sequence.P_3d.copy()
            passthrough[..., :2] = xy - xy[:, :1]
            altered = replace(sequence, P_3d=passthrough, P_3d_raw=passthrough,
                              metadata=deepcopy(sequence.metadata))
            altered.metadata["audit_gt_xy_passthrough"] = True
            alternate_dir = destination.parent / (destination.name + "_xy_passthrough")
            alternate_dir.mkdir(exist_ok=True)
            alternate_eval = evaluate_sequence(altered, annotations)
            if (destination / "official_25d_labels.npz").is_file():
                alternate_eval = apply_official_25d(alternate_eval, altered, destination / "official_25d_labels.npz")
            save_sequence(alternate_dir, altered)
            save_evaluation(alternate_dir, alternate_eval)
            alternate_viewer = create_viewer(alternate_dir, altered, evaluation=alternate_eval,
                                            image_dir=image_dir, sequence_prefix=annotations.provenance["sequence"])
            candidates.append((label + " + official XY passthrough", passthrough, alternate_eval))
        for name, poses, current_eval in candidates:
            stats = current_eval[0]["variants"]["output"]
            pred25 = poses.astype(np.float64) * (sequence.metadata["image_size"][0]/2) * factor[:,None,None]
            rows.append(dict(case=name, frames=len(sequence.frame_ids),
                             n_mpjpe_camera_gt_mm=stats["n_mpjpe_mm"]["mean"],
                             pa_mpjpe_camera_gt_mm=stats["pa_mpjpe_mm"]["mean"],
                             mpjpe_camera_gt_root_depth_mm=stats["mpjpe_camera_root_depth_mm"]["mean"],
                             mpjpe_official_25d_mm=stats.get("mpjpe_official_25d_mm", {}).get("mean"),
                             reconstructed_25d_mpjpe_mm=float(np.linalg.norm(pred25-gt25,axis=-1).mean()),
                             reconstructed_25d_pa_mm=float(np.linalg.norm(align_procrustes(pred25,gt25)-gt25,axis=-1).mean()),
                             label_projection_error_px=current_eval[0]["camera_root_depth_calibration"]["mean_projection_error_px"],
                             source_frame_stride=int(np.median(np.diff(sequence.source_frame_ids))),
                             actual_fps=float(1/np.median(np.diff(sequence.timestamps))),
                             viewer=str(viewer if name == label else alternate_viewer)))
        print(label, rows[-len(candidates):], flush=True)
        np.savez_compressed(destination / "reconstructed_25d_evaluation.npz", target_25d_mm=gt25,
                            prediction_25d_mm=model25, reconstructed_factor=factor, source_frame_ids=sequence.source_frame_ids)
    report = dict(rows=rows, scope="one action each for S1 (training subject) and S9 (test subject); not a whole-test benchmark",
                  official_p1_available=any(row["mpjpe_official_25d_mm"] is not None for row in rows),
                  reason="official subset available for S9; original train split has no 2.5d_factor/joints_2.5d_image for S1",
                  reconstructed_protocol="LCN ICCV2019 image-depth + PAMI camera-to-image least-squares GT scale; independent reconstruction, not verified against original pickle")
    write_json(AUDIT / "evaluation_audit.json", report)
    write_csv(AUDIT / "evaluation_audit.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["structure", "scan", "evaluate"])
    args = parser.parse_args()
    {"structure": audit_structure, "scan": audit_scan, "evaluate": audit_evaluation}[args.mode]()
