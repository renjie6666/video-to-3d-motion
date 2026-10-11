"""S1-03/04/05 replay entry point; GT is opened only after inference."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import yaml

from video_to_3d_motion.pose_adapter.exceptions import PoseAdapterError
from video_to_3d_motion.pose_adapter.windows import prepare_sequence_windows
from video_to_3d_motion.stage_output.archive import save_evaluation, save_sequence, write_json
from video_to_3d_motion.stage_output.fusion import fuse_windows
from video_to_3d_motion.stage_output.viewer import create_viewer

from .backend import PoseMambaBackend, file_sha256
from .config import EvaluationSettings, load_pose3d_config
from .exceptions import Pose3DError


def build_parser():
    parser = argparse.ArgumentParser(description="Lift one completed 2D sequence with PoseMamba-L and export 3D poses.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "full"))
    parser.add_argument("--smoothing", choices=("none", "ema"))
    parser.add_argument("--tau-seconds", type=float)
    parser.add_argument("--device", choices=("cpu", "cuda:0"))
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--annotation-names", type=Path)
    parser.add_argument("--sequence")
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--image-prefix", help="source filename prefix; defaults to annotation sequence or video_id")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    directory = args.output_dir.resolve()
    created = False
    started = perf_counter()
    summary = dict(status="failed", input=str(args.input.resolve()), input_summary=str(args.input_summary.resolve()))
    try:
        config = load_pose3d_config(args.config)
        config = replace(config, pipeline_mode=args.mode or config.pipeline_mode,
                         model=replace(config.model, device=args.device or config.model.device,
                                       flip_test=False if args.no_flip else config.model.flip_test),
                         output=replace(config.output, smoothing=args.smoothing or config.output.smoothing,
                                        tau_seconds=args.tau_seconds if args.tau_seconds is not None else config.output.tau_seconds))
        if args.annotations is not None or args.annotation_names is not None or args.sequence is not None:
            config = replace(config, evaluation=EvaluationSettings(
                args.annotations.resolve() if args.annotations is not None else None,
                args.annotation_names.resolve() if args.annotation_names is not None else None,
                args.sequence,
            ))
        directory.mkdir(parents=True, exist_ok=False)
        created = True
        snapshot = json.loads(json.dumps(asdict(config), default=str))
        (directory / "config_snapshot.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
        windows, adapter_metrics = prepare_sequence_windows(args.input, args.input_summary,
                                                           mode=config.pipeline_mode, stride=config.window_stride)
        summary.update(pipeline_mode=config.pipeline_mode, pose_adapter_metrics=asdict(adapter_metrics),
                       input_sha256=file_sha256(args.input), input_summary_sha256=file_sha256(args.input_summary))
        backend = PoseMambaBackend(config.model)
        lifted = []
        (directory / "windows").mkdir()
        for index, window in enumerate(windows, 1):
            prediction = backend.predict(window)
            lifted.append(prediction)
            np.savez_compressed(directory / "windows" / f"window_{index:06d}.npz", pose_3d=prediction.pose_3d,
                                frame_ids=window.frame_ids, source_frame_ids=window.source_frame_ids,
                                timestamps=window.timestamps, source_timestamps=window.source_timestamps)
            print(f"Pose3D: window {index}/{len(windows)}, samples {window.frame_ids[0]}-{window.frame_ids[-1]}, {prediction.inference_seconds:.2f}s", flush=True)
        expected = windows[0].frame_ids if config.pipeline_mode == "baseline" else np.arange(1, adapter_metrics.frames_received + 1)
        sequence = fuse_windows(lifted, expected, config.output)
        evaluation = None
        if config.evaluation.annotation_h5 is not None:
            from video_to_3d_motion.stage_output.annotations import load_annotations
            from video_to_3d_motion.stage_output.metrics import evaluate_sequence
            annotations = load_annotations(config.evaluation.annotation_h5, config.evaluation.annotation_names,
                                           config.evaluation.sequence, sequence.source_frame_ids)
            evaluation = evaluate_sequence(sequence, annotations)
            save_evaluation(directory, evaluation)
        save_sequence(directory, sequence)
        viewer = create_viewer(directory, sequence, evaluation=evaluation, image_dir=args.image_dir,
                               sequence_prefix=args.image_prefix or config.evaluation.sequence or sequence.metadata["video_id"])
        summary.update(status="completed", frame_count=len(sequence.frame_ids), window_count=len(lifted),
                       model_info=backend.runtime_info(), inference_seconds=sum(w.inference_seconds for w in lifted),
                       total_seconds=perf_counter() - started,
                       first_frame_id=int(sequence.frame_ids[0]), last_frame_id=int(sequence.frame_ids[-1]),
                       P_3d_shape=list(sequence.P_3d.shape), P_obs_shape=[10, 17, 3],
                       sequence_output=str(directory / "sequence_3d.npz"), viewer=str(viewer),
                       image_dir=str(args.image_dir.resolve()) if args.image_dir else None,
                       evaluation_status="completed" if evaluation else "not_requested")
        write_json(directory / "run_summary.json", summary)
        print(f"COMPLETED: {directory / 'run_summary.json'}\nVIEWER: {viewer}")
        if evaluation:
            metrics = evaluation[0]["variants"]["output"]
            print(f"N-MPJPE: {metrics['n_mpjpe_mm']['mean']:.2f} mm; PA-MPJPE: {metrics['pa_mpjpe_mm']['mean']:.2f} mm; GT-scale MPJPE: {metrics['mpjpe_gt_scale_mm']['mean']:.2f} mm (GT-aided diagnostic)")
        return 0
    except (Pose3DError, PoseAdapterError, OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        summary.update(error_code=getattr(exc, "code", "configuration_or_io_error"), message=str(exc), total_seconds=perf_counter() - started)
        if created:
            write_json(directory / "run_summary.json", summary)
        print(f"FAILED: {summary['error_code']}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
