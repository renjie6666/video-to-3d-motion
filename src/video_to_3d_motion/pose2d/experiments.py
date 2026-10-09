"""Archive one image-sequence experiment and update detailed/compact reports."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from .config import load_pose2d_config
from .experiment_reports import flatten_summary, rebuild_reports, rebuild_simple_report, save_experiment

BEIJING = timezone(timedelta(hours=8))


def _sha256(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _new_run(root: Path, model: str, started: datetime) -> Path:
    if not model or any(char in model for char in '/\\:') or model in {'.', '..'}:
        raise ValueError("active_model must be a safe experiment filename")
    root.mkdir(parents=True, exist_ok=True)
    base = model + "_" + started.strftime("%Y-%m-%d-%H-%M-%S")
    suffix = 1
    while True:
        run = root / (base if suffix == 1 else f"{base}_{suffix:02d}")
        try:
            run.mkdir()
            return run
        except FileExistsError:
            suffix += 1


def run_experiment(args) -> int:
    started, tick = datetime.now(BEIJING), time.monotonic()
    root = args.output_root.resolve()
    original = args.pose2d_config.resolve()
    raw = yaml.safe_load(original.read_text(encoding="utf-8"))
    model = raw.get("pose2d", {}).get("active_model", "rtmpose_m")
    run = _new_run(root, model, started)
    experiment_id = run.name
    snapshot = run / (experiment_id + "_effective_config.yaml")
    media_snapshot = run / (experiment_id + "_media_config.yaml")
    frames = run / (experiment_id + "_pose2d_frames.jsonl")
    internal_summary = run / ".inference_summary.json"
    log_path = run / (experiment_id + "_run.log")
    row = dict(experiment_id=experiment_id, active_model=model,
               started_at=started.isoformat(timespec="seconds"), finished_at="",
               elapsed_seconds="", status="running", summary_schema_version="s1-02-experiment-v2",
               design_doc_version="V3.16", source=str(args.source.resolve()), source_kind="images",
               annotations=str(args.annotations.resolve()), max_frames=args.max_frames or "",
               media_config=str(args.media_config.resolve()), media_config_snapshot=str(media_snapshot),
               pose2d_config=str(original), config_snapshot=str(snapshot), frames_output=str(frames),
               comparison_every_n_frames=args.comparison_every_n_frames)
    save_experiment(root, run, row)
    exit_code = 1
    try:
        def absolute(value):
            path = Path(value)
            return str(path if path.is_absolute() else (original.parent / path).resolve())

        for selected in raw.get("pose2d", {}).get("models", {}).values():
            for key in ("config", "checkpoint"):
                value = selected.get(key)
                if isinstance(value, dict):
                    selected[key] = {mode: absolute(item) for mode, item in value.items() if item}
                elif value:
                    selected[key] = absolute(value)
        detection = raw.setdefault("detection", {})
        for key in ("detector_config", "detector_checkpoint"):
            if detection.get(key):
                detection[key] = absolute(detection[key])
        detection["strict_single_person"] = args.person_selection == "strict"
        raw.setdefault("visualization", {})["output_root"] = str(run / "pose_overlays")
        raw["experiment"] = dict(experiment_id=experiment_id, max_frames=args.max_frames,
                                 source=row["source"], annotations=row["annotations"],
                                 person_selection=args.person_selection,
                                 comparison_every_n_frames=args.comparison_every_n_frames)
        snapshot.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
        media_snapshot.write_bytes(args.media_config.read_bytes())
        config = load_pose2d_config(snapshot)
        row.update(pipeline_mode=config.pipeline_mode, joint_schema=config.model.joint_schema,
                   model_config=str(config.model.config_path), checkpoint=str(config.model.checkpoint_path),
                   checkpoint_sha256=_sha256(config.model.checkpoint_path),
                   detector_checkpoint_sha256=_sha256(config.detection.detector_checkpoint),
                   keypoint_score_threshold=config.runtime.keypoint_score_threshold)
        row["dependency_versions"] = "; ".join(
            name + "=" + importlib.metadata.version(name)
            for name in ("av", "numpy", "PyYAML"))
        command = [sys.executable, "-m", "video_to_3d_motion.pose2d.cli", row["source"],
                   "--media-config", str(media_snapshot), "--pose2d-config", str(snapshot),
                   "--frames-output", str(frames), "--summary-output", str(internal_summary),
                   "--annotations", row["annotations"], "--comparison-output", str(run / "comparison"),
                   "--person-selection", args.person_selection,
                   "--comparison-every-n-frames", str(args.comparison_every_n_frames)]
        if args.max_frames is not None:
            command.extend(("--max-frames", str(args.max_frames)))
        print(f"Experiment: {experiment_id}\nLog: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            exit_code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        if internal_summary.exists():
            summary = json.loads(internal_summary.read_text(encoding="utf-8"))
            row.update(flatten_summary(summary))
            row["inference_status"] = summary.get("inference_status", summary["status"])
            comparison = summary.get("comparison")
            if comparison and comparison.get("status") == "completed":
                for name in ("joint_comparison", "joint_summary"):
                    destination = run / (experiment_id + "_" + name + ".csv")
                    (run / "comparison" / (name + ".csv")).rename(destination)
                    row["comparison." + name] = str(destination)
                old = (run / "comparison/overlays").resolve()
                new = (run / "comparison_overlays" / args.source.resolve().name).resolve()
                if not old.is_relative_to(run) or not new.is_relative_to(run) or old.is_symlink():
                    raise ValueError("comparison overlay path outside experiment")
                new.parent.mkdir()
                old.rename(new)
                row["comparison.overlays"] = str(new)
                (run / "comparison/summary.json").unlink()
                (run / "comparison").rmdir()
            internal_summary.unlink()
        else:
            row.update(status="failed", error_code="startup_failed", message=f"See {log_path}")
        if exit_code != 0 and row["status"] == "completed":
            row.update(status="failed", error_code="process_failed", message=f"Exit code {exit_code}")
        if row["status"] != "completed" and exit_code == 0:
            exit_code = 1
    except KeyboardInterrupt:
        row.update(status="cancelled", error_code="interrupted", message="Experiment interrupted")
        exit_code = 130
    except Exception as exc:
        row.update(status="failed", error_code="experiment_failed", message=str(exc))
        exit_code = 1
    finally:
        row["finished_at"] = datetime.now(BEIJING).isoformat(timespec="seconds")
        row["elapsed_seconds"] = round(time.monotonic() - tick, 3)
        save_experiment(root, run, row)
    print(f"{row['status']}: {run}\nCompact report: {root / 'experiment_summary_simple.csv'}")
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or rebuild archived S1-02 image experiments")
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--media-config", type=Path)
    parser.add_argument("--pose2d-config", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--person-selection", choices=("strict", "highest_score"), default="highest_score")
    parser.add_argument("--comparison-every-n-frames", type=int, default=1)
    rebuild = parser.add_mutually_exclusive_group()
    rebuild.add_argument("--rebuild-reports", action="store_true")
    rebuild.add_argument("--rebuild-simple", action="store_true")
    args = parser.parse_args()
    if args.rebuild_reports:
        rebuild_reports(args.output_root.resolve())
        print(args.output_root.resolve() / "experiment_summary_simple.csv")
        return 0
    if args.rebuild_simple:
        rebuild_simple_report(args.output_root.resolve())
        print(args.output_root.resolve() / "experiment_summary_simple.csv")
        return 0
    for key in ("source", "media_config", "pose2d_config", "annotations"):
        if getattr(args, key) is None:
            parser.error(f"{key.replace('_', '-')} is required for inference")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.comparison_every_n_frames <= 0:
        parser.error("--comparison-every-n-frames must be positive")
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
