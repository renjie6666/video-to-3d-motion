"""Commands for phase one only: frozen generation, evaluation, and scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from .runtime import Generator, find_asset, load_config, load_file, norms, provenance, write_json


def extract_assets(cfg: dict) -> None:
    """Extract official archives without accepting links or path traversal."""
    root = Path(cfg["paths"]["assets"]).resolve()
    for filename in ("t2m.tar.gz", "smpl.tar.gz"):
        archive = root / filename
        if not archive.exists():
            print(f"NOT DOWNLOADED {archive}")
            continue
        target = root / ("evaluator" if filename.startswith("t2m") else "smpl")
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as stream:
            for member in stream:
                path = (target / member.name).resolve()
                if not path.is_relative_to(target.resolve()) or member.issym() or member.islnk():
                    raise ValueError(f"Unsafe archive member: {member.name}")
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists() and path.stat().st_size == member.size:
                        continue
                    source = stream.extractfile(member)
                    if source is None:
                        raise ValueError(member.name)
                    with source, path.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
        print(f"EXTRACTED {archive}")


def readiness(cfg: dict) -> dict:
    assets = Path(cfg["paths"]["assets"])
    report = {"phase": 1, "complete": False, "missing": [], "ready": []}
    for suffix in ("motiongpt_s3_h3d.tar", "motioncritic_pre.pth", "flan-t5-base/config.json",
                   "flan-t5-base/tokenizer_config.json", "t2m/text_mot_match/model/finest.tar",
                   "our_vab_data.npy", "our_vab_words.pkl", "our_vab_idx.pkl",
                   "SMPL_NEUTRAL.pkl", "neutral_smpl_mean_params.h5", "gmm_08.pkl"):
        try:
            found = find_asset(assets, "evaluator/t2m/glove/" + suffix if suffix.startswith("our_vab_") else suffix)
            if not found.stat().st_size:
                raise FileNotFoundError(str(found))
            report["ready"].append(str(found))
        except FileNotFoundError:
            report["missing"].append(suffix)
    try:
        norms(cfg)
        report["normalization"] = "ready"
    except (FileNotFoundError, ValueError) as exc:
        report["missing"].append(str(exc))
    dataset = Path(cfg["paths"]["dataset"])
    dataset_manifest = dataset / "source_manifest.json"
    if not dataset_manifest.exists():
        report["missing"].append(str(dataset_manifest))
    else:
        data_source = json.loads(dataset_manifest.read_text(encoding="utf-8"))
        if data_source.get("status") != "validated_public_mirror":
            report["missing"].append("HumanML3D source validation is incomplete")
        report["data_source"] = data_source
    split = dataset / f"{cfg['evaluation']['split']}.txt"
    if not split.exists():
        report["missing"].append(str(split))
    else:
        ids = [line.strip() for line in split.read_text(encoding="utf-8").splitlines() if line.strip()]
        missing_files = []
        for identity in ids:
            if Path(identity).name != identity or identity in (".", ".."):
                raise ValueError("Invalid sample ID in split")
            for relative in (f"new_joint_vecs/{identity}.npy", f"texts/{identity}.txt"):
                if not (dataset / relative).is_file():
                    missing_files.append(relative)
        report["dataset"] = {"path": str(dataset), "split_ids": len(ids),
                             "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
                             "missing_files_count": len(missing_files), "missing_files_first20": missing_files[:20]}
        if not ids or missing_files:
            report["missing"].append("HumanML3D split data incomplete")
    report["environment"] = provenance(cfg)
    report["assets_and_data_ready"] = not report["missing"]
    report["note"] = "Readiness is not an evaluated result. Phase one requires baseline_metrics.json AND critic_summary.json from the same generated sample bank."
    return report


def score_manifest(cfg: dict, manifest: Path, max_samples: int | None = None, resume: bool = False) -> dict:
    from .critic import Critic, SmplFitter
    critic = Critic(cfg)
    fitter = SmplFitter(cfg)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_samples:
        positions = np.sort(np.random.default_rng(cfg["seed"]).choice(
            len(rows), min(max_samples, len(rows)), replace=False))
        selected = [rows[int(position)] for position in positions]
    else:
        selected = rows
    output = manifest.parent / "critic"
    selection = {"seed": cfg["seed"], "method": "seeded_random_without_replacement" if max_samples else "all",
                 "source_manifest": str(manifest.resolve()),
                 "source_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 "protocol": cfg["critic"], "samples": selected}
    previous = []
    if resume:
        saved_selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
        if saved_selection != selection:
            raise ValueError("Resume requires the same manifest, sample selection and fitting protocol")
        scores_file = output / "scores.jsonl"
        if scores_file.exists():
            previous = [json.loads(line) for line in scores_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if [row["index"] for row in previous] != [row["index"] for row in selected[:len(previous)]]:
            raise ValueError("Completed score records do not match the selected sample prefix")
    else:
        output.mkdir(exist_ok=False)
        write_json(output / "selection.json", selection)
    successful, failures, scores = 0, [], []
    with (output / "scores.jsonl").open("a" if resume else "w", encoding="utf-8") as handle:
        for index, row in enumerate(selected):
            if index < len(previous):
                record = previous[index]
                if record.get("score") is not None:
                    successful += 1
                    scores.append(record["score"])
                if record["status"] == "failed":
                    failures.append({"index": record["index"], "error": record["error"]})
                continue
            identity = int(row["index"])
            started = time.monotonic()
            record = {"index": identity, "motion_id": row.get("motion_id"),
                      "caption": row["caption"], "file": row["file"]}
            try:
                if not row["valid_generation"]:
                    raise ValueError("Invalid generation is not eligible for critic scoring")
                sample_path = (manifest.parent / row["file"]).resolve()
                if not sample_path.is_relative_to(manifest.parent.resolve()):
                    raise ValueError("Sample path leaves manifest directory")
                with np.load(sample_path, allow_pickle=False) as sample:
                    fitted = fitter.fit(sample["joints"])
                np.savez_compressed(output / f"{identity:06d}_smpl.npz", **fitted)
                record.update(critic.score_smpl(fitted["pose"], fitted["translation"]))
                record["fit_rmse_mean_m"] = float(fitted["fit_rmse_m"].mean())
                record["fit_rmse_max_m"] = float(fitted["fit_rmse_m"].max())
                if record["score"] is not None:
                    successful += 1
                    scores.append(record["score"])
            except Exception as exc:
                record.update(status="failed", score=None, error=f"{type(exc).__name__}: {exc}")
                failures.append({"index": identity, "error": record["error"]})
            record["elapsed_seconds"] = time.monotonic() - started
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            print(f"MotionCritic {index + 1}/{len(selected)}: {record['status']}", flush=True)
    summary = {"status": "scored" if successful == len(selected) and selected else "partial_or_failed",
               "manifest": str(manifest.resolve()), "subset": max_samples is not None,
               "requested": len(selected), "scored": successful, "coverage": successful / len(selected) if selected else 0,
               "failed": failures, "too_short": len(selected) - successful - len(failures),
               "motioncritic_mean": float(np.mean(scores)) if scores else None,
               "motioncritic_std": float(np.std(scores)) if scores else None,
               "selection": str((output / "selection.json").resolve()),
               "protocol": cfg["critic"], "environment": provenance(cfg),
               "interpretation": "Frozen learned perceptual score, not a calibrated human approval probability. Mean weights motions equally after averaging their windows."}
    write_json(manifest.parent / "critic_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    commands.add_parser("extract")
    example = commands.add_parser("score-example")
    example.add_argument("--file", type=Path)
    demo = commands.add_parser("demo")
    demo.add_argument("--captions", type=Path, required=True)
    demo.add_argument("--adapter", type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--max-samples", type=int)
    evaluate.add_argument("--repetitions", type=int)
    evaluate.add_argument("--adapter", type=Path)
    score = commands.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--max-samples", type=int)
    score.add_argument("--resume", action="store_true", help="Continue the same saved selection without repeating finished fits")
    args = parser.parse_args()
    if getattr(args, "max_samples", None) is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if getattr(args, "repetitions", None) is not None and args.repetitions < 1:
        parser.error("--repetitions must be positive")
    cfg = load_config(args.config)
    if getattr(args, "adapter", None) is not None:
        cfg["adapter_checkpoint"] = str(args.adapter.resolve(strict=True))
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    output = Path(cfg["paths"]["output"])
    if args.command == "extract":
        extract_assets(cfg)
    elif args.command == "check":
        report = readiness(cfg)
        write_json(output / "readiness.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["assets_and_data_ready"] else 2
    elif args.command == "demo":
        from .evaluation import seed_everything
        seed_everything(cfg["seed"])
        version = provenance(cfg)
        model = Generator(cfg)
        captions = [line.strip() for line in args.captions.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not captions:
            raise ValueError("No captions")
        target = output / f"demo_{timestamp}"
        target.mkdir(parents=True, exist_ok=False)
        with (target / "samples.jsonl").open("w", encoding="utf-8") as manifest:
            for index, caption in enumerate(captions):
                record = model.generate([caption])[0]
                filename = f"{index:06d}.npz"
                np.savez_compressed(target / filename, **record)
                row = {"index": index, "caption": caption, "file": filename, "frames": len(record["features"]),
                       "fps": 20, "valid_generation": record["valid_generation"]}
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(row, flush=True)
        write_json(target / "demo.json", {"status": "smoke_only", "humanml3d_benchmark": False,
                                         "environment": version, "config": cfg})
    elif args.command == "evaluate":
        from .evaluation import evaluate
        report = readiness(cfg)
        # Baseline evaluation can run before SMPL/critic assets arrive.
        missing = [item for item in report["missing"] if not any(name in item for name in
                   ("motioncritic_pre.pth", "SMPL_NEUTRAL.pkl", "neutral_smpl_mean_params.h5", "gmm_08.pkl"))]
        if missing:
            write_json(output / "readiness.json", report)
            raise FileNotFoundError("Baseline prerequisites missing: " + "; ".join(missing))
        destination = output / f"baseline_{timestamp}"
        result = evaluate(cfg, destination, args.max_samples, args.repetitions)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "score":
        limit = args.max_samples if args.max_samples is not None else cfg["critic"].get("sample_count")
        print(json.dumps(score_manifest(cfg, args.manifest, limit, args.resume), ensure_ascii=False, indent=2))
    elif args.command == "score-example":
        from .critic import Critic
        critic = Critic(cfg)
        example_path = args.file or Path(cfg["paths"]["assets"]) / "visexample.pth"
        raw = torch.load(example_path, map_location="cpu")
        source = Path(cfg["paths"]["motioncritic_source"]) / "MotionCritic"
        into_critic = load_file("motioncritic_parsedata", source / "parsedata.py").into_critic
        motion = into_critic(raw["motion"]).cpu().numpy()
        scores = critic.score_input(motion)
        with torch.inference_mode():
            cpu = critic.model.cpu().batch_critic(torch.from_numpy(motion)).flatten().tolist()
        np.testing.assert_allclose(scores, cpu, atol=2e-4, rtol=2e-4)
        result = {"status": "official_example_scored", "file": str(example_path.resolve()),
                  "shape": list(motion.shape), "scores": scores, "cpu_scores": cpu,
                  "cpu_cuda_parity": "passed", "humanml3d_benchmark": False,
                  "environment": provenance(cfg)}
        write_json(output / "critic_example.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
