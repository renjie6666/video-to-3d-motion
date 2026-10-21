"""Build a standalone offline metrics and HumanML3D skeleton playback page."""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from video_to_3d_motion.motion_generation.runtime import load_config, official_modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/motiongpt_stage1/baseline_20261019_230154")
    parser.add_argument("--single", type=Path, default=ROOT / "results/motiongpt_stage1/single_test_20261019_224114")
    parser.add_argument("--output", type=Path, default=ROOT / "results/motiongpt_stage1/experiment_viewer.html")
    parser.add_argument("--panel", type=Path, default=ROOT / "results/motiongpt_stage1/critic_panel_five.json")
    parser.add_argument("--pilot", type=Path, default=ROOT / "results/motiongpt_stage2/pilot")
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs/motiongpt_stage1/config.yaml")
    motion = official_modules(cfg)
    torch.set_num_threads(1)
    baseline = json.loads((args.baseline / "baseline_metrics.json").read_text(encoding="utf-8"))
    single = json.loads((args.single / "single_test_result.json").read_text(encoding="utf-8"))
    critic = json.loads((args.single / "critic/scores.jsonl").read_text(encoding="utf-8").strip())
    panel = json.loads(args.panel.read_text(encoding="utf-8")) if args.panel.exists() else None
    scored = {str(Path(row["sample_path"]).resolve()): row["critic"] for row in panel["samples"]} if panel else {}
    if not panel:
        scored[str((args.single / "000000.npz").resolve())] = critic
    samples = []
    for directory, group in [(args.single, "critic_single"), (args.baseline / "rep_00", "baseline")]:
        for line in (directory / "samples.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample_path = (directory / row["file"]).resolve()
            if not sample_path.is_relative_to(directory.resolve()):
                raise ValueError("Motion file leaves its sample directory")
            with np.load(sample_path, allow_pickle=False) as arrays:
                generated = arrays["joints"]
                reference = motion.recover_from_ric(torch.from_numpy(arrays["reference_features"].astype(np.float32)), 22).numpy()
            for values in (generated, reference):
                if values.ndim != 3 or values.shape[1:] != (22, 3) or not len(values) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid skeleton in {sample_path}")
            samples.append({**row, "group": group, "critic": scored.get(str(sample_path)),
                            "generated": generated.astype(np.float64).round(4).tolist(),
                            "reference": reference.astype(np.float64).round(4).tolist()})
    if len([row for row in samples if row["group"] == "baseline"]) != baseline["samples"]:
        raise ValueError("Sample bank size differs from reported metrics")
    pilot_path = args.pilot / "pilot_result.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8")) if pilot_path.exists() else None
    if pilot and panel:
        references = {row["motion_id"]: row for row in samples}
        for preview in json.loads((args.pilot / "heldout_previews.json").read_text(encoding="utf-8")):
            with np.load(args.pilot / preview["file"], allow_pickle=False) as arrays:
                generated = arrays["joints"].astype(np.float64).round(4).tolist()
            if not np.isfinite(np.asarray(generated)).all():
                raise ValueError("Non-finite pilot preview")
            samples.append({**preview, "group": "pilot_" + preview["branch"], "critic": None,
                            "generated": generated, "reference": references[preview["motion_id"]]["reference"]})
    payload = {"baseline": baseline, "critic": critic, "single": single["sample"], "samples": samples,
               "panel": panel, "pilot": pilot,
               "sources": {"baseline": str(args.baseline.resolve()), "single": str(args.single.resolve())},
               "chains": [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
                          [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]}
    encoded = base64.b64encode(gzip.compress(json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                                     separators=(",", ":")).encode("utf-8"), mtime=0)).decode("ascii")
    template = ROOT / "src/video_to_3d_motion/motion_generation/report.html"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.read_text(encoding="utf-8").replace("__REPORT_GZIP_BASE64__", encoded), encoding="utf-8")
    print(json.dumps({"html": str(args.output.resolve()), "baseline_samples": baseline["samples"],
                      "viewable_sequences": len(samples), "bytes": args.output.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
