"""Run one canonical HumanML3D test caption through frozen MotionGPT and MotionCritic."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_to_3d_motion.motion_generation.cli import score_manifest
from video_to_3d_motion.motion_generation.evaluation import seed_everything
from video_to_3d_motion.motion_generation.runtime import Generator, load_config, provenance, write_json


def main():
    cfg = load_config(ROOT / "configs/motiongpt_stage1/config.yaml")
    dataset = Path(cfg["paths"]["dataset"])
    identity = (dataset / "test.txt").read_text(encoding="utf-8").splitlines()[0]
    caption = (dataset / "texts" / f"{identity}.txt").read_text(encoding="utf-8").splitlines()[0].split("#")[0]
    reference = np.load(dataset / "new_joint_vecs" / f"{identity}.npy", allow_pickle=False)
    output = Path(cfg["paths"]["output"]) / datetime.now().strftime("single_test_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", cfg)
    write_json(output / "provenance.json", provenance(cfg))
    seed_everything(cfg["seed"])
    generator = Generator(cfg)
    generated = generator.generate([caption], [len(reference)])[0]
    np.savez_compressed(output / "000000.npz", **generated, reference_features=reference)
    row = {"index": 0, "motion_id": identity, "caption": caption, "file": "000000.npz",
           "frames": len(generated["features"]), "reference_frames": len(reference),
           "fps": 20, "valid_generation": generated["valid_generation"]}
    manifest = output / "samples.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    print(json.dumps(row), flush=True)
    del generator
    gc.collect()
    torch.cuda.empty_cache()
    score = score_manifest(cfg, manifest, max_samples=1)
    report = {"status": "single_test_link_verified" if score["scored"] == 1 else "single_test_incomplete",
              "dataset": "HumanML3D", "split": "test", "samples": 1, "sample": row,
              "motioncritic": score, "humanml3d_benchmark": False,
              "note": "One test-caption generation and perceptual score only; no FID, Diversity or standard 32-candidate R-Precision reported."}
    write_json(output / "single_test_result.json", report)
    print(json.dumps({"result": str(output / "single_test_result.json"), "status": report["status"],
                      "motioncritic_mean": score["motioncritic_mean"]}), flush=True)


if __name__ == "__main__":
    main()
