"""Freeze the five held-out test conditions after the four extra scores finish."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from video_to_3d_motion.motion_generation.runtime import write_json


def main():
    root = ROOT / "results/motiongpt_stage1"
    original = root / "single_test_20261019_224114"
    extra = root / "baseline_20261019_230154/rep_00"
    rows = []
    protocols = []
    for directory, expected in [(original, 1), (extra, 4)]:
        scores = [json.loads(line) for line in (directory / "critic/scores.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = json.loads((directory / "critic_summary.json").read_text(encoding="utf-8"))
        if len(scores) != expected or summary["scored"] != expected or any(row["status"] != "scored" for row in scores):
            raise ValueError("Five complete scores required before stage two")
        metadata = {row["index"]: row for row in map(json.loads, (directory / "samples.jsonl").read_text(encoding="utf-8").splitlines())}
        for score in scores:
            sample = metadata[score["index"]]
            rows.append({**sample, "sample_path": str((directory / sample["file"]).resolve()),
                         "score_path": str((directory / "critic/scores.jsonl").resolve()),
                         "critic": score, "source_group": "independent_single" if expected == 1 else "baseline_320"})
        protocols.append({key: value for key, value in summary["protocol"].items() if key != "sample_count"})
    if protocols[0] != protocols[1] or len({row["motion_id"] for row in rows}) != 5:
        raise ValueError("Panel protocols or unique test identities do not agree")
    test_ids = set((ROOT / "datasets/HumanML3D/test.txt").read_text(encoding="utf-8").split())
    if any(row["motion_id"] not in test_ids for row in rows):
        raise ValueError("All five panel conditions must belong to the official test split")
    values = [row["critic"]["score"] for row in rows]
    report = {"status": "five_test_conditions_frozen", "samples": rows, "count": 5,
              "mean": float(np.mean(values)), "std": float(np.std(values)), "coverage": 1.,
              "protocol": protocols[0], "selection_seed": 1234,
              "scope": "One original independent sample plus four uniformly sampled baseline-320 records. Not the perception mean of all 320 motions.",
              "use": "Held-out comparison only. Never use these captions or their mirrored identities for reward optimization."}
    write_json(root / "critic_panel_five.json", report)
    print(json.dumps({"panel": str(root / "critic_panel_five.json"), "scores": values, "mean": report["mean"]}))


if __name__ == "__main__":
    main()
