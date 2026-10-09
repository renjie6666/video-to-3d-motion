from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_to_3d_motion.pose2d.experiment_reports import (
    SIMPLE_FIELDS, compact_row, read_csv, rebuild_reports, rebuild_simple_report, save_experiment, write_csv_atomic,
)
from video_to_3d_motion.pose2d.experiments import BEIJING, _new_run, run_experiment


class ExperimentReportsTests(unittest.TestCase):
    def test_common10_excludes_head_and_foot_and_keeps_low_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / "joints.csv"
            fields = ("gt_joint_index", "mapping", "gt_visible", "error_px")
            write_csv_atomic(table, fields, [
                dict(gt_joint_index=11, mapping="direct", gt_visible=True, error_px=2),
                dict(gt_joint_index=1, mapping="direct", gt_visible=True, error_px=8),
                dict(gt_joint_index=10, mapping="direct", gt_visible=True, error_px=90),
                dict(gt_joint_index=3, mapping="ankle_foot_proxy", gt_visible=True, error_px=70),
                dict(gt_joint_index=2, mapping="direct", gt_visible=False, error_px=""),
            ])
            result = compact_row(dict(experiment_id="a", active_model="model", status="completed",
                                      **{"comparison.status": "completed", "comparison.joint_comparison": str(table)}))
            self.assertEqual(result["common10_mean_error_px"], 5)

    def test_missing_or_failed_comparison_scores_are_blank(self):
        row = dict(experiment_id="a", active_model="model", status="failed",
                   **{"comparison.status": "completed", "comparison.all_visible_pairs.mean_px": 0})
        self.assertEqual(list(compact_row(row)), list(SIMPLE_FIELDS))
        self.assertTrue(all(value == "" for key, value in compact_row(row).items()
                            if key not in ("experiment_id", "active_model")))

    def test_upsert_preserves_experiments_and_rebuild_recovers_local_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = dict(experiment_id="a", active_model="m", status="running")
            b = dict(experiment_id="b", active_model="l", status="completed",
                     **{"comparison.status": "completed", "comparison.common10_pairs.mean_px": 0,
                        "comparison.confidence_coverage": 0})
            save_experiment(root, root / "a", a)
            save_experiment(root, root / "b", b)
            a["status"] = "failed"
            save_experiment(root, root / "a", a)
            fields, rows = read_csv(root / "experiment_summary_simple.csv")
            self.assertEqual(fields, list(SIMPLE_FIELDS))
            self.assertEqual(len(rows), 2)
            self.assertEqual(next(r for r in rows if r["experiment_id"] == "b")["common10_mean_error_px"], "0")
            (root / "experiment_summary.csv").unlink()
            rebuild_reports(root)
            _, recovered = read_csv(root / "experiment_summary.csv")
            self.assertEqual({r["experiment_id"]: r["status"] for r in recovered}, {"a": "failed", "b": "completed"})

    def test_busy_master_preserves_local_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = dict(experiment_id="a", active_model="m", status="failed")
            original_writer = write_csv_atomic
            def writer(path, fields, rows):
                if path.name == "experiment_summary.csv":
                    raise PermissionError("Excel has the table open")
                original_writer(path, fields, rows)
            with patch("video_to_3d_motion.pose2d.experiment_reports.write_csv_atomic", side_effect=writer):
                with self.assertRaises(PermissionError):
                    save_experiment(root, root / "a", row)
            self.assertEqual(read_csv(root / "a/a_summary.csv")[1][0]["status"], "failed")

    def test_refresh_simple_leaves_user_modified_detailed_csv_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = dict(experiment_id="a", active_model="m", status="completed",
                       **{"comparison.status": "completed", "comparison.confidence_coverage": "1",
                          "comparison.common10_pairs.mean_px": "3.5", "git_dirty": "TRUE"})
            master = root / "experiment_summary.csv"
            write_csv_atomic(master, list(row), [row])
            before = master.read_bytes()
            rebuild_simple_report(root)
            self.assertEqual(master.read_bytes(), before)
            self.assertEqual(read_csv(root / "experiment_summary_simple.csv")[1][0]["confidence_coverage"], "1")

    def test_same_second_uses_new_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            started = datetime(2026, 10, 8, 10, 5, 2, tzinfo=BEIJING)
            first = _new_run(root, "model", started)
            marker = first / "keep.txt"
            marker.write_text("previous experiment", encoding="utf-8")
            second = _new_run(root, "model", started)
            self.assertEqual(second.name, first.name + "_02")
            self.assertEqual(marker.read_text(encoding="utf-8"), "previous experiment")

    def test_runner_archives_completed_run_and_records_startup_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "pose.yaml"
            config.write_text(yaml.safe_dump(dict(pose2d=dict(active_model="model", models={
                "model": dict(config={"baseline": "model.py"}, checkpoint={"baseline": "weight.pth"})}))), encoding="utf-8")
            media = root / "media.yaml"
            media.write_text("media: {}\n", encoding="utf-8")
            args = argparse.Namespace(output_root=root / "results", pose2d_config=config,
                                      media_config=media, source=root / "sequence", annotations=root / "train.pkl",
                                      max_frames=100, comparison_every_n_frames=1, person_selection="highest_score")
            def child(command, **kwargs):
                summary_path = Path(command[command.index("--summary-output") + 1])
                comparison = summary_path.parent / "comparison"
                comparison.mkdir()
                (comparison / "overlays").mkdir()
                for name in ("joint_comparison.csv", "joint_summary.csv"):
                    (comparison / name).write_text("example\n", encoding="utf-8")
                (comparison / "summary.json").write_text("{}", encoding="utf-8")
                summary_path.write_text(json.dumps(dict(status="completed", comparison=dict(
                    status="completed", common10_pairs=dict(mean_px=3.5),
                    all_visible_pairs=dict(mean_px=3.14), confident_pairs=dict(mean_px=3.14),
                    confidence_coverage=1))), encoding="utf-8")
                return 0
            with patch("video_to_3d_motion.pose2d.experiments.subprocess.call", side_effect=child):
                self.assertEqual(run_experiment(args), 0)
            with patch("video_to_3d_motion.pose2d.experiments.subprocess.call", return_value=1):
                self.assertEqual(run_experiment(args), 1)
            _, rows = read_csv(args.output_root / "experiment_summary_simple.csv")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["common10_mean_error_px"], "3.5")
            self.assertEqual(rows[1]["mean_error_px"], "")
            snapshot = next(args.output_root.glob("*/*_effective_config.yaml"))
            effective = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
            self.assertTrue(Path(effective["pose2d"]["models"]["model"]["config"]["baseline"]).is_absolute())
            self.assertFalse(effective["detection"]["strict_single_person"])
            self.assertEqual(effective["experiment"]["max_frames"], 100)
            self.assertFalse(list(args.output_root.rglob("*.json")))


if __name__ == "__main__":
    unittest.main()
