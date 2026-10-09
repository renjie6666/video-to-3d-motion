from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from video_to_3d_motion.pose2d.config import ModelConfig, Pose2DConfig, load_pose2d_config
from video_to_3d_motion.pose2d.exceptions import Pose2DError
from video_to_3d_motion.pose2d.factory import check_gpu_environment


class Pose2DConfigTests(unittest.TestCase):
    def test_project_config_is_gpu_only_rtmpose(self) -> None:
        config = load_pose2d_config(PROJECT_ROOT / "configs" / "pose2d.yaml")
        self.assertEqual(config.pipeline_mode, "baseline")
        self.assertEqual(config.model.name, "rtmpose_l_h36m")
        self.assertEqual(config.model.joint_schema, "h36m17")
        self.assertEqual(config.model.checkpoint_path, PROJECT_ROOT / "checkpoints/rtmpose_h36m.pth")
        self.assertTrue(config.model.config_path.is_file())
        self.assertEqual(config.model.score_source, "rtmpose_simcc")
        self.assertNotIn("device", config.runtime.__dataclass_fields__)
        self.assertNotIn("backend", config.runtime.__dataclass_fields__)

    def test_baseline_can_still_use_coco_schema(self) -> None:
        import yaml

        raw = yaml.safe_load((PROJECT_ROOT / "configs/pose2d.yaml").read_text(encoding="utf-8"))
        raw["pose2d"]["active_model"] = "rtmpose_l"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            config = load_pose2d_config(path)
        self.assertEqual(config.pipeline_mode, "baseline")
        self.assertEqual(config.model.joint_schema, "coco17")

    def test_unknown_joint_schema_is_rejected_in_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "joint_schema must be"):
            Pose2DConfig(model=ModelConfig(joint_schema="unknown"))

    def test_full_mode_rejects_coco_schema(self) -> None:
        text = """
pipeline_mode: full
pose2d:
  active_model: rtmpose_m
  models:
    rtmpose_m:
      joint_schema: coco17
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_pose2d_config(path)

    def test_relative_paths_are_anchored_to_yaml_directory(self) -> None:
        text = """
pose2d:
  models:
    rtmpose_m:
      config:
        baseline: models/pose.py
      checkpoint:
        baseline: ../checkpoints/pose.pth
detection:
  detector_config: models/detector.py
  detector_checkpoint: ../checkpoints/detector.pth
visualization:
  output_root: ../results/overlays
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            config_dir = root / "configs"
            config_dir.mkdir()
            path = config_dir / "pose2d.yaml"
            path.write_text(text, encoding="utf-8")
            with patch("os.getcwd", return_value=str(root / "unrelated")):
                config = load_pose2d_config(path)
            self.assertEqual(config.model.config_path, config_dir / "models/pose.py")
            self.assertEqual(config.model.checkpoint_path, root / "checkpoints/pose.pth")
            self.assertEqual(config.detection.detector_config, config_dir / "models/detector.py")
            self.assertEqual(config.detection.detector_checkpoint, root / "checkpoints/detector.pth")
            self.assertEqual(config.visualization.output_root, root / "results/overlays")

    def test_absolute_paths_are_preserved(self) -> None:
        import yaml

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            pose_config = root / "pose.py"
            weights = root / "pose.pth"
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump({
                "pose2d": {"models": {"rtmpose_m": {
                    "config": str(pose_config), "checkpoint": str(weights),
                }}},
                "detection": {"detector_config": str(pose_config), "detector_checkpoint": str(weights)},
                "visualization": {"output_root": str(root / "overlays")},
            }), encoding="utf-8")
            config = load_pose2d_config(path)
            self.assertEqual(config.model.config_path, pose_config)
            self.assertEqual(config.model.checkpoint_path, weights)
            self.assertEqual(config.detection.detector_config, pose_config)
            self.assertEqual(config.detection.detector_checkpoint, weights)
            self.assertEqual(config.visualization.output_root, root / "overlays")

    def test_cuda_unavailable_fails_without_cpu_fallback(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(Pose2DError) as raised:
                check_gpu_environment()
        self.assertEqual(raised.exception.code, "cuda_unavailable")


if __name__ == "__main__":
    unittest.main()
