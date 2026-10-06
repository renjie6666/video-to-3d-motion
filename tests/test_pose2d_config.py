from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from video_to_3d_motion.pose2d.config import load_pose2d_config
from video_to_3d_motion.pose2d.exceptions import Pose2DError
from video_to_3d_motion.pose2d.factory import check_gpu_environment


class Pose2DConfigTests(unittest.TestCase):
    def test_project_config_is_gpu_only_rtmpose(self) -> None:
        config = load_pose2d_config(PROJECT_ROOT / "configs" / "pose2d.yaml")
        self.assertEqual(config.pipeline_mode, "baseline")
        self.assertEqual(config.model.name, "rtmpose_m")
        self.assertEqual(config.model.joint_schema, "coco17")
        self.assertEqual(config.model.score_source, "rtmpose_simcc")
        self.assertNotIn("device", config.runtime.__dataclass_fields__)
        self.assertNotIn("backend", config.runtime.__dataclass_fields__)

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

    def test_cuda_unavailable_fails_without_cpu_fallback(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(Pose2DError) as raised:
                check_gpu_environment()
        self.assertEqual(raised.exception.code, "cuda_unavailable")


if __name__ == "__main__":
    unittest.main()
