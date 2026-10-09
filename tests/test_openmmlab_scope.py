"""Regression for actual transform lookup after MMPose changes default scope."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_to_3d_motion.pose2d.backends.mmdet_backend import MMDetPersonDetector


class OpenMMLabScopeTests(unittest.TestCase):
    def test_detector_works_after_pose_scope_and_restores_caller_scope(self):
        try:
            import torch
            from mmengine.config import Config
            from mmengine.registry import DefaultScope
            from mmdet.utils import register_all_modules as register_detection
            from mmpose.utils import register_all_modules as register_pose
        except ImportError:
            self.skipTest("optional OpenMMLab dependencies are not installed")
        register_detection(init_default_scope=False)
        register_pose(init_default_scope=False)
        detector = MMDetPersonDetector.__new__(MMDetPersonDetector)
        detector.person_class_id = 0
        # Real inference_detector/Compose/PackDetInputs with a tiny model seam:
        # no weights or GPU operations are needed to catch registry pollution.
        detector.model = SimpleNamespace(
            cfg=Config(dict(test_dataloader=dict(dataset=dict(pipeline=[
                dict(type="LoadImageFromNDArray"), dict(type="PackDetInputs")
            ])))),
            data_preprocessor=SimpleNamespace(device=torch.device("cuda")),
            test_step=lambda data: [SimpleNamespace(pred_instances=SimpleNamespace(
                bboxes=np.array([[0, 0, 8, 8]], dtype=np.float32),
                scores=np.array([0.9]), labels=np.array([0])
            ))],
        )
        with DefaultScope.overwrite_default_scope("mmpose"):
            for _ in range(2):
                output = detector.detect_batch([np.zeros((8, 8, 3), dtype=np.uint8)])
                self.assertEqual(len(output[0]), 1)
                np.testing.assert_array_equal(output[0][0].bbox, [0, 0, 8, 8])
                self.assertEqual(DefaultScope.get_current_instance().scope_name, "mmpose")


if __name__ == "__main__":
    unittest.main()
