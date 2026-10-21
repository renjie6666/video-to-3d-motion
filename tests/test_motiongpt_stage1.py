"""Test temporal correctness and metric semantics, without model downloads."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from video_to_3d_motion.motion_generation.critic import Critic, resample_smpl, window_starts
from video_to_3d_motion.motion_generation.runtime import find_asset, load_file
from video_to_3d_motion.motion_generation.evaluation import Evaluator, seed_everything

METRIC_SOURCE = Path(__file__).resolve().parents[1] / 'work/motiongpt_source/mGPT/metrics/utils.py'


class CriticProtocolTests(unittest.TestCase):
    def test_resampling_preserves_seconds_and_linear_root_velocity(self):
        pose = np.zeros((101, 24, 3), dtype=np.float32)
        translation = np.zeros((101, 3), dtype=np.float32)
        translation[:, 0] = np.arange(101) / 20
        rotations, root = resample_smpl(pose, translation, 20, 24)
        self.assertEqual(rotations.shape, (121, 24, 3))
        np.testing.assert_allclose(root[:, 0], np.arange(121) / 24, atol=1e-6)
        self.assertAlmostEqual(float(root[-1, 0]), 5.)

    def test_slerp_takes_short_rotation_path_across_axis_angle_wrap(self):
        pose = np.zeros((2, 24, 3), dtype=np.float32)
        pose[:, 0, 1] = np.deg2rad([179., -179.])
        rotation, _ = resample_smpl(pose, np.zeros((2, 3)), 1, 2)
        matrix = Rotation.from_rotvec(rotation[1, 0]).as_matrix()
        np.testing.assert_allclose(matrix, Rotation.from_euler('y', 180, degrees=True).as_matrix(), atol=1e-6)

    def test_windows_include_tail_without_duplicate_last_window(self):
        self.assertEqual(window_starts(59), [])
        self.assertEqual(window_starts(60), [0])
        self.assertEqual(window_starts(120), [0, 60])
        self.assertEqual(window_starts(125), [0, 60, 65])

    def test_invalid_input_is_rejected_instead_of_scored(self):
        with self.assertRaises(ValueError):
            resample_smpl(np.zeros((60, 22, 3)), np.zeros((60, 3)), 20, 24)
        with self.assertRaises(ValueError):
            resample_smpl(np.full((60, 24, 3), np.nan), np.zeros((60, 3)), 20, 24)
        with self.assertRaises(ValueError):
            window_starts(120, stride=61)

    def test_perceptual_windows_are_invariant_to_world_translation(self):
        critic = Critic.__new__(Critic)
        critic.protocol = {"source_fps": 20, "target_fps": 24, "window_frames": 60,
                           "stride_frames": 60, "optimizer": "adam"}
        critic.score_input = lambda motion: [float(np.square(motion[:, :, -1]).sum())]
        pose = np.zeros((101, 24, 3), dtype=np.float32)
        translation = np.zeros((101, 3), dtype=np.float32)
        translation[:, 0] = np.arange(101) / 20
        original = critic.score_smpl(pose, translation)
        shifted = critic.score_smpl(pose, translation + np.array([5., -2., 10.]))
        np.testing.assert_allclose(original['window_scores'], shifted['window_scores'], rtol=1e-6)

    def test_asset_lookup_does_not_silently_choose_between_normalizations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['one', 'two']:
                (root / name).mkdir()
                np.save(root / name / 'mean.npy', np.zeros(263))
            with self.assertRaises(FileNotFoundError):
                find_asset(root, 'mean.npy')
            self.assertEqual(find_asset(root, 'one/mean.npy'), root / 'one/mean.npy')


@unittest.skipUnless(METRIC_SOURCE.is_file(), 'Run source setup before upstream metric integration tests')
class MetricSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator.__new__(Evaluator)
        self.evaluator.utils = load_file('motiongpt_metric_math', METRIC_SOURCE)
        self.features = np.eye(64, dtype=np.float32)

    def test_perfect_alignment_and_identical_distributions(self):
        seed_everything(1)
        result = self.evaluator.summarize(self.features, self.features, self.features, 20)
        self.assertEqual(result['R_precision_top_1'], 1.)
        self.assertEqual(result['MM_Dist'], 0.)
        self.assertAlmostEqual(result['FID'], 0., places=6)

    def test_wrong_condition_can_have_perfect_fid(self):
        # Every motion is reassigned to another text. Distribution is unchanged,
        # but condition alignment is wrong; an independent metric sanity case.
        seed_everything(2)
        features = self.features[:32]
        result = self.evaluator.summarize(features, np.roll(features, 1, axis=0), features, 20)
        self.assertEqual(result['R_precision_top_1'], 0.)
        self.assertGreater(result['MM_Dist'], 1.)
        self.assertAlmostEqual(result['FID'], 0., places=6)

    def test_retrieval_uses_full_32_candidate_groups_only(self):
        seed_everything(3)
        result = self.evaluator.summarize(self.features[:33], self.features[:33], self.features[:33], 20)
        self.assertEqual(result['samples'], 33)
        self.assertEqual(result['retrieval_samples'], 32)
        self.assertEqual(result['retrieval_group_size'], 32)


if __name__ == '__main__':
    unittest.main()
