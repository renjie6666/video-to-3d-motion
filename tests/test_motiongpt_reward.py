"""Independent checks of adapter freezing and reward-driven update direction."""
import unittest
import tempfile
from pathlib import Path

import numpy as np

import torch
from torch import nn

from video_to_3d_motion.motion_generation.adapters import LoRALinear, clipped_reward_loss
from video_to_3d_motion.motion_generation.runtime import load_file


class RewardPilotTests(unittest.TestCase):
    def test_train_selector_excludes_mirrored_test_ids_and_heldout_captions(self):
        module = load_file('reward_pilot_selector_test', Path(__file__).resolve().parents[1] / 'tools/run_motiongpt_reward_pilot.py')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'new_joint_vecs').mkdir()
            (root / 'texts').mkdir()
            (root / 'train.txt').write_text('M000001\n000002\n000003\n')
            (root / 'test.txt').write_text('000001\n')
            (root / 'val.txt').write_text('')
            for identity, caption in [('M000001','mirror'), ('000002','held out caption'), ('000003','train only')]:
                np.save(root / 'new_joint_vecs' / (identity+'.npy'), np.zeros((104,263),dtype=np.float32))
                (root / 'texts' / (identity+'.txt')).write_text(caption+'#tokens#0.0#0.0')
            panel = {'samples':[{'motion_id':'000999','caption':'HELD OUT CAPTION'}]}
            selected, _ = module.select_training_condition(root, panel, 1)
            self.assertEqual(selected['motion_id'],'000003')

    def test_zero_adapter_preserves_base_and_only_adapter_updates(self):
        torch.manual_seed(17)
        base = nn.Linear(4, 3)
        before = base.weight.detach().clone()
        layer = LoRALinear(base, 2, 4.)
        x = torch.randn(5, 4)
        torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
        optimizer = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=.1)
        layer(x).square().mean().backward()
        optimizer.step()
        torch.testing.assert_close(before, base.weight, rtol=0, atol=0)
        self.assertIsNone(base.weight.grad)
        self.assertGreater(layer.b.abs().sum().item(), 0)
        layer.enabled = False
        torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)

    def test_reversing_rewards_reverses_preferred_action_update(self):
        for advantage, expected_sign in [(1., 1), (-1., -1)]:
            logits = nn.Parameter(torch.zeros(1, 1, 2))
            old = torch.log_softmax(logits.detach(), -1)
            optimizer = torch.optim.SGD([logits], lr=.1)
            labels = torch.zeros(1, 1, dtype=torch.long)
            loss, _ = clipped_reward_loss(torch.log_softmax(logits, -1), old[:, :, 0], old,
                                          labels, torch.ones_like(labels, dtype=torch.bool), advantage)
            loss.backward()
            optimizer.step()
            self.assertGreater(expected_sign * (logits[0, 0, 0] - logits[0, 0, 1]).item(), 0)

    def test_padding_cannot_change_reward_loss_or_gradient(self):
        logits = torch.tensor([[[1., 0.], [0., 9.]]], requires_grad=True)
        reference = torch.log_softmax(torch.zeros_like(logits), -1)
        labels = torch.zeros(1, 2, dtype=torch.long)
        loss, _ = clipped_reward_loss(torch.log_softmax(logits, -1), reference[:, :, 0], reference,
                                      labels, torch.tensor([[True, False]]), .5)
        loss.backward()
        torch.testing.assert_close(logits.grad[:, 1], torch.zeros_like(logits.grad[:, 1]))

    def test_empty_rollout_is_rejected(self):
        p = torch.log_softmax(torch.zeros(1, 1, 2), -1)
        with self.assertRaises(ValueError):
            clipped_reward_loss(p, p[:, :, 0], p, torch.zeros(1, 1, dtype=torch.long),
                                torch.zeros(1, 1, dtype=torch.bool), 1.)


if __name__ == "__main__":
    unittest.main()
