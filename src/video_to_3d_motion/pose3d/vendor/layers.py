"""Evaluation-only replacement for timm's parameter-free DropPath."""

from torch import nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training:
            raise RuntimeError("the vendored PoseMamba backend supports inference only")
        return x
