"""PoseMamba's four-way scan with an inference-only PyTorch SSM recurrence.

Cross-scan/merge forward formulas are from the pinned Apache-2.0 csms6s.py.
The SSM recurrence composes (a,b) affine maps within short blocks. It never
divides by cumulative decay products and remains stable when they underflow.
"""

import torch
from torch.nn import functional as F


def selective_scan_torch(u, delta, A, B, C, D=None, delta_bias=None,
                         delta_softplus=True, chunk_size=256):
    batch, channels, length = u.shape
    groups, states = B.shape[1:3]
    if channels % groups or A.shape != (channels, states) or C.shape != B.shape:
        raise ValueError("incompatible grouped selective-scan shapes")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    per_group = channels // groups
    u = u.float().reshape(batch, groups, per_group, length)
    delta = delta.float().reshape(batch, groups, per_group, length)
    if delta_bias is not None:
        delta = delta + delta_bias.float().reshape(1, groups, per_group, 1)
    if delta_softplus:
        delta = F.softplus(delta)
    A = A.float().reshape(1, groups, per_group, states, 1)
    B, C = B.float(), C.float()
    state = u.new_zeros(batch, groups, per_group, states)
    output = torch.empty_like(u)
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        dt = delta[..., start:end].unsqueeze(-2)
        a = torch.exp(dt * A)
        b = dt * B[:, :, None, :, start:end] * u[..., start:end].unsqueeze(-2)
        # Inclusive affine prefix: (a2,b2) o (a1,b1) = (a2*a1,b2+a2*b1).
        offset = 1
        while offset < end - start:
            next_b = b[..., offset:] + a[..., offset:] * b[..., :-offset]
            next_a = a[..., offset:] * a[..., :-offset]
            b[..., offset:] = next_b
            a[..., offset:] = next_a
            offset *= 2
        b += a * state.unsqueeze(-1)
        state = b[..., -1].clone()
        output[..., start:end] = (b * C[:, :, None, :, start:end]).sum(dim=-2)
    if D is not None:
        output += D.float().reshape(1, groups, per_group, 1) * u
    return output.reshape(batch, channels, length)


class SelectiveScanCore:
    @staticmethod
    def apply(u, delta, A, B, C, D, delta_bias, delta_softplus, *unused):
        if torch.is_grad_enabled():
            raise RuntimeError("selective_scan_torch is an inference-only backend")
        return selective_scan_torch(u, delta, A, B, C, D, delta_bias, delta_softplus)


class CrossScan_plus_poselimbs:
    @staticmethod
    def apply(x):
        batch, channels, frames, joints = x.shape
        if joints != 17:
            raise ValueError("PoseMamba requires 17 joints")
        xs = x.new_empty(batch, 4, channels, frames * joints)
        indices = [0, 0, 1, 2, 3, 0, 4, 5, 6, 8, 11, 12, 13, 8, 14, 15, 16]
        xs[:, 0] = (x + x[..., indices]).flatten(2, 3)
        xs[:, 1] = x.transpose(2, 3).flatten(2, 3)
        xs[:, 2:4] = xs[:, 0:2].flip(dims=[-1])
        return xs


class CrossMerge_plus_poselimbs:
    @staticmethod
    def apply(ys):
        batch, groups, channels, frames, joints = ys.shape
        ys = ys.reshape(batch, groups, channels, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1])
        return ys[:, 0] + ys[:, 1].reshape(batch, channels, joints, frames).transpose(2, 3).reshape(batch, channels, -1)


# Defaults are unused by the pinned v2_plus_poselimbs configuration.
CrossScan = CrossScan_plus_poselimbs
CrossMerge = CrossMerge_plus_poselimbs
