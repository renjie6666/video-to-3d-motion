"""Small, explicit LoRA adapters for frozen T5 decoder self-attention."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, x):
        result = self.base(x)
        return result + (x @ self.a.T @ self.b.T) * self.scale if self.enabled else result


def inject_lora(model: nn.Module, rank: int = 4, alpha: float = 8.) -> list[str]:
    model.requires_grad_(False)
    targets = [(name, module) for name, module in model.named_modules()
               if name.startswith("decoder.") and ".SelfAttention." in name
               and name.rsplit(".", 1)[-1] in ("q", "v") and isinstance(module, nn.Linear)]
    if not targets:
        raise ValueError("No T5 decoder self-attention q/v projections found")
    for name, module in targets:
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, LoRALinear(module, rank, alpha))
    return [name for name, _ in targets]


@contextmanager
def adapters_disabled(model):
    adapters = [module for module in model.modules() if isinstance(module, LoRALinear)]
    states = [module.enabled for module in adapters]
    try:
        for module in adapters:
            module.enabled = False
        yield
    finally:
        for module, enabled in zip(adapters, states):
            module.enabled = enabled


def adapter_state(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def restore_adapter_state(model, state):
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError("Adapter parameter names do not match the model")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != state[name].shape:
                raise ValueError(f"Adapter shape mismatch: {name}")
            parameter.copy_(state[name].to(parameter))


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            array = parameter.detach().cpu().contiguous().numpy()
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def clipped_reward_loss(log_probs, old_log_probs, reference_log_probs, labels, mask, advantage,
                        clip_range=.2, kl_weight=.02):
    """Token-level clipped group advantage loss plus exact categorical KL.

    The per-motion terminal reward advantage is shared across its valid tokens.
    Average tokens within each motion before averaging motions at the caller.
    This pilot uses no learned value function and is not a full PPO pipeline.
    """
    if not mask.any():
        raise ValueError("Cannot optimize an empty sequence")
    taken = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    ratio = (taken - old_log_probs).exp()
    advantage = torch.as_tensor(advantage, dtype=log_probs.dtype, device=log_probs.device)
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_range, 1 + clip_range) * advantage)
    kl = (log_probs.exp() * (log_probs - reference_log_probs)).sum(-1)
    objective = (-surrogate[mask]).mean() + kl_weight * kl[mask].mean()
    return objective, {"kl": kl[mask].mean().detach(), "ratio": ratio[mask].mean().detach()}
