"""Bridge networks: one shared trunk for h (across arms), one for q (across arms).

Trunk: MLP 3x128 ReLU over concat(standardized inputs, arm_embedding(8)); scalar head.
Parameter sharing across arms helps rare arms (spec 1.5).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class TrunkNet(nn.Module):
    def __init__(self, feat_dim: int, K: int, arm_emb: int = 8, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.arm = nn.Embedding(K, arm_emb)
        dims = [feat_dim + arm_emb] + [hidden] * depth
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, feat: torch.Tensor, arm: torch.Tensor) -> torch.Tensor:
        e = self.arm(arm)
        z = torch.cat([feat, e], dim=-1)
        return self.head(self.trunk(z)).squeeze(-1)


class EMA:
    """Exponential moving average of parameters (decay 0.99)."""

    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

    def state_dict(self):
        return copy.deepcopy(self.shadow.state_dict())
