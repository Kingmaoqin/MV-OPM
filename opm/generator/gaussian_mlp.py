"""Gaussian-MLP generator (mean-matched) — comparator for E6(e).

Models Y | X, arm as Gaussian with an MLP mean/log-variance head, trained on factual
data.  'Mean-matched' means the sampling mean is replaced by the proximal mu_hat_k(X),
so it enjoys the same debiased mean as the distilled DDPM but a Gaussian shape.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..utils import resolve_device, standardize_apply, standardize_fit


class _GaussHead(nn.Module):
    def __init__(self, p, K, hidden=256, arm_emb=8):
        super().__init__()
        self.arm = nn.Embedding(K, arm_emb)
        self.body = nn.Sequential(nn.Linear(p + arm_emb, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, 1)
        self.logvar = nn.Linear(hidden, 1)

    def forward(self, x, arm):
        h = self.body(torch.cat([x, self.arm(arm)], -1))
        return self.mean(h)[:, 0], self.logvar(h)[:, 0].clamp(-6, 4)


class GaussianMLPGenerator:
    def __init__(self, p, K, device="cpu", seed=0):
        self.device = resolve_device(device)
        self.K = K
        torch.manual_seed(seed)
        self.model = _GaussHead(p, K).to(self.device)

    def fit(self, ds, epochs=200, lr=1e-3, batch=256, seed=0):
        rng = np.random.default_rng(seed)
        self.mx, self.sx = standardize_fit(ds.X.astype(np.float32))
        Xs = standardize_apply(ds.X.astype(np.float32), self.mx, self.sx)
        self.my, self.sy = float(ds.Y.mean()), float(ds.Y.std() + 1e-8)
        y = ((ds.Y - self.my) / self.sy).astype(np.float32)
        dev = self.device
        Xt = torch.tensor(Xs, device=dev); Yt = torch.tensor(y, device=dev)
        Tt = torch.tensor(ds.T, dtype=torch.long, device=dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        n = ds.n
        for _ in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, batch):
                b = torch.tensor(perm[s:s + batch], dtype=torch.long, device=dev)
                mean, logvar = self.model(Xt[b], Tt[b])
                nll = 0.5 * (logvar + (Yt[b] - mean) ** 2 / logvar.exp())
                opt.zero_grad(); nll.mean().backward(); opt.step()
        self.model.eval()
        return self

    @torch.no_grad()
    def sample(self, X, arm_k, M, mu_hat_k=None):
        Xs = standardize_apply(X.astype(np.float32), self.mx, self.sx)
        xt = torch.tensor(Xs, device=self.device)
        arm = torch.full((xt.shape[0],), arm_k, dtype=torch.long, device=self.device)
        mean, logvar = self.model(xt, arm)
        std = (0.5 * logvar).exp().cpu().numpy() * self.sy
        eps = np.random.default_rng(0).normal(size=(xt.shape[0], M))
        if mu_hat_k is not None:            # mean-matched to the proximal estimate
            base = mu_hat_k.reshape(-1, 1)
        else:
            base = (mean.cpu().numpy() * self.sy + self.my).reshape(-1, 1)
        return base + std.reshape(-1, 1) * eps
