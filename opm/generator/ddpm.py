"""Stage-3 conditional DDPM over scalar y (spec 1.7). Stages 1-2 stay frozen.

Conditioning c = concat(MLP_embed(X), arm_embedding).  T_steps=200, cosine schedule,
eps-prediction MLP (4x256), trained on FACTUAL (X_i,T_i,Y_i) with the denoising loss.
For distillation gradients we use a short DDIM sampler (n_distill_steps, deterministic)
so backprop through sampling is tractable on CPU (documented reduction in DECISIONS.md).
Plain E7 sampling uses full ancestral DDPM.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import resolve_device, standardize_apply, standardize_fit


def cosine_alpha_bar(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos(((steps / T) + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    return ab.clamp(1e-5, 1.0)          # abar[0..T]


def _sin_embed(t, dim=32):
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class EpsMLP(nn.Module):
    def __init__(self, p, K, x_emb=32, arm_emb=8, t_emb=32, hidden=256):
        super().__init__()
        self.x_embed = nn.Sequential(nn.Linear(p, 64), nn.SiLU(), nn.Linear(64, x_emb))
        self.arm = nn.Embedding(K, arm_emb)
        cdim = x_emb + arm_emb + t_emb + 1
        self.net = nn.Sequential(
            nn.Linear(cdim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))
        self.t_emb = t_emb

    def forward(self, y_t, t, x, arm):
        c = torch.cat([self.x_embed(x), self.arm(arm), _sin_embed(t, self.t_emb),
                       y_t[:, None]], dim=-1)
        return self.net(c)[:, 0]


class ConditionalDDPM:
    def __init__(self, p, K, T=200, device="cpu", seed=0, hidden=256):
        self.device = resolve_device(device)
        self.T = T
        self.K = K
        torch.manual_seed(seed)
        self.model = EpsMLP(p, K, hidden=hidden).to(self.device)
        ab = cosine_alpha_bar(T).to(self.device)
        self.abar = ab[1:]                          # index by t in [0,T-1]
        self.sqrt_ab = self.abar.sqrt()
        self.sqrt_1mab = (1 - self.abar).sqrt()

    # ---- factual denoising loss ----
    def _denoise_loss(self, y0, x, arm, rng):
        n = y0.shape[0]
        t = torch.randint(0, self.T, (n,), device=self.device)
        eps = torch.randn(n, device=self.device)
        y_t = self.sqrt_ab[t] * y0 + self.sqrt_1mab[t] * eps
        eps_hat = self.model(y_t, t, x, arm)
        return F.mse_loss(eps_hat, eps)

    def fit(self, ds, y_scaler=None, mu_hat=None, lambda_distill=0.0,
            epochs=200, lr=1e-3, batch=256, distill_subset=64, M=16,
            n_distill_steps=25, distill_every=10, seed=0, verbose=False):
        rng = np.random.default_rng(seed)
        self.mx, self.sx = standardize_fit(ds.X.astype(np.float32))
        Xs = standardize_apply(ds.X.astype(np.float32), self.mx, self.sx)
        self.my, self.sy = float(ds.Y.mean()), float(ds.Y.std() + 1e-8)
        y0 = ((ds.Y - self.my) / self.sy).astype(np.float32)
        dev = self.device
        Xt = torch.tensor(Xs, device=dev)
        Yt = torch.tensor(y0, device=dev)
        Tt = torch.tensor(ds.T, dtype=torch.long, device=dev)
        mu_t = None
        if mu_hat is not None:
            mu_t = torch.tensor(((mu_hat - self.my) / self.sy).astype(np.float32), device=dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        n = ds.n
        step = 0
        for _ in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, batch):
                b = torch.tensor(perm[s:s + batch], dtype=torch.long, device=dev)
                opt.zero_grad()
                loss = self._denoise_loss(Yt[b], Xt[b], Tt[b], rng)
                if lambda_distill > 0 and mu_t is not None and step % distill_every == 0:
                    loss = loss + lambda_distill * self._distill_loss(
                        Xt, mu_t, distill_subset, M, n_distill_steps, rng)
                loss.backward(); opt.step()
                step += 1
        self.model.eval()
        return self

    def _distill_loss(self, Xt, mu_t, subset, M, n_steps, rng):
        n = Xt.shape[0]
        idx = torch.tensor(rng.choice(n, size=min(subset, n), replace=False),
                           dtype=torch.long, device=self.device)
        total = 0.0
        for k in range(self.K):
            xk = Xt[idx]
            arm = torch.full((len(idx),), k, dtype=torch.long, device=self.device)
            samp = self._ddim_sample(xk, arm, M, n_steps)     # (len(idx), M)
            mean_M = samp.mean(dim=1)
            total = total + F.mse_loss(mean_M, mu_t[idx, k])
        return total / self.K

    def _ddim_sample(self, x, arm, M, n_steps):
        """Deterministic DDIM sampling; gradient-friendly. Returns (n, M) in scaled y."""
        dev = self.device
        nrep = x.shape[0]
        xr = x.repeat_interleave(M, 0)
        ar = arm.repeat_interleave(M, 0)
        y = torch.randn(nrep * M, device=dev)
        ts = torch.linspace(self.T - 1, 0, n_steps, device=dev).long()
        for i in range(len(ts)):
            t = ts[i]
            tb = torch.full((nrep * M,), int(t), dtype=torch.long, device=dev)
            eps = self.model(y, tb, xr, ar)
            ab = self.abar[t]
            y0 = (y - (1 - ab).sqrt() * eps) / ab.sqrt()
            if i < len(ts) - 1:
                ab_next = self.abar[ts[i + 1]]
                y = ab_next.sqrt() * y0 + (1 - ab_next).sqrt() * eps
            else:
                y = y0
        return y.view(nrep, M)

    @torch.no_grad()
    def sample(self, X, arm_k, M, ancestral=True, n_steps=None):
        """Sample M outcomes per row of X for arm k. Returns (n, M) in ORIGINAL y units."""
        Xs = standardize_apply(X.astype(np.float32), self.mx, self.sx)
        xt = torch.tensor(Xs, device=self.device)
        arm = torch.full((xt.shape[0],), arm_k, dtype=torch.long, device=self.device)
        if ancestral:
            samp = self._ancestral_sample(xt, arm, M)
        else:
            samp = self._ddim_sample(xt, arm, M, n_steps or 50)
        return samp.cpu().numpy() * self.sy + self.my

    @torch.no_grad()
    def _ancestral_sample(self, x, arm, M):
        dev = self.device
        nrep = x.shape[0]
        xr = x.repeat_interleave(M, 0)
        ar = arm.repeat_interleave(M, 0)
        y = torch.randn(nrep * M, device=dev)
        abar = self.abar
        for t in range(self.T - 1, -1, -1):
            tb = torch.full((nrep * M,), t, dtype=torch.long, device=dev)
            eps = self.model(y, tb, xr, ar)
            ab = abar[t]
            ab_prev = abar[t - 1] if t > 0 else torch.tensor(1.0, device=dev)
            beta = 1 - ab / ab_prev
            y0 = (y - (1 - ab).sqrt() * eps) / ab.sqrt()
            mean = ab_prev.sqrt() * beta / (1 - ab) * y0 + (ab / ab_prev).sqrt() * (1 - ab_prev) / (1 - ab) * y
            if t > 0:
                y = mean + beta.sqrt() * torch.randn_like(y)
            else:
                y = y0
        return y.view(nrep, M)
