"""In-repo neural ignorability baselines: TARNet and DragonNet (spec Section 4).

Multi-arm: a shared representation with K arm-specific outcome heads.  DragonNet adds a
propensity head over the shared representation (softmax over K arms).  Targeted
regularization is omitted (documented simplification in DECISIONS.md); the propensity
head still shapes the representation.  CATE: tau_k(x) = head_k(phi(x)) - head_0(phi(x)).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..dgp.base import CausalDataset
from ..utils import resolve_device, standardize_apply, standardize_fit
from .base import Baseline


class _TarNetModule(nn.Module):
    def __init__(self, p, K, rep=128, hyp=64, dragon=False):
        super().__init__()
        self.rep = nn.Sequential(nn.Linear(p, rep), nn.ELU(), nn.Linear(rep, rep), nn.ELU(),
                                 nn.Linear(rep, rep), nn.ELU())
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(rep, hyp), nn.ELU(), nn.Linear(hyp, hyp), nn.ELU(),
                          nn.Linear(hyp, 1)) for _ in range(K)])
        self.dragon = dragon
        if dragon:
            self.prop = nn.Linear(rep, K)

    def forward(self, x):
        r = self.rep(x)
        outs = torch.cat([h(r) for h in self.heads], dim=1)   # (n, K)
        logits = self.prop(r) if self.dragon else None
        return outs, logits


class _NeuralBase(Baseline):
    def __init__(self, name, K, seed=0, device="cpu", dragon=False,
                 epochs=150, lr=1e-3, batch=256, prop_weight=1.0,
                 val_frac=0.2, patience=15, weight_decay=1e-3):
        self.name = name
        self.K = K
        self.seed = seed
        self.device = resolve_device(device)
        self.dragon = dragon
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.prop_weight = prop_weight
        self.val_frac = val_frac
        self.patience = patience
        self.weight_decay = weight_decay

    def fit(self, ds: CausalDataset) -> "_NeuralBase":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.mx, self.sx = standardize_fit(ds.X.astype(np.float32))
        Xs = standardize_apply(ds.X.astype(np.float32), self.mx, self.sx)
        self.my, self.sy = float(ds.Y.mean()), float(ds.Y.std() + 1e-8)
        Yn = ((ds.Y - self.my) / self.sy).astype(np.float32)
        dev = self.device
        Xt = torch.tensor(Xs, device=dev)
        Yt = torch.tensor(Yn, device=dev)
        Tt = torch.tensor(ds.T, dtype=torch.long, device=dev)
        onehot = torch.zeros(ds.n, self.K, device=dev)
        onehot[torch.arange(ds.n), Tt] = 1.0

        n = ds.n
        perm0 = rng.permutation(n)
        nv = max(int(self.val_frac * n), 1)
        vi = torch.tensor(perm0[:nv], dtype=torch.long, device=dev)
        ti = perm0[nv:]

        self.model = _TarNetModule(ds.p, self.K, dragon=self.dragon).to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        mse = nn.MSELoss(reduction="none")
        ce = nn.CrossEntropyLoss()
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.model.train()
            ep = rng.permutation(len(ti))
            for s in range(0, len(ti), self.batch):
                b = torch.tensor(ti[ep[s:s + self.batch]], dtype=torch.long, device=dev)
                opt.zero_grad()
                outs, logits = self.model(Xt[b])
                factual = (outs * onehot[b]).sum(1)
                loss = mse(factual, Yt[b]).mean()
                if self.dragon:
                    loss = loss + self.prop_weight * ce(logits, Tt[b])
                loss.backward(); opt.step()
            self.model.eval()
            with torch.no_grad():
                outs, _ = self.model(Xt[vi])
                vfac = (outs * onehot[vi]).sum(1)
                vloss = float(mse(vfac, Yt[vi]).mean())
            if vloss < best - 1e-6:
                best, best_state, bad = vloss, {k: v.clone() for k, v in self.model.state_dict().items()}, 0
            else:
                bad += 1
            if bad >= self.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_cate(self, X):
        Xs = standardize_apply(X.astype(np.float32), self.mx, self.sx)
        outs, _ = self.model(torch.tensor(Xs, device=self.device))
        outs = outs.cpu().numpy() * self.sy + self.my
        return outs[:, 1:] - outs[:, [0]]

    @torch.no_grad()
    def predict_mu(self, X):
        Xs = standardize_apply(X.astype(np.float32), self.mx, self.sx)
        outs, _ = self.model(torch.tensor(Xs, device=self.device))
        return outs.cpu().numpy() * self.sy + self.my

    def ate_vector(self):
        raise RuntimeError("call ate_from(X)")

    def ate_from(self, X):
        return self.predict_cate(X).mean(axis=0)


class TARNet(_NeuralBase):
    def __init__(self, name="TARNet", **kw):
        super().__init__(name=name, dragon=False, **kw)


class DragonNet(_NeuralBase):
    def __init__(self, name="DragonNet", **kw):
        super().__init__(name=name, dragon=True, **kw)
