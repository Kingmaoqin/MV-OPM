"""Stage-2: pseudo-outcome regression (STARSTAR), g0 head, ATE contrasts + CIs (spec 1.6)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..utils import resolve_device, standardize_apply, standardize_fit


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden] * depth
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class HeadConfig:
    hidden: int = 256
    depth: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 300
    patience: int = 20
    batch_size: int = 256
    val_frac: float = 0.1
    device: str = "cpu"
    seed: int = 0


class MLPRegressor:
    """Standardized-input MLP regressor with early stopping on validation MSE."""

    def __init__(self, cfg: HeadConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MLPRegressor":
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        Y = Y.reshape(len(Y), -1).astype(np.float32)
        self.mx, self.sx = standardize_fit(X.astype(np.float32))
        Xs = standardize_apply(X.astype(np.float32), self.mx, self.sx)
        n = Xs.shape[0]
        perm = rng.permutation(n)
        nv = max(int(cfg.val_frac * n), 1)
        vi, ti = perm[:nv], perm[nv:]

        dev = self.device
        Xt = torch.tensor(Xs, device=dev)
        Yt = torch.tensor(Y, device=dev)
        fork_devices = [] if dev.type != "cuda" else [dev.index or torch.cuda.current_device()]
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(cfg.seed)
            self.model = _MLP(Xs.shape[1], Y.shape[1], cfg.hidden, cfg.depth).to(dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        lossf = nn.MSELoss()

        best, best_state, bad = np.inf, None, 0
        bs = min(cfg.batch_size, len(ti))
        for epoch in range(cfg.max_epochs):
            self.model.train()
            ep = rng.permutation(len(ti))
            for s in range(0, len(ti), bs):
                b = torch.tensor(ti[ep[s:s + bs]], dtype=torch.long, device=dev)
                opt.zero_grad()
                loss = lossf(self.model(Xt[b]), Yt[b])
                loss.backward(); opt.step()
            self.model.eval()
            with torch.no_grad():
                vloss = float(lossf(self.model(Xt[vi]), Yt[vi]))
            if vloss < best - 1e-6:
                best, best_state, bad = vloss, {k: v.clone() for k, v in self.model.state_dict().items()}, 0
            else:
                bad += 1
            if bad >= cfg.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = standardize_apply(X.astype(np.float32), self.mx, self.sx)
        Xt = torch.tensor(Xs, device=self.device)
        out = self.model(Xt).cpu().numpy()
        return out


def fit_cate_and_ate(X, phi, K, device="cpu", seed=0, head_kwargs=None):
    """Stage-2 from a pseudo-outcome matrix phi (n,K): tau head, g0 head, ATE CIs.

    Shared by OPM and every proximal baseline that produces a phi matrix.
    """
    head_kwargs = head_kwargs or {}
    D = phi[:, 1:] - phi[:, [0]]
    tau_head = MLPRegressor(HeadConfig(device=device, seed=seed, **head_kwargs)).fit(X, D)
    g0_head = MLPRegressor(HeadConfig(device=device, seed=seed + 1, **head_kwargs)).fit(X, phi[:, 0])
    ate = {k: ate_with_ci(phi[:, k] - phi[:, 0]) for k in range(1, K)}
    return tau_head, g0_head, ate


def ate_with_ci(psi: np.ndarray) -> dict:
    """ATE point + 95% CI from the per-sample contrast psi_i = phi_k - phi_0."""
    n = len(psi)
    ate = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    return {"ate": ate, "se": se, "ci_low": ate - 1.96 * se, "ci_high": ate + 1.96 * se}
