"""Kernel-moment (U-statistic) estimation of outcome/treatment bridges (spec 1.5).

For arm k, Gaussian kernel Kern(a,b)=exp(-||a-b||^2/(2 bw^2)):

  L_h = 1/(B(B-1)) sum_{i!=j} R_i R_j Kern(z_i,z_j),
        R_i = 1{T_i=k}(Y_i - h_k(W_i,X_i)),  z_i = std(concat(V_i,X_i))
  L_q = 1/(B(B-1)) sum_{i!=j} S_i S_j Kern(u_i,u_j),
        S_i = 1{T_i=k} q_k(V_i,X_i) - 1,     u_i = std(concat(W_i,X_i))

with q_k = 1 + softplus(g_k), clipped at q_max=50.  Bandwidth: median heuristic on
the training fold.  EMA(0.99) weights are used for all downstream predictions.
Early stopping uses the random-Fourier-feature bridge-residual diagnostic (Section 6.1).
No adversarial inner loop (NMMR-style closed form).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..dgp.base import CausalDataset
from ..utils import (median_bandwidth, resolve_device, standardize_apply,
                     standardize_fit)
from .networks import EMA, TrunkNet


# --------------------------- kernel primitives -----------------------------
def gaussian_kernel(z: torch.Tensor, bw: float) -> torch.Tensor:
    """Full BxB Gaussian kernel matrix (diagonal NOT zeroed)."""
    d2 = torch.cdist(z, z) ** 2
    return torch.exp(-d2 / (2.0 * bw * bw))


def ustat_quadratic(vals: torch.Tensor, Kmat: torch.Tensor) -> torch.Tensor:
    """U-statistic (1/(B(B-1))) sum_{i!=j} vals_i vals_j K_ij, diagonal zeroed."""
    B = vals.shape[0]
    Kz = Kmat - torch.diag(torch.diag(Kmat))
    quad = vals @ (Kz @ vals)
    return quad / (B * (B - 1))


@dataclass
class BridgeConfig:
    K: int
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 300
    patience: int = 30
    batch_size: int = 512
    hidden: int = 128
    depth: int = 3
    arm_emb: int = 8
    ema_decay: float = 0.99
    q_max: float = 50.0
    n_rff: int = 100
    val_frac: float = 0.2
    eval_every: int = 2
    device: str = "cpu"
    seed: int = 0
    verbose: bool = False


class KernelBridges:
    """Estimates outcome bridge h_k and treatment bridge q_k on one training fold."""

    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.K = cfg.K
        self._fitted = False

    # ---- input featurization ----
    def _feat_h(self, W, X):     # network input for h_k: (W, X)
        return np.concatenate([W, X], axis=1)

    def _feat_q(self, V, X):     # network input for q_k: (V, X)
        return np.concatenate([V, X], axis=1)

    def _inst_h(self, V, X):     # kernel instrument for h: z=(V, X)
        return np.concatenate([V, X], axis=1)

    def _inst_q(self, W, X):     # kernel instrument for q: u=(W, X)
        return np.concatenate([W, X], axis=1)

    def _rff(self, n_dim, bw, key):
        rng = np.random.default_rng(self.cfg.seed + key)
        omega = rng.normal(0, 1.0 / bw, size=(self.cfg.n_rff, n_dim))
        b = rng.uniform(0, 2 * np.pi, size=self.cfg.n_rff)
        return (torch.tensor(omega, dtype=torch.float32, device=self.device),
                torch.tensor(b, dtype=torch.float32, device=self.device))

    def fit(self, ds: CausalDataset) -> "KernelBridges":
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        W, V, X = ds.W, ds.V, ds.X
        Y, T = ds.Y.astype(np.float32), ds.T.astype(np.int64)

        feat_h = self._feat_h(W, X).astype(np.float32)
        feat_q = self._feat_q(V, X).astype(np.float32)
        inst_h = self._inst_h(V, X).astype(np.float32)
        inst_q = self._inst_q(W, X).astype(np.float32)

        # standardizers (fit on this training fold)
        self.mh, self.sh = standardize_fit(feat_h)
        self.mq, self.sq = standardize_fit(feat_q)
        self.mzh, self.szh = standardize_fit(inst_h)
        self.mzq, self.szq = standardize_fit(inst_q)
        feat_h = standardize_apply(feat_h, self.mh, self.sh)
        feat_q = standardize_apply(feat_q, self.mq, self.sq)
        inst_h = standardize_apply(inst_h, self.mzh, self.szh)
        inst_q = standardize_apply(inst_q, self.mzq, self.szq)

        self.bw_h = median_bandwidth(inst_h, seed=cfg.seed)
        self.bw_q = median_bandwidth(inst_q, seed=cfg.seed + 1)

        # nets
        self.net_h = TrunkNet(feat_h.shape[1], self.K, cfg.arm_emb, cfg.hidden, cfg.depth).to(self.device)
        self.net_q = TrunkNet(feat_q.shape[1], self.K, cfg.arm_emb, cfg.hidden, cfg.depth).to(self.device)
        ema_h, ema_q = EMA(self.net_h, cfg.ema_decay), EMA(self.net_q, cfg.ema_decay)
        opt = torch.optim.AdamW(list(self.net_h.parameters()) + list(self.net_q.parameters()),
                                lr=cfg.lr, weight_decay=cfg.weight_decay)

        # train/val split
        n = X.shape[0]
        perm = rng.permutation(n)
        n_val = max(int(cfg.val_frac * n), 1)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        dev = self.device
        tens = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        fh, fq = tens(feat_h), tens(feat_q)
        zh, zq = tens(inst_h), tens(inst_q)
        Yt = tens(Y)
        Tt = torch.tensor(T, dtype=torch.long, device=dev)

        omega_h, b_h = self._rff(inst_h.shape[1], self.bw_h, key=11)
        omega_q, b_q = self._rff(inst_q.shape[1], self.bw_q, key=13)

        best_score = np.inf
        best_state = (ema_h.state_dict(), ema_q.state_dict())
        bad = 0
        bs = min(cfg.batch_size, len(tr_idx))

        for epoch in range(cfg.max_epochs):
            self.net_h.train(); self.net_q.train()
            ep = rng.permutation(len(tr_idx))
            for start in range(0, len(tr_idx) - 1, bs):
                bidx = tr_idx[ep[start:start + bs]]
                if len(bidx) < 8:
                    continue
                bt = torch.tensor(bidx, dtype=torch.long, device=dev)
                Kh = gaussian_kernel(zh[bt], self.bw_h)
                Kq = gaussian_kernel(zq[bt], self.bw_q)
                loss = torch.zeros((), device=dev)
                Tb, Yb = Tt[bt], Yt[bt]
                for k in range(self.K):
                    arm = torch.full((len(bidx),), k, dtype=torch.long, device=dev)
                    mask = (Tb == k).float()
                    hk = self.net_h(fh[bt], arm)
                    R = mask * (Yb - hk)
                    loss = loss + ustat_quadratic(R, Kh)
                    gk = self.net_q(fq[bt], arm)
                    qk = torch.clamp(1.0 + F.softplus(gk), max=cfg.q_max)
                    S = mask * qk - 1.0
                    loss = loss + ustat_quadratic(S, Kq)
                opt.zero_grad(); loss.backward(); opt.step()
                ema_h.update(self.net_h); ema_q.update(self.net_q)

            if epoch % cfg.eval_every == 0 or epoch == cfg.max_epochs - 1:
                score = self._val_residual(ema_h, ema_q, fh, fq, zh, zq, Yt, Tt,
                                           torch.tensor(val_idx, dtype=torch.long, device=dev),
                                           omega_h, b_h, omega_q, b_q)
                if score < best_score - 1e-5:
                    best_score, best_state, bad = score, (ema_h.state_dict(), ema_q.state_dict()), 0
                else:
                    bad += cfg.eval_every
                if cfg.verbose and epoch % 20 == 0:
                    print(f"  [bridge] epoch {epoch} val_resid {score:.4f} best {best_score:.4f}")
                if bad >= cfg.patience:
                    break

        # load best EMA weights for downstream predictions
        self.net_h.load_state_dict(best_state[0])
        self.net_q.load_state_dict(best_state[1])
        self.net_h.eval(); self.net_q.eval()
        self.best_resid = float(best_score)
        self._fitted = True
        return self

    @torch.no_grad()
    def _val_residual(self, ema_h, ema_q, fh, fq, zh, zq, Yt, Tt, vidx,
                      omega_h, b_h, omega_q, b_q) -> float:
        nh, nq = ema_h.shadow, ema_q.shadow
        gz_h = torch.cos(zh[vidx] @ omega_h.T + b_h)   # (nv, n_rff)
        gz_q = torch.cos(zq[vidx] @ omega_q.T + b_q)
        score = 0.0
        for k in range(self.K):
            arm = torch.full((len(vidx),), k, dtype=torch.long, device=self.device)
            mask = (Tt[vidx] == k).float()
            hk = nh(fh[vidx], arm)
            R = mask * (Yt[vidx] - hk)
            sdR = R.std() + 1e-6
            m_h = (R[:, None] * gz_h).mean(0).abs().max() / sdR
            gk = nq(fq[vidx], arm)
            qk = torch.clamp(1.0 + F.softplus(gk), max=self.cfg.q_max)
            S = mask * qk - 1.0
            sdS = S.std() + 1e-6
            m_q = (S[:, None] * gz_q).mean(0).abs().max() / sdS
            score += float(m_h + m_q)
        return score / self.K

    # ------------------------- prediction API -------------------------
    @torch.no_grad()
    def predict_h(self, W, X) -> np.ndarray:
        """(n, K) outcome-bridge values h_k(W_i,X_i) for all arms."""
        assert self._fitted
        feat = standardize_apply(self._feat_h(W, X).astype(np.float32), self.mh, self.sh)
        ft = torch.tensor(feat, dtype=torch.float32, device=self.device)
        out = np.zeros((X.shape[0], self.K), dtype=np.float64)
        for k in range(self.K):
            arm = torch.full((X.shape[0],), k, dtype=torch.long, device=self.device)
            out[:, k] = self.net_h(ft, arm).cpu().numpy()
        return out

    @torch.no_grad()
    def predict_q(self, V, X) -> np.ndarray:
        """(n, K) treatment-bridge values q_k(V_i,X_i) for all arms (clipped at q_max)."""
        assert self._fitted
        feat = standardize_apply(self._feat_q(V, X).astype(np.float32), self.mq, self.sq)
        ft = torch.tensor(feat, dtype=torch.float32, device=self.device)
        out = np.zeros((X.shape[0], self.K), dtype=np.float64)
        for k in range(self.K):
            arm = torch.full((X.shape[0],), k, dtype=torch.long, device=self.device)
            gk = self.net_q(ft, arm)
            qk = torch.clamp(1.0 + F.softplus(gk), max=self.cfg.q_max)
            out[:, k] = qk.cpu().numpy()
        return out
