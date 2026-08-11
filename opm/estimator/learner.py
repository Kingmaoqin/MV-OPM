"""Top-level OPM learner: cross-fit bridges -> OOF (STAR) -> Stage-2 tau head.

Modes (spec 1.8 / Section 4):
  * ``proximal``    kernel-moment bridges (the method).
  * ``dr_fallback`` plug-in DR-learner (no proxies).
  * ``sieve``       polynomial-sieve proximal bridges (baseline / E6(a) ablation).

Strictly sequential: Stage-1 bridges are fully fit and frozen before Stage-2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..bridges.kernel_moment import BridgeConfig, KernelBridges
from ..bridges.plugin import PlugInBridges
from ..dgp.base import CausalDataset
from .crossfit import CrossFitter
from .pseudo_outcome import compute_phi
from .tau_head import HeadConfig, MLPRegressor, ate_with_ci


@dataclass
class OPMConfig:
    K: int
    mode: str = "proximal"          # proximal | dr_fallback | sieve
    n_folds: int = 5
    q_max: float = 50.0
    device: str = "cpu"
    seed: int = 0
    sieve_degree: int = 2
    fallback_linear: bool = False   # dr_fallback with linear nuisances (E4 gate)
    bridge_kwargs: dict = field(default_factory=dict)
    head_kwargs: dict = field(default_factory=dict)


class OPM:
    def __init__(self, cfg: OPMConfig):
        self.cfg = cfg
        self.K = cfg.K

    def _make_bridge(self, seed: int):
        c = self.cfg
        if c.mode == "proximal":
            bc = BridgeConfig(K=c.K, q_max=c.q_max, device=c.device, seed=seed, **c.bridge_kwargs)
            return KernelBridges(bc)
        if c.mode == "dr_fallback":
            return PlugInBridges(K=c.K, q_max=c.q_max, seed=seed, linear=c.fallback_linear)
        if c.mode == "sieve":
            from ..bridges.sieve import SieveBridges
            return SieveBridges(K=c.K, degree=c.sieve_degree, q_max=c.q_max)
        raise ValueError(f"unknown mode {c.mode}")

    def fit(self, ds: CausalDataset) -> "OPM":
        c = self.cfg
        n, K = ds.n, self.K
        phi = np.zeros((n, K))
        h_oof = np.zeros((n, K))
        q_oof = np.zeros((n, K))

        if c.n_folds <= 1:
            # cross-fitting OFF (E6b ablation): fit on all, predict in-sample (leakage).
            self.cf = None
            br = self._make_bridge(seed=c.seed + 1).fit(ds)
            h_oof = br.predict_h(ds.W, ds.X)
            q_oof = br.predict_q(ds.V, ds.X)
            phi = compute_phi(h_oof, q_oof, ds)
        else:
            self.cf = CrossFitter(n, c.n_folds, c.seed)
            for fold, (tr, te) in enumerate(self.cf):
                br = self._make_bridge(seed=c.seed + 100 * fold + 1).fit(ds.subset(tr))
                sub = ds.subset(te)
                h = br.predict_h(sub.W, sub.X)
                q = br.predict_q(sub.V, sub.X)
                h_oof[te], q_oof[te] = h, q
                phi[te] = compute_phi(h, q, sub)

        self.phi, self.h_oof, self.q_oof = phi, h_oof, q_oof

        # Stage 2: tau head on D = phi_k - phi_0 (k>=1), plus g0 head on phi_0.
        head_cfg = HeadConfig(device=c.device, seed=c.seed, **c.head_kwargs)
        D = phi[:, 1:] - phi[:, [0]]
        self.tau_head = MLPRegressor(head_cfg).fit(ds.X, D)
        self.g0_head = MLPRegressor(HeadConfig(device=c.device, seed=c.seed + 1, **c.head_kwargs)).fit(ds.X, phi[:, 0])

        self.ate = {k: ate_with_ci(phi[:, k] - phi[:, 0]) for k in range(1, K)}
        self._diag_q(ds)
        return self

    def _diag_q(self, ds: CausalDataset) -> None:
        n = ds.n
        onehot = np.zeros((n, self.K))
        onehot[np.arange(n), ds.T] = 1.0
        w = onehot * self.q_oof
        self.q_diag = {}
        for k in range(self.K):
            wk = w[:, k][ds.T == k]
            ess = (wk.sum() ** 2) / (np.sum(wk ** 2) + 1e-12) if len(wk) else 0.0
            self.q_diag[k] = {
                "mean_indicator_q": float(w[:, k].mean() * n / max((ds.T == k).sum(), 1)),
                "En_1Tk_qk": float(w[:, k].mean()),   # target ~1
                "max_q": float(self.q_oof[:, k].max()),
                "ess": float(ess),
            }

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """(n, K-1) estimated tau_k(x), k=1..K-1."""
        return self.tau_head.predict(X).reshape(X.shape[0], self.K - 1)

    def predict_mu(self, X: np.ndarray) -> np.ndarray:
        """(n, K) mu_hat_k(x) = g0_hat(x) + tau_hat_k(x), tau_0=0."""
        g0 = self.g0_head.predict(X).reshape(-1)
        tau = self.predict_cate(X)
        mu = np.concatenate([g0[:, None], g0[:, None] + tau], axis=1)
        return mu

    def ate_vector(self) -> np.ndarray:
        return np.array([self.ate[k]["ate"] for k in range(1, self.K)])
