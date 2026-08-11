"""Common baseline interface: fit(X,T,Y,W,V) / predict_cate(X) / ate_vector() (spec Section 4).

Multi-arm handling: one-vs-reference contrasts unless a package supports multiclass.
Baselines that can only produce ATE (no CATE) set ``cate_available = False``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..dgp.base import CausalDataset
from ..estimator.crossfit import CrossFitter
from ..estimator.pseudo_outcome import compute_phi
from ..estimator.tau_head import ate_with_ci, fit_cate_and_ate


class Baseline:
    name = "baseline"
    cate_available = True

    def fit(self, ds: CausalDataset) -> "Baseline":
        raise NotImplementedError

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def ate_vector(self) -> np.ndarray:
        raise NotImplementedError

    # convenience for uniform experiment calls
    def predict_mu(self, X: np.ndarray) -> Optional[np.ndarray]:
        return None


class CrossFitProximal(Baseline):
    """POR / PIPW / PDR built on cross-fitted bridges + shared Stage-2.

    phi_mode:
      * 'por'  : phi_k = h_k(W,X)                              (proximal outcome regression)
      * 'pipw' : phi_k = 1{T=k} q_k(V,X) Y                     (proximal IPW)
      * 'pdr'  : (STAR) doubly-robust                          (proximal DR)
    """

    def __init__(self, name, bridge_factory, phi_mode, K, n_folds=5, seed=0,
                 device="cpu", head_kwargs=None):
        self.name = name
        self.bridge_factory = bridge_factory
        self.phi_mode = phi_mode
        self.K = K
        self.n_folds = n_folds
        self.seed = seed
        self.device = device
        self.head_kwargs = head_kwargs or {}

    def _phi(self, h, q, sub):
        if self.phi_mode == "por":
            return h.copy()
        if self.phi_mode == "pipw":
            n = sub.n
            onehot = np.zeros((n, self.K)); onehot[np.arange(n), sub.T] = 1.0
            return onehot * q * sub.Y.reshape(-1, 1)
        if self.phi_mode == "pdr":
            return compute_phi(h, q, sub)
        raise ValueError(self.phi_mode)

    def fit(self, ds: CausalDataset) -> "CrossFitProximal":
        n = ds.n
        cf = CrossFitter(n, self.n_folds, self.seed)
        phi = np.zeros((n, self.K))
        for fold, (tr, te) in enumerate(cf):
            br = self.bridge_factory(self.seed + 100 * fold + 1).fit(ds.subset(tr))
            sub = ds.subset(te)
            h = br.predict_h(sub.W, sub.X)
            q = br.predict_q(sub.V, sub.X)
            phi[te] = self._phi(h, q, sub)
        self.phi = phi
        self.tau_head, self.g0_head, self.ate = fit_cate_and_ate(
            ds.X, phi, self.K, self.device, self.seed, self.head_kwargs)
        return self

    def predict_cate(self, X):
        return self.tau_head.predict(X).reshape(X.shape[0], self.K - 1)

    def predict_mu(self, X):
        g0 = self.g0_head.predict(X).reshape(-1)
        tau = self.predict_cate(X)
        return np.concatenate([g0[:, None], g0[:, None] + tau], axis=1)

    def ate_vector(self):
        return np.array([self.ate[k]["ate"] for k in range(1, self.K)])
