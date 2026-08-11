"""No-proxy degradation mode (spec 1.8): plug-in DR-learner nuisances.

  h_k -> m_hat_k(X)         (per-arm outcome regression)
  q_k -> 1 / pi_hat_k(X)    (softmax propensity), pi clipped to [1/q_max, 1]

Same predict_h(W,X)/predict_q(V,X) interface as KernelBridges, so the rest of the
pipeline (cross-fitting, (STAR), Stage 2) is identical.  W and V are ignored here —
this is exactly what runs on real data without proxies.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from ..dgp.base import CausalDataset


class PlugInBridges:
    """Plug-in DR nuisances. ``linear=True`` uses Ridge + multinomial logistic — the
    interpretable 'linear plug-in' used as the E4 milestone gate on real data."""

    def __init__(self, K: int, q_max: float = 50.0, seed: int = 0, linear: bool = False):
        self.K = K
        self.q_max = q_max
        self.seed = seed
        self.linear = linear
        self._fitted = False

    def _make_reg(self, k):
        if self.linear:
            return Ridge(alpha=1.0)
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                             max_depth=3, random_state=self.seed + k)

    def fit(self, ds: CausalDataset) -> "PlugInBridges":
        X, T, Y = ds.X, ds.T, ds.Y
        self._scaler = StandardScaler().fit(X) if self.linear else None
        Xf = self._scaler.transform(X) if self.linear else X
        self.models_m = {}
        for k in range(self.K):
            m = self._make_reg(k)
            idx = T == k
            if idx.sum() >= 10:
                m.fit(Xf[idx], Y[idx])
            else:  # degenerate arm: fall back to global mean
                m = None
                self._global_mean = float(Y.mean())
            self.models_m[k] = m
        self.present = np.unique(T)
        if self.linear:
            self.clf = LogisticRegression(max_iter=1000, C=1.0)
        else:
            self.clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                      max_depth=3, random_state=self.seed)
        self.clf.fit(Xf, T)
        # Stabilized (Hajek) weights: normalize so En_train[1{T=k} q_k] = 1 per arm. The
        # proximal q-bridge enforces this by construction (moment B2); the plug-in path does
        # not, so we self-normalize (standard stabilized IPW). See DECISIONS.md.
        n = len(T)
        q_raw = self._raw_q(X)
        onehot = np.zeros((n, self.K)); onehot[np.arange(n), T] = 1.0
        self.norm_k = np.array([max((onehot[:, k] * q_raw[:, k]).mean(), 1e-6) for k in range(self.K)])
        self._fitted = True
        return self

    def _raw_q(self, X):
        proba = self.clf.predict_proba(self._xf(X))
        pi = np.full((X.shape[0], self.K), 1.0 / self.q_max)
        for j, k in enumerate(self.clf.classes_):
            pi[:, int(k)] = proba[:, j]
        pi = np.clip(pi, 1.0 / self.q_max, 1.0)
        return 1.0 / pi

    def _xf(self, X):
        return self._scaler.transform(X) if self.linear else X

    def predict_h(self, W, X) -> np.ndarray:   # W ignored (no proxy)
        Xf = self._xf(X)
        out = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            m = self.models_m[k]
            out[:, k] = getattr(self, "_global_mean", 0.0) if m is None else m.predict(Xf)
        return out

    def predict_q(self, V, X) -> np.ndarray:   # V ignored (no proxy)
        q = self._raw_q(X) / self.norm_k[None, :]         # stabilized (Hajek) weights
        return np.clip(q, 0.0, self.q_max)
