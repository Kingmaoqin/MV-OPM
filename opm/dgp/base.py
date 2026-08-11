"""Common container for a generated (or loaded) causal dataset.

Oracle fields (``tau_true``, ``mu_true``, ``cf_*``) are populated by synthetic DGPs
and left ``None`` on real data.  They are used ONLY for evaluation and unit tests,
never inside the estimator (spec Section 9: never tune on causal ground truth).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np


@dataclass
class CausalDataset:
    X: np.ndarray                       # (n, p) covariates
    T: np.ndarray                       # (n,) int treatment index in {0..K-1}
    Y: np.ndarray                       # (n,) outcome
    K: int                              # number of arms
    W: Optional[np.ndarray] = None      # (n, d_w) outcome-inducing proxy
    V: Optional[np.ndarray] = None      # (n, d_v) treatment-inducing proxy

    # --- oracle quantities (synthetic only) ---
    tau_true: Optional[np.ndarray] = None   # (n, K-1) tau_k(x_i), k=1..K-1
    mu_true: Optional[np.ndarray] = None     # (n, K) mu_k(x_i)
    ate_true: Optional[np.ndarray] = None    # (K-1,) E[tau_k]
    # true counterfactual law N(cf_mean[:,k], cf_var) for arm k (E7)
    cf_mean: Optional[np.ndarray] = None     # (n, K)
    cf_var: Optional[float] = None
    h_true: Optional[Callable] = None        # h_true(W, X, T) closed form (E1/E2 only)

    meta: Dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def p(self) -> int:
        return self.X.shape[1]

    @property
    def has_proxies(self) -> bool:
        return self.W is not None and self.V is not None

    def subset(self, idx: np.ndarray) -> "CausalDataset":
        def sl(a):
            return None if a is None else a[idx]
        return CausalDataset(
            X=self.X[idx], T=self.T[idx], Y=self.Y[idx], K=self.K,
            W=sl(self.W), V=sl(self.V),
            tau_true=sl(self.tau_true), mu_true=sl(self.mu_true),
            ate_true=self.ate_true,
            cf_mean=sl(self.cf_mean), cf_var=self.cf_var, h_true=self.h_true,
            meta=dict(self.meta),
        )
