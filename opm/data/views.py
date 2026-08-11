"""Observed-data views that physically hide oracle truth from selection code (MV-OPM §3/§8).

The selector and candidate/validation packages receive ONLY an ObservedDatasetView, which
exposes X, T, Y, W, V, K, n, p and index subsetting — never tau_true, mu_true, ate_true,
cf_mean, h_true, or the latent U. Attempting to read those raises AttributeError.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..dgp.base import CausalDataset

_ORACLE_ATTRS = {"tau_true", "mu_true", "ate_true", "cf_mean", "cf_var", "h_true", "U", "latent_U",
                 "h_true_val", "q_true_val", "W_cat", "V_cat", "U_cat"}


class ObservedDatasetView:
    """Read-only, oracle-free view of a CausalDataset."""

    __slots__ = ("X", "T", "Y", "W", "V", "K", "_n", "_p", "meta")

    def __init__(self, ds: CausalDataset):
        self.X = ds.X
        self.T = ds.T
        self.Y = ds.Y
        self.W = ds.W
        self.V = ds.V
        self.K = ds.K
        self._n = ds.n
        self._p = ds.p
        # copy only non-oracle metadata
        self.meta = {k: v for k, v in (ds.meta or {}).items() if k not in _ORACLE_ATTRS}

    @property
    def n(self) -> int:
        return self._n

    @property
    def p(self) -> int:
        return self._p

    @property
    def has_proxies(self) -> bool:
        return self.W is not None and self.V is not None

    def subset(self, idx: np.ndarray) -> "ObservedDatasetView":
        v = ObservedDatasetView.__new__(ObservedDatasetView)
        v.X = self.X[idx]
        v.T = self.T[idx]
        v.Y = self.Y[idx]
        v.W = None if self.W is None else self.W[idx]
        v.V = None if self.V is None else self.V[idx]
        v.K = self.K
        v._n = len(idx)
        v._p = self._p
        v.meta = dict(self.meta)
        return v

    def __getattr__(self, name):
        if name in _ORACLE_ATTRS:
            raise AttributeError(
                f"ObservedDatasetView hides oracle attribute '{name}' from selection code "
                "(MV-OPM §8). Use opm.eval.oracle in the evaluation package instead.")
        raise AttributeError(name)


def observed(ds: CausalDataset) -> ObservedDatasetView:
    return ObservedDatasetView(ds)
