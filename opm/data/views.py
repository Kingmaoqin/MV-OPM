"""Observed-data views that physically hide oracle truth from selection code (MV-OPM §3/§8).

The selector and candidate/validation packages receive ONLY an ObservedDatasetView, which
exposes X, T, Y, W, V, K, n, p and index subsetting — never tau_true, mu_true, ate_true,
cf_mean, h_true, or the latent U. Attempting to read those raises AttributeError.
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np

from ..dgp.base import CausalDataset

_ORACLE_ATTRS = {"tau_true", "mu_true", "ate_true", "cf_mean", "cf_var", "h_true", "U", "latent_U",
                 "h_true_val", "q_true_val", "W_cat", "V_cat", "U_cat"}
def _readonly_copy(value):
    if value is None:
        return None
    contiguous = np.ascontiguousarray(value)
    if contiguous.dtype.hasobject:
        raise TypeError("observed-data arrays must not use object dtype")
    # ``bytes`` owns immutable storage. Unlike an owning ndarray with writeable=False, a consumer
    # cannot call setflags(write=True) to recover a mutable alias.
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


class ObservedDatasetView:
    """Immutable, oracle-free observed-data projection of a ``CausalDataset``."""

    __slots__ = ("_X", "_T", "_Y", "_W", "_V", "_K", "_n", "_p", "_meta", "_sealed")

    def __init__(self, ds: CausalDataset):
        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "_X", _readonly_copy(ds.X))
        object.__setattr__(self, "_T", _readonly_copy(ds.T))
        object.__setattr__(self, "_Y", _readonly_copy(ds.Y))
        object.__setattr__(self, "_W", _readonly_copy(ds.W))
        object.__setattr__(self, "_V", _readonly_copy(ds.V))
        object.__setattr__(self, "_K", int(ds.K))
        object.__setattr__(self, "_n", int(ds.n))
        object.__setattr__(self, "_p", int(ds.p))
        # Selection currently consumes no metadata. Expose an empty mapping rather than a
        # supposedly safe key whose value could carry an oracle array or nested mutable object.
        object.__setattr__(self, "_meta", MappingProxyType({}))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("ObservedDatasetView is immutable")
        object.__setattr__(self, name, value)

    @property
    def X(self):
        return self._X

    @property
    def T(self):
        return self._T

    @property
    def Y(self):
        return self._Y

    @property
    def W(self):
        return self._W

    @property
    def V(self):
        return self._V

    @property
    def K(self):
        return self._K

    @property
    def meta(self):
        return self._meta

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
        object.__setattr__(v, "_sealed", False)
        object.__setattr__(v, "_X", _readonly_copy(self.X[idx]))
        object.__setattr__(v, "_T", _readonly_copy(self.T[idx]))
        object.__setattr__(v, "_Y", _readonly_copy(self.Y[idx]))
        object.__setattr__(v, "_W", None if self.W is None else _readonly_copy(self.W[idx]))
        object.__setattr__(v, "_V", None if self.V is None else _readonly_copy(self.V[idx]))
        object.__setattr__(v, "_K", self.K)
        object.__setattr__(v, "_n", len(idx))
        object.__setattr__(v, "_p", self._p)
        object.__setattr__(v, "_meta", self.meta)
        object.__setattr__(v, "_sealed", True)
        return v

    def __getattr__(self, name):
        if name in _ORACLE_ATTRS:
            raise AttributeError(
                f"ObservedDatasetView hides oracle attribute '{name}' from selection code "
                "(MV-OPM §8). Use opm.eval.oracle in the evaluation package instead.")
        raise AttributeError(name)


def observed(ds: CausalDataset) -> ObservedDatasetView:
    return ObservedDatasetView(ds)
