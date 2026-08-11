"""Semi-synthetic DGP on real covariates (spec 3.5).

Takes a real covariate table X (e.g. from RHC or the local HAMD covariates via a
plug-in interface), then simulates U, K=4 multinomial T, Y, W, V exactly as in S1,
with the real X entering ``theta_k'X`` and ``g0`` through meta-seeded random
projections of the STANDARDIZED real covariates.  The estimator sees the real X;
the oracle quantities are computed from the projected coordinates.
"""
from __future__ import annotations

import numpy as np

from .base import CausalDataset
from .params import META_SEED, get_params
from .synthetic_main import _assemble_s1_like
from ..utils import standardize_fit, standardize_apply


def _projection(p_real: int) -> np.ndarray:
    """Meta-seeded random projection R^{p_real} -> R^{10}, column-normalized."""
    rng = np.random.default_rng(META_SEED + 100 + p_real)
    P = rng.normal(size=(p_real, 10))
    return P / (np.linalg.norm(P, axis=0, keepdims=True) + 1e-12)


def generate_semisynth(X_real: np.ndarray, seed: int = 0, c_U: float = 1.0) -> CausalDataset:
    P = get_params()
    rng = np.random.default_rng(seed)
    n, p_real = X_real.shape

    mean, std = standardize_fit(X_real)
    Xs = standardize_apply(X_real, mean, std)
    Xc = Xs @ _projection(p_real)                 # (n, 10) projected coordinates

    U = rng.normal(0, 1.0, size=(n, 2))
    ds = _assemble_s1_like(X_real.astype(float), Xc, U, rng, P, c_U)
    ds.meta.update({"dgp": "semisynth", "n": n, "seed": seed, "c_U": c_U, "p_real": p_real})
    return ds
