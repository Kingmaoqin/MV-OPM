"""Meta-seeded random matrices for S1/S2/semi-synthetic DGPs.

All matrices are drawn ONCE with meta-seed 12345, cached under ``opm/dgp/params/``,
and reused across per-run seeds (spec Section 3 preamble).
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np

META_SEED = 12345
_PARAM_DIR = os.path.join(os.path.dirname(__file__), "params")
_CACHE = os.path.join(_PARAM_DIR, "synthetic_meta.npz")

_KEYS = ["A_W", "B_W", "A_V", "B_V", "theta", "gamma", "theta_b", "gamma_b"]


def _colnorm(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=0, keepdims=True) + 1e-12)


def _build() -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(META_SEED)
    A_W = _colnorm(rng.normal(size=(2, 2)))
    B_W = _colnorm(rng.normal(size=(2, 2)))
    A_V = _colnorm(rng.normal(size=(2, 2)))
    B_V = _colnorm(rng.normal(size=(2, 2)))
    theta = rng.normal(0.0, 0.3, size=(3, 10))     # S1 arms k=1..3
    gamma = rng.normal(0.0, 1.0, size=(3, 2))
    theta_b = rng.normal(0.0, 0.3, size=(2, 10))    # S2 binary components j=1,2
    gamma_b = rng.normal(0.0, 1.0, size=(2, 2))
    return dict(A_W=A_W, B_W=B_W, A_V=A_V, B_V=B_V,
                theta=theta, gamma=gamma, theta_b=theta_b, gamma_b=gamma_b)


def get_params() -> Dict[str, np.ndarray]:
    if os.path.exists(_CACHE):
        with np.load(_CACHE) as f:
            return {k: f[k] for k in _KEYS}
    os.makedirs(_PARAM_DIR, exist_ok=True)
    params = _build()
    np.savez(_CACHE, **params)
    return params
