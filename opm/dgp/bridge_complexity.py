"""Prespecified symmetric bridge-complexity benchmark for MV-OPM Round 2.

The four families alter structural functions without consulting candidate performance. They are
not hand-tuned to make a particular solver win. W and V use independent noise conditional on
(U, X), treatment depends on (U, X), and outcome depends on (T, U, X).
"""
from __future__ import annotations

import numpy as np

from .base import CausalDataset

FAMILIES = ("L", "Q", "N", "M")
PROXY_NOISE = {"low": 0.35, "high": 1.00}


def _structural_terms(family: str, U: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return centered outcome-confounding and treatment-confounding scores."""
    u1, u2 = U[:, 0], U[:, 1]
    x1, x2 = X[:, 0], X[:, 1]
    if family == "L":
        f_h = 0.8 * u1 - 0.6 * u2 + 0.20 * x1
        f_q = 0.7 * u1 + 0.5 * u2 + 0.20 * x2
    elif family == "Q":
        f_h = 0.55 * (u1 * u1 - 1.0) - 0.45 * (u2 * u2 - 1.0) + 0.35 * u1 * u2 + 0.20 * u1 * x1
        f_q = 0.50 * (u1 * u1 - 1.0) + 0.30 * u1 * u2 - 0.35 * (u2 * u2 - 1.0) + 0.20 * u2 * x2
    elif family == "N":
        f_h = 0.90 * np.tanh(1.2 * u1) - 0.55 * np.sin(0.9 * u2) + 0.25 * np.tanh(u1 * x1)
        f_q = 0.75 * np.sin(0.8 * u1) + 0.55 * np.tanh(1.1 * u2) + 0.20 * np.sin(u2 * x2)
    elif family == "M":
        # Deliberately asymmetric but not extreme: nonlinear h side, low-order q side.
        f_h = 0.90 * np.tanh(1.2 * u1) - 0.55 * np.sin(0.9 * u2) + 0.30 * np.tanh(u1 * x1)
        f_q = 0.70 * u1 + 0.45 * u2 + 0.20 * x2
    else:
        raise ValueError(f"unknown family {family!r}")
    return f_h, f_q


def _proxy_signal(family: str, U: np.ndarray, X: np.ndarray, side: str) -> np.ndarray:
    u1, u2 = U[:, 0], U[:, 1]
    if family == "L" or (family == "M" and side == "q"):
        return np.column_stack([u1, u2, 0.65 * u1 - 0.45 * u2])
    if family == "Q":
        return np.column_stack([u1, u2, 0.55 * (u1 * u1 - 1.0) + 0.35 * u1 * u2])
    # N and the h side of M use the same moderate smooth non-polynomial channel.
    return np.column_stack([u1, u2, 0.70 * np.tanh(u1) + 0.45 * np.sin(0.8 * u2)])


def generate(
    n: int,
    seed: int,
    *,
    family: str,
    proxy_noise: str = "low",
    p: int = 10,
    c_U: float = 1.0,
) -> CausalDataset:
    """Generate one frozen binary-treatment complexity regime."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if proxy_noise not in PROXY_NOISE:
        raise ValueError(f"proxy_noise must be one of {tuple(PROXY_NOISE)}")
    if p < 4:
        raise ValueError("p must be at least four")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    U = rng.normal(size=(n, 2))
    f_h, f_q = _structural_terms(family, U, X)

    # A bounded linear probability avoids extreme, solver-favoring inverse propensities.
    e = np.clip(0.5 + 0.14 * c_U * f_q, 0.10, 0.90)
    T = (rng.uniform(size=n) < e).astype(np.int64)
    tau = 1.0 + 0.50 * np.tanh(X[:, 0]) + 0.25 * X[:, 1]
    g0 = 0.55 * X[:, 0] - 0.40 * X[:, 1] + 0.25 * X[:, 2] * X[:, 3]
    Y = g0 + tau * T + c_U * f_h + rng.normal(0, 1.0, size=n)

    sigma = PROXY_NOISE[proxy_noise]
    # Independent noises preserve W ⟂ V conditional on U,X.
    W = _proxy_signal(family, U, X, "h") + rng.normal(0, sigma, size=(n, 3))
    V = _proxy_signal(family, U, X, "q") + rng.normal(0, sigma, size=(n, 3))
    mu = np.column_stack([g0, g0 + tau])
    return CausalDataset(
        X=X,
        T=T,
        Y=Y,
        K=2,
        W=W,
        V=V,
        tau_true=tau[:, None],
        mu_true=mu,
        ate_true=np.array([float(np.mean(tau))]),
        cf_mean=mu,
        cf_var=float(1.0 + c_U * c_U * np.var(f_h)),
        meta={
            "dgp": "bridge_complexity",
            "family": family,
            "proxy_noise": proxy_noise,
            "proxy_noise_sd": sigma,
            "n": n,
            "p": p,
            "seed": seed,
            "c_U": c_U,
            "treatment_probability_min": float(np.min(e)),
            "treatment_probability_max": float(np.max(e)),
        },
    )
