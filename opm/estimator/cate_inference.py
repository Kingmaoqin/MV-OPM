"""Pointwise inference for a 1-D CATE surface from DR pseudo-outcomes.

Under the proximal-DR identification the out-of-fold contrast D_i = phi_1(O_i) - phi_0(O_i)
is a noisy UNBIASED signal for tau(X_i) = E[D_i | X_i] (Theorem 1). A local-linear regression
of D on a scalar effect modifier m = x_j therefore estimates tau(m) with a valid pointwise
CI: local-linear is unbiased for a locally-linear target (no boundary bias), and the
heteroskedasticity-robust sandwich gives the variance.  This is the DR-learner pointwise
inference of Kennedy (2020), applied to the proximal pseudo-outcome.

Note: the bridge-estimation error enters only at second order (the DR / Neyman-orthogonality
property), so with cross-fitted, consistent bridges the nominal coverage is attained — which
is exactly what E10 checks empirically.
"""
from __future__ import annotations

import numpy as np


def local_linear_ci(m: np.ndarray, D: np.ndarray, x0: float, bw: float, z: float = 1.96):
    """Local-linear estimate of E[D | m = x0] with a robust pointwise CI.

    m, D: (n,) scalar modifier and pseudo-outcome contrast. Returns
    (tau_hat, ci_low, ci_high, ess) where ess is the kernel effective sample size.
    """
    u = (m - x0) / bw
    w = np.exp(-0.5 * u * u)                       # Gaussian kernel weights
    ess = float(w.sum() ** 2 / np.sum(w ** 2 + 1e-12))
    Z = np.column_stack([np.ones_like(m), (m - x0)])  # local-linear design
    WZ = Z * w[:, None]
    A = Z.T @ WZ                                    # Z' W Z  (2x2)
    b = WZ.T @ D                                     # Z' W D
    try:
        Ainv = np.linalg.inv(A + 1e-8 * np.eye(2))
    except np.linalg.LinAlgError:
        return float(np.nan), float(np.nan), float(np.nan), ess
    beta = Ainv @ b
    tau_hat = float(beta[0])
    # heteroskedasticity-robust sandwich: Var(beta) = Ainv (sum w_i^2 r_i^2 Z_i Z_i') Ainv
    r = D - Z @ beta
    meat = (Z * (w * r)[:, None]).T @ (Z * (w * r)[:, None])   # sum w_i^2 r_i^2 Z_i Z_i'
    Vbeta = Ainv @ meat @ Ainv
    se = float(np.sqrt(max(Vbeta[0, 0], 0.0)))
    return tau_hat, tau_hat - z * se, tau_hat + z * se, ess


def silverman_bw(m: np.ndarray, scale: float = 1.5) -> float:
    """Silverman rule bandwidth, scaled up (local-linear is unbiased for linear tau, so a
    wider window only tightens the CI without adding bias)."""
    n = len(m)
    s = np.std(m)
    iqr = np.subtract(*np.percentile(m, [75, 25]))
    sigma = min(s, iqr / 1.349) if iqr > 0 else s
    return float(scale * 0.9 * sigma * n ** (-1 / 5))
