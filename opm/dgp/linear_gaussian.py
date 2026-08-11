"""E1/E2 DGP: linear-Gaussian with a CLOSED-FORM outcome bridge (spec 3.1).

Model
-----
    U ~ N(0, 1);  X ~ N(0, I_5)  (independent of U)
    W = 1.2 U + eps_w,  eps_w ~ N(0, 0.5^2)
    V = 1.0 U + eps_v,  eps_v ~ N(0, 0.5^2)
    T = 1{ 0.8 U + 0.5 X_1 + eps_t > 0 },  eps_t ~ N(0, 1)      (binary, K=2)
    Y = 2.0 T + 1.5 U + beta_x' X + eps_y,  beta_x = (1,-1,0.5,0,0), eps_y ~ N(0,1)

Ground truth:  ATE = 2.0;  tau(x) = 2.0 (constant).

Closed-form outcome bridge:
    h_t(W, X) = 2 t + (1.5/1.2) W + beta_x' X = 2 t + 1.25 W + beta_x' X.

Verification (docstring proof of B1 at arm t):
    E[Y - h_t | U, X, T=t]
      = E[2t + 1.5 U + beta_x'X + eps_y - (2t + 1.25 W + beta_x'X) | U, X, T=t]
      = 1.5 U - 1.25 E[W | U]            (eps_y mean 0; W indep of T given U)
      = 1.5 U - 1.25 * (1.2 U)
      = 1.5 U - 1.5 U = 0.               QED
No closed form is required for q; it is tested via its moment identity (Section 7).
"""
from __future__ import annotations

import numpy as np

from .base import CausalDataset

BETA_X = np.array([1.0, -1.0, 0.5, 0.0, 0.0])
ATE = 2.0


def h_true(W: np.ndarray, X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Closed-form outcome bridge h_T(W, X). Shapes: W (n,1), X (n,5), T (n,)."""
    w = W[:, 0]
    return 2.0 * T + 1.25 * w + X @ BETA_X


def generate(n: int, seed: int, tau_slope: float = 0.0, tx_coef: float = 0.5) -> CausalDataset:
    """DGP 3.1. ``tau_slope`` adds a HETEROGENEOUS effect tau(x)=2+tau_slope*x_1 (default 0
    keeps the constant tau=2 of the unit-test anchor). E[tau]=2 either way, so ATE=2.
    ``tx_coef`` is the treatment's dependence on x_1 (default 0.5 = spec 3.1); E10 sets it to
    0 so treatment depends only on U — overlap is then uniform across the effect modifier x_1,
    isolating CATE-inference coverage from overlap artefacts. Used by E10."""
    rng = np.random.default_rng(seed)
    U = rng.normal(0.0, 1.0, size=n)
    X = rng.normal(0.0, 1.0, size=(n, 5))
    W = (1.2 * U + rng.normal(0.0, 0.5, size=n)).reshape(-1, 1)
    V = (1.0 * U + rng.normal(0.0, 0.5, size=n)).reshape(-1, 1)
    eps_t = rng.normal(0.0, 1.0, size=n)
    T = (0.8 * U + tx_coef * X[:, 0] + eps_t > 0).astype(np.int64)
    eps_y = rng.normal(0.0, 1.0, size=n)
    tau_x = ATE + tau_slope * X[:, 0]
    Y = tau_x * T + 1.5 * U + X @ BETA_X + eps_y

    tau_true = tau_x.reshape(-1, 1)
    # mu_k(x) = E[Y(t_k)|X=x] = tau(x) t_k + beta_x'X  (E[U]=0, U indep X).
    base = X @ BETA_X
    mu_true = np.stack([base, base + tau_x], axis=1)  # arms 0,1
    return CausalDataset(
        X=X, T=T, Y=Y, K=2, W=W, V=V,
        tau_true=tau_true, mu_true=mu_true, ate_true=np.array([ATE]),
        h_true=(h_true if tau_slope == 0.0 else None),   # closed form only for constant tau
        meta={"dgp": "linear_gaussian", "n": n, "seed": seed, "tau_slope": tau_slope},
    )
