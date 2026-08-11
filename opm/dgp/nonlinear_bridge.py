"""Nonlinear-bridge DGP (E9): the true outcome/treatment bridges are strongly NONLINEAR,
multi-dimensional functions of the proxies.

Purpose: expose the regime that justifies the kernel-moment NEURAL bridge over a polynomial
sieve. The proxies are multi-dimensional nonlinear (interaction + high-frequency) views of a
2-D latent U, and the confounder enters the outcome through interactions U_1 U_2 and sin.
A polynomial sieve then faces the curse of dimensionality (many cross terms, high variance /
instability) and still cannot represent the interactions/high-frequency terms at feasible
degree, while the neural kernel-moment bridge learns them.  Ignorability methods (X only)
cannot remove the U-confounding at all.

Model (K=2)
-----------
    U ~ N(0, I_2);  X ~ N(0, I_5)
    W = [ U1 + 0.5 U1^2 ,  U2 + 0.5 U2^2 ,  tanh(U1 U2) + 0.5 sin(2 U1) ] + eps_w  (R^3)
    V = [ U1 + 0.4 sin(3 U1) ,  U2 + 0.4 tanh(2 U2) ,  0.7 U1 U2 ] + eps_v          (R^3)
        eps ~ N(0, 0.4^2)
    T = 1{ 0.8 U1 + 0.7 U2 + 0.5 X_1 + eps_t > 0 }
    Y = g0(X) + tau(X) T + phi(U) + eps_y
      g0(x) = beta_x' x, beta_x=(1,-1,0.5,0,0);  tau(x) = 2 + 0.8 x_1  (ATE=2)
      phi(U) = U1 U2 + 0.6 (U1^2 - 1) + 0.8 sin(U1 + U2)      (E[phi]=0)
"""
from __future__ import annotations

import numpy as np

from .base import CausalDataset

BETA_X = np.array([1.0, -1.0, 0.5, 0.0, 0.0])
ATE = 2.0


def _phi(U):
    return U[:, 0] * U[:, 1] + 0.6 * (U[:, 0] ** 2 - 1.0) + 0.8 * np.sin(U[:, 0] + U[:, 1])


def generate(n: int, seed: int) -> CausalDataset:
    rng = np.random.default_rng(seed)
    U = rng.normal(0.0, 1.0, size=(n, 2))
    X = rng.normal(0.0, 1.0, size=(n, 5))
    u1, u2 = U[:, 0], U[:, 1]
    W = np.stack([u1 + 0.5 * u1 ** 2,
                  u2 + 0.5 * u2 ** 2,
                  np.tanh(u1 * u2) + 0.5 * np.sin(2 * u1)], axis=1) + rng.normal(0, 0.4, size=(n, 3))
    V = np.stack([u1 + 0.4 * np.sin(3 * u1),
                  u2 + 0.4 * np.tanh(2 * u2),
                  0.7 * u1 * u2], axis=1) + rng.normal(0, 0.4, size=(n, 3))
    eps_t = rng.normal(0, 1.0, size=n)
    T = (0.8 * u1 + 0.7 * u2 + 0.5 * X[:, 0] + eps_t > 0).astype(np.int64)

    g0 = X @ BETA_X
    tau = 2.0 + 0.8 * X[:, 0]
    Y = g0 + tau * T + _phi(U) + rng.normal(0, 1.0, size=n)

    tau_true = tau.reshape(-1, 1)
    mu_true = np.stack([g0, g0 + tau], axis=1)
    return CausalDataset(
        X=X, T=T, Y=Y, K=2, W=W, V=V,
        tau_true=tau_true, mu_true=mu_true, ate_true=np.array([ATE]),
        meta={"dgp": "nonlinear_bridge", "n": n, "seed": seed},
    )
