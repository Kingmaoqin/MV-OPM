"""Polynomial-sieve proximal bridges via two-stage least squares (spec 3.4 / Section 4).

Solves the same integral equations as the kernel bridges but with a linear sieve:
  h_k(W,X) = phi_h(W,X)'beta_k,  instruments psi(V,X)   (2SLS on samples T=k)
  q_k(V,X) = phi_q(V,X)'alpha_k, instruments xi(W,X)     (linear moment solve)

Doubles as the kernel-vs-sieve ablation (E6a) and provides the POR/PIPW/PDR baselines
and the linear-POR milestone gate (degree=1).  Same predict_h/predict_q interface as
KernelBridges so it drops straight into the OPM pipeline (mode='sieve').
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ..dgp.base import CausalDataset


class _Basis:
    """Standardize raw inputs -> polynomial expansion -> standardize features -> add bias.

    Standardizing AFTER the polynomial expansion is essential: cross terms x_i·x_j have
    very different scales and otherwise make the 2SLS normal equations ill-conditioned.
    """

    def __init__(self, degree: int):
        self.scaler_in = StandardScaler()
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.scaler_out = StandardScaler()

    MAX_FEATURES = 6000    # guard: prevents polynomial blow-up (e.g. deg-3 on 69-dim -> ~57k feats)

    def fit(self, A):
        n_in = A.shape[1]
        if getattr(self.poly, "degree", 1) >= 2:
            from math import comb
            est = comb(n_in + self.poly.degree, self.poly.degree) - 1
            if est > self.MAX_FEATURES:
                raise ValueError(f"polynomial sieve degree {self.poly.degree} infeasible for "
                                 f"{n_in}-dim input (~{est} features > {self.MAX_FEATURES}); "
                                 "use a lower degree / kernel candidate on high-dim covariates")
        P = self.poly.fit_transform(self.scaler_in.fit_transform(A))
        self.scaler_out.fit(P)
        return self

    def transform(self, A):
        P = self.scaler_out.transform(self.poly.transform(self.scaler_in.transform(A)))
        P = np.clip(P, -8.0, 8.0)   # tame heavy-tailed high-degree features (e.g. from U^2)
        return np.concatenate([np.ones((P.shape[0], 1)), P], axis=1)


class SieveBridges:
    def __init__(self, K: int, degree: int = 2, q_max: float = 50.0, ridge: float = 1e-4):
        self.K = K
        self.degree = degree
        self.q_max = q_max
        self.ridge = ridge
        self._fitted = False

    def fit(self, ds: CausalDataset) -> "SieveBridges":
        W, V, X, Y, T = ds.W, ds.V, ds.X, ds.Y, ds.T
        self.b_h = _Basis(self.degree).fit(np.concatenate([W, X], axis=1))    # phi_h(W,X)
        self.b_zh = _Basis(self.degree).fit(np.concatenate([V, X], axis=1))   # psi(V,X)
        self.b_q = _Basis(self.degree).fit(np.concatenate([V, X], axis=1))    # phi_q(V,X)
        self.b_zq = _Basis(self.degree).fit(np.concatenate([W, X], axis=1))   # xi(W,X)

        Phi = self.b_h.transform(np.concatenate([W, X], axis=1))
        Psi = self.b_zh.transform(np.concatenate([V, X], axis=1))
        PhiQ = self.b_q.transform(np.concatenate([V, X], axis=1))
        Xi = self.b_zq.transform(np.concatenate([W, X], axis=1))

        self.beta = {}
        self.alpha = {}
        for k in range(self.K):
            idx = T == k
            self.beta[k] = self._twosls(Phi[idx], Psi[idx], Y[idx])
            self.alpha[k] = self._q_solve(PhiQ, Xi, idx)
        self._fitted = True
        return self

    def _twosls(self, Phi, Psi, y):
        if Phi.shape[0] < Phi.shape[1] + 2:   # degenerate/rare arm
            return np.linalg.lstsq(Phi, y, rcond=None)[0]
        PtP = Psi.T @ Psi + self.ridge * np.eye(Psi.shape[1])
        Pinv = np.linalg.solve(PtP, Psi.T)            # (m_z, n)
        Phi_hat = Psi @ (Pinv @ Phi)                   # projected regressors
        A = Phi_hat.T @ Phi + self.ridge * np.eye(Phi.shape[1])
        b = Phi_hat.T @ y
        return np.linalg.solve(A, b)

    def _q_solve(self, PhiQ, Xi, idx):
        n = PhiQ.shape[0]
        mask = idx.astype(float)
        # A alpha = b:  A = E[1{T=k} xi phi_q'],  b = E[xi]
        A = (Xi * mask[:, None]).T @ PhiQ / n
        b = Xi.mean(axis=0)
        AtA = A.T @ A + self.ridge * np.eye(A.shape[1])
        return np.linalg.solve(AtA, A.T @ b)

    def predict_h(self, W, X) -> np.ndarray:
        Phi = self.b_h.transform(np.concatenate([W, X], axis=1))
        return np.column_stack([Phi @ self.beta[k] for k in range(self.K)])

    def predict_q(self, V, X) -> np.ndarray:
        PhiQ = self.b_q.transform(np.concatenate([V, X], axis=1))
        q = np.column_stack([PhiQ @ self.alpha[k] for k in range(self.K)])
        return np.clip(q, 0.0, self.q_max)
