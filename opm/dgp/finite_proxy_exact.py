"""Exact-bridge proximal DGP (MV-OPM §4): both h_true and q_true are solvable in closed form.

Finite discrete latent confounder U with full-rank proxy channels W, V. Treatment depends on U
(hidden confounding); outcome depends on U and observed X. Because W, V are discrete and the
U-channels are full-rank (completeness), BOTH identifying bridges reduce to finite linear
systems and are computed EXACTLY from the population matrices — replacing the old E2 setup that
had to use an estimated q as a stand-in for q_true.

Structure (K=2)
---------------
    U ~ Cat(pi_U),   u in {0..dU-1}
    W | U ~ Cat(B_W[u,:]),   V | U ~ Cat(B_V[u,:])          (W ⟂ V | U; full column rank)
    T | U ~ Bernoulli(e_u),   e_u = clip(0.5 + c_U*(e0_u-0.5), .05, .95)   (confounded, X-free)
    X ~ N(0, I_p),  independent of U
    Y = tau(X)*T + c_U * f_U[U] + beta' X + N(0, sigma^2),   tau(X) = 2 + 1.0*X_1  (ATE=2)

Outcome bridge   h_t(W,X) = tau(X) t + beta'X + h~_t[W],   with h~_t solving  M_t h~_t = r_t
Treatment bridge q_t(V)   = q_t[V],                        with q_t   solving  N_t q_t = 1
(derivations in the module; verified numerically by tests to satisfy E[Y-h|V,X,T=t]=0 and
E[1{T=t}q|W]=1, and to recover ATE=2 via the (STAR) pseudo-outcome).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import CausalDataset

BETA_X = np.array([1.0, -1.0, 0.5, 0.0, 0.0])
ATE = 2.0


@dataclass
class ExactProxyParams:
    dU: int = 3
    dW: int = 3
    dV: int = 3
    p: int = 5
    sigma: float = 1.0
    c_U: float = 1.0
    seed_params: int = 20240  # meta-seed for the (once-generated) full-rank matrices

    pi_U: np.ndarray = field(default=None)
    B_W: np.ndarray = field(default=None)
    B_V: np.ndarray = field(default=None)
    e0: np.ndarray = field(default=None)      # base P(T=1|U) before c_U scaling
    f_U: np.ndarray = field(default=None)     # outcome U-effect (centered)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed_params)
        self.pi_U = np.full(self.dU, 1.0 / self.dU)
        # full-rank, well-conditioned channels: identity-dominant + noise, row-normalized
        self.B_W = _rowstoch(np.eye(self.dU, self.dW) * 3.0 + rng.uniform(0.3, 1.0, (self.dU, self.dW)))
        self.B_V = _rowstoch(np.eye(self.dU, self.dV) * 3.0 + rng.uniform(0.3, 1.0, (self.dU, self.dV)))
        self.e0 = np.linspace(0.25, 0.75, self.dU)               # U-dependent selection
        self.f_U = np.linspace(-1.0, 1.0, self.dU)               # U-dependent outcome shift (mean 0)


def _rowstoch(M):
    return M / M.sum(axis=1, keepdims=True)


def _e_of_u(prm: ExactProxyParams) -> np.ndarray:
    return np.clip(0.5 + prm.c_U * (prm.e0 - 0.5), 0.05, 0.95)   # P(T=1|U=u)


def _Pt(prm: ExactProxyParams) -> np.ndarray:
    """(dU, 2) matrix P(T=t|U=u)."""
    e = _e_of_u(prm)
    return np.stack([1.0 - e, e], axis=1)


def population_bridges(prm: ExactProxyParams):
    """Return exact h~_t[w] (2,dW) and q_t[v] (2,dV) from population equations."""
    pi, BW, BV, Pt = prm.pi_U, prm.B_W, prm.B_V, _Pt(prm)
    dU, dW, dV = prm.dU, prm.dW, prm.dV
    h_tilde = np.zeros((2, dW))
    q = np.zeros((2, dV))
    for t in range(2):
        # P(U=u | V=v, T=t) ∝ pi_u B_V[u,v] Pt[u,t]
        PUvt = (pi[:, None] * BV * Pt[:, [t]])                 # (dU, dV) unnormalized
        PUvt = PUvt / PUvt.sum(0, keepdims=True)               # normalize over u per v
        M_t = (BW.T @ PUvt).T                                  # (dV, dW): M[v,w]=Σ_u BW[u,w] P(U=u|V=v,T=t)
        r_t = (PUvt.T @ (prm.c_U * prm.f_U))                   # (dV,): Σ_u P(U|V,T) c_U f_U
        h_tilde[t] = np.linalg.lstsq(M_t, r_t, rcond=None)[0]
        # P(U=u | W=w) ∝ pi_u B_W[u,w]
        PUw = (pi[:, None] * BW); PUw = PUw / PUw.sum(0, keepdims=True)   # (dU, dW)
        # N_t[w,v] = Σ_u B_V[u,v] Pt[u,t] P(U=u|W=w)
        N_t = (BV * Pt[:, [t]]).T @ PUw                        # (dV, dW)
        N_t = N_t.T                                            # (dW, dV)
        q[t] = np.linalg.lstsq(N_t, np.ones(dW), rcond=None)[0]
    return h_tilde, q


def generate(n: int, seed: int, c_U: float = 1.0, params: ExactProxyParams = None) -> CausalDataset:
    prm = params or ExactProxyParams(c_U=c_U)
    if params is not None and c_U != prm.c_U:
        prm = ExactProxyParams(dU=prm.dU, dW=prm.dW, dV=prm.dV, p=prm.p, sigma=prm.sigma,
                               c_U=c_U, seed_params=prm.seed_params)
    rng = np.random.default_rng(seed)
    U = rng.choice(prm.dU, size=n, p=prm.pi_U)
    Wc = np.array([rng.choice(prm.dW, p=prm.B_W[u]) for u in U])
    Vc = np.array([rng.choice(prm.dV, p=prm.B_V[u]) for u in U])
    e = _e_of_u(prm)
    T = (rng.uniform(size=n) < e[U]).astype(np.int64)
    X = rng.normal(0, 1.0, size=(n, prm.p))
    tau_x = ATE + 1.0 * X[:, 0]
    Y = tau_x * T + prm.c_U * prm.f_U[U] + X @ BETA_X[:prm.p] + rng.normal(0, prm.sigma, size=n)

    h_tilde, q = population_bridges(prm)
    g0 = X @ BETA_X[:prm.p]
    mu0 = g0 + prm.c_U * prm.f_U.mean()          # E[phi(U)] centered ~ 0 (f_U mean 0)
    mu1 = mu0 + tau_x
    # exact per-sample bridge values (oracle; hidden by ObservedDatasetView)
    h_true_val = np.stack([g0 + h_tilde[0, Wc], g0 + tau_x + h_tilde[1, Wc]], axis=1)
    q_true_val = np.stack([q[0, Vc], q[1, Vc]], axis=1)

    W = Wc.reshape(-1, 1).astype(float)
    V = Vc.reshape(-1, 1).astype(float)
    ds = CausalDataset(
        X=X, T=T, Y=Y, K=2, W=W, V=V,
        tau_true=tau_x.reshape(-1, 1), mu_true=np.stack([mu0, mu1], axis=1),
        ate_true=np.array([ATE]),
        meta={"dgp": "finite_proxy_exact", "n": n, "seed": seed, "c_U": prm.c_U,
              "h_true_val": h_true_val, "q_true_val": q_true_val,
              "W_cat": Wc, "V_cat": Vc, "U_cat": U},
    )
    return ds
