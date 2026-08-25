"""Main synthetic scenarios S1, S2 (spec 3.2) and proxy corruption transforms (3.3)."""
from __future__ import annotations

import numpy as np

from .base import CausalDataset
from .params import get_params

BETA_U = np.array([1.0, -0.8])
CF_VAR = 1.0 + float(BETA_U @ BETA_U)   # 1 + ||beta_U||^2 = 2.64


def _g0(X: np.ndarray) -> np.ndarray:
    return X[:, 0] + 0.5 * X[:, 1] - 0.5 * X[:, 2] ** 2 + np.tanh(X[:, 3])


def _proxies(U: np.ndarray, rng, P) -> tuple[np.ndarray, np.ndarray]:
    """W, V from U per S1 (also reused by S2 and semi-synth)."""
    n = U.shape[0]
    W = U @ P["A_W"].T + 0.3 * np.tanh(U @ P["B_W"].T) + rng.normal(0, 0.5, size=(n, 2))
    V = U @ P["A_V"].T + 0.3 * np.tanh(U @ P["B_V"].T) + rng.normal(0, 0.5, size=(n, 2))
    return W, V


def _tau_s1(Xc: np.ndarray) -> np.ndarray:
    """(n, 3) arm-wise tau_k for k=1..3 using covariate coordinates Xc (>=10 dims)."""
    x1, x2 = Xc[:, 0], Xc[:, 1]
    tau1 = 1.0 + 0.6 * x1
    tau2 = -0.5 + 0.8 * np.abs(x2)
    tau3 = 0.5 * x1 * x2
    return np.stack([tau1, tau2, tau3], axis=1)


def _assemble_s1_like(X_est, Xc, U, rng, P, c_U):
    """Shared S1 assembly given estimator covariates X_est and 'logit' covariates Xc."""
    n = X_est.shape[0]
    # Treatment: logits l_0 = 0; l_k = theta_k'X + c_U gamma_k'U.
    logits = np.zeros((n, 4))
    logits[:, 1:] = Xc @ P["theta"].T + c_U * (U @ P["gamma"].T)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    gum = -np.log(-np.log(rng.uniform(size=(n, 4)) + 1e-12) + 1e-12)
    T = np.argmax(np.log(probs + 1e-12) + gum, axis=1).astype(np.int64)

    tau = _tau_s1(Xc)                       # (n, 3)
    tau_full = np.concatenate([np.zeros((n, 1)), tau], axis=1)  # (n, 4)
    g0 = _g0(Xc)
    eps_y = rng.normal(0, 1.0, size=n)
    bu = U @ BETA_U
    Y = g0 + tau_full[np.arange(n), T] + bu + eps_y

    mu_true = g0[:, None] + tau_full        # mu_k(x) = g0 + tau_k (E[U]=0)
    cf_mean = mu_true.copy()                 # counterfactual mean per arm
    ate_true = tau.mean(axis=0)
    W, V = _proxies(U, rng, P)
    return CausalDataset(
        X=X_est, T=T, Y=Y, K=4, W=W, V=V,
        tau_true=tau, mu_true=mu_true, ate_true=ate_true,
        cf_mean=cf_mean, cf_var=CF_VAR,
        meta={"g0_var": float(g0.var()), "true_propensity": probs},
    )


def generate_s1(n: int = 4000, seed: int = 0, c_U: float = 1.0, p: int = 10) -> CausalDataset:
    """S1: K=4 multinomial. n=4000, p=10 defaults."""
    assert p == 10, "S1 is specified with p=10"
    P = get_params()
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1.0, size=(n, 2))
    X = rng.normal(0, 1.0, size=(n, 10))
    ds = _assemble_s1_like(X, X, U, rng, P, c_U)
    ds.meta.update({"dgp": "S1", "n": n, "seed": seed, "c_U": c_U})
    return ds


def generate_s2(n: int = 4000, seed: int = 0, c_U: float = 1.0, p: int = 10) -> CausalDataset:
    """S2: two binary components combined into K=4 classes (spec 3.2)."""
    assert p == 10, "S2 is specified with p=10"
    P = get_params()
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1.0, size=(n, 2))
    X = rng.normal(0, 1.0, size=(n, 10))
    n = X.shape[0]

    # B_j ~ Bernoulli(sigmoid(theta_j'X + c_U gamma_j'U)); arm = B1 + 2 B2.
    logit_b = X @ P["theta_b"].T + c_U * (U @ P["gamma_b"].T)   # (n, 2)
    pb = 1.0 / (1.0 + np.exp(-logit_b))
    B = (rng.uniform(size=(n, 2)) < pb).astype(np.int64)
    T = (B[:, 0] + 2 * B[:, 1]).astype(np.int64)
    probs = np.column_stack([
        (1 - pb[:, 0]) * (1 - pb[:, 1]),
        pb[:, 0] * (1 - pb[:, 1]),
        (1 - pb[:, 0]) * pb[:, 1],
        pb[:, 0] * pb[:, 1],
    ])

    def tau_of(b1, b2):
        return (b1 * (1.0 + 0.5 * X[:, 0])
                + b2 * (-0.7 + 0.4 * X[:, 1])
                + b1 * b2 * (0.6 - 0.3 * X[:, 0]))

    tau_full = np.stack([tau_of(k % 2, k // 2) for k in range(4)], axis=1)  # (n,4), arm0=0
    g0 = _g0(X)
    eps_y = rng.normal(0, 1.0, size=n)
    bu = U @ BETA_U
    Y = g0 + tau_full[np.arange(n), T] + bu + eps_y

    mu_true = g0[:, None] + tau_full
    W, V = _proxies(U, rng, P)
    return CausalDataset(
        X=X, T=T, Y=Y, K=4, W=W, V=V,
        tau_true=tau_full[:, 1:], mu_true=mu_true, ate_true=tau_full[:, 1:].mean(axis=0),
        cf_mean=mu_true.copy(), cf_var=CF_VAR,
        meta={"dgp": "S2", "n": n, "seed": seed, "c_U": c_U,
              "true_propensity": probs},
    )


# ---------------------------------------------------------------------------
# Proxy corruption transforms (spec 3.3). Applied to W, V at load time.
# ---------------------------------------------------------------------------
def corrupt_proxies(W, V, kind: str, s: float, seed: int = 0):
    """Return corrupted (W, V). ``kind`` in {additive, missing, shift, heavy_tail}."""
    if kind in (None, "none") or s == 0:
        return W, V
    rng = np.random.default_rng(seed + 777)

    def _apply(A):
        A = A.astype(float).copy()
        col_var = A.var(axis=0, keepdims=True)
        col_std = np.sqrt(col_var + 1e-12)
        if kind == "additive":
            A = A + rng.normal(0, 1.0, size=A.shape) * (s * col_std)
        elif kind == "shift":
            A = A + s * 1.0
        elif kind == "heavy_tail":
            # 'replace Gaussian proxy noise with s-scaled Student-t(3)' -> additive
            # heavy-tailed perturbation at load time (see DECISIONS.md).
            t = rng.standard_t(df=3, size=A.shape)
            A = A + s * col_std * t
        elif kind == "missing":
            mask = rng.uniform(size=A.shape) < s
            col_mean = A.mean(axis=0, keepdims=True)
            A = np.where(mask, np.broadcast_to(col_mean, A.shape), A)
        else:
            raise ValueError(f"unknown corruption kind: {kind}")
        return A

    return _apply(W), _apply(V)
