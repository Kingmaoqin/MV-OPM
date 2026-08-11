"""Diagnostics (spec 6.1) — also drive bridge early stopping and E8 validity study."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression

from ..dgp.base import CausalDataset
from ..utils import standardize_apply, standardize_fit


def bridge_residual_score(R: np.ndarray, inst: np.ndarray, bw: float,
                          n_rff: int = 100, seed: int = 0) -> float:
    """max_j |En[R_i g_j(inst)]| / sd(R) over 100 random Fourier features."""
    inst = standardize_apply(inst, *standardize_fit(inst))
    rng = np.random.default_rng(seed)
    omega = rng.normal(0, 1.0 / bw, size=(n_rff, inst.shape[1]))
    b = rng.uniform(0, 2 * np.pi, size=n_rff)
    g = np.cos(inst @ omega.T + b)                 # (n, n_rff)
    moments = np.abs((R[:, None] * g).mean(0))
    return float(moments.max() / (R.std() + 1e-8))


def q_sanity(q_pred: np.ndarray, T: np.ndarray, K: int) -> Dict[int, Dict[str, float]]:
    """En[1{T=k} q_k] (target 1), max weight, ESS_k."""
    n = len(T)
    onehot = np.zeros((n, K)); onehot[np.arange(n), T] = 1.0
    w = onehot * q_pred
    out = {}
    for k in range(K):
        wk = w[:, k][T == k]
        ess = (wk.sum() ** 2) / (np.sum(wk ** 2) + 1e-12) if len(wk) else 0.0
        out[k] = {"En_1Tk_qk": float(w[:, k].mean()), "max_q": float(q_pred[:, k].max()),
                  "ess": float(ess)}
    return out


def proxy_strength(W: np.ndarray, V: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    """Regress X out of W and V, CCA on residuals; report top-2 canonical correlations.

    rho_min is the 2nd canonical correlation (weaker proxy direction).
    """
    lrW = LinearRegression().fit(X, W); rW = W - lrW.predict(X)
    lrV = LinearRegression().fit(X, V); rV = V - lrV.predict(X)
    n_comp = min(2, rW.shape[1], rV.shape[1])
    cca = CCA(n_components=n_comp, max_iter=1000)
    cca.fit(rW, rV)
    Wc, Vc = cca.transform(rW, rV)
    rhos = [float(np.corrcoef(Wc[:, i], Vc[:, i])[0, 1]) for i in range(n_comp)]
    rhos = sorted(np.abs(rhos), reverse=True)
    return {"rho_top": rhos[0] if rhos else 0.0,
            "rho_min": rhos[-1] if rhos else 0.0}


def arm_overlap(T: np.ndarray, pi_hat: np.ndarray = None, K: int = None) -> Dict:
    """Per-arm counts and (if given) the distribution of min_k pi_hat_k(X)."""
    K = K or int(T.max() + 1)
    counts = {k: int((T == k).sum()) for k in range(K)}
    out = {"counts": counts}
    if pi_hat is not None:
        out["min_pi_mean"] = float(pi_hat.min(axis=1).mean())
        out["min_pi_q05"] = float(np.quantile(pi_hat.min(axis=1), 0.05))
    return out
