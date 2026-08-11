"""Solver-independent held-out moment validator (MV-OPM §1.2).

Identifying-moment residuals for arm k:
    R_ik = 1{T_i=k} (Y_i - h_k(W_i,X_i))          (outcome side; instrument z=(V,X))
    S_ik = 1{T_i=k} q_k(V_i,X_i) - 1              (treatment side; instrument u=(W,X))

The validator does NOT depend on how a candidate was trained. A SINGLE feature/instrument bank
(RFF weights, standardizers, bandwidths) is built once per comparison from the held-out
validation instruments and shared across ALL candidates, so scale/feature choices cannot
trivially decide the ranking. Residuals are normalized within arm by sd(R) / sd(S) (documented)
so the h- and q-side discrepancies are dimensionless and their product is meaningful.

Diagnostics per side:
  * RFF-L2:  D = sqrt( mean_j ( E_n[ R_norm g_j(z) ] )^2 )        [primary]
  * RFF-max: D = max_j | E_n[ R_norm g_j(z) ] |                    [sensitivity]
  * MMR:     D = sqrt( max( U-stat(R_norm, K_bw), 0 ) )            [kernel discrepancy]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import median_bandwidth, standardize_apply, standardize_fit

EPS = 1e-8


@dataclass
class MomentBank:
    """Shared, candidate-independent validation bank (built on held-out instruments)."""
    om_h: np.ndarray; b_h: np.ndarray; mh: np.ndarray; sh: np.ndarray; bw_h: float
    om_q: np.ndarray; b_q: np.ndarray; mq: np.ndarray; sq: np.ndarray; bw_q: float
    n_rff: int


def build_bank(inst_h: np.ndarray, inst_q: np.ndarray, n_rff: int = 200, seed: int = 20240) -> MomentBank:
    """Build the shared bank from held-out instruments (z=(V,X) for h, u=(W,X) for q).

    ``seed`` is fixed and candidate-independent. Standardizers and bandwidths come from the
    validation instruments only (never from candidate parameters or outer-test data).
    """
    rng = np.random.default_rng(seed)
    mh, sh = standardize_fit(inst_h.astype(np.float64))
    mq, sq = standardize_fit(inst_q.astype(np.float64))
    zh = standardize_apply(inst_h, mh, sh); zq = standardize_apply(inst_q, mq, sq)
    bw_h = median_bandwidth(zh, seed=seed); bw_q = median_bandwidth(zq, seed=seed + 1)
    om_h = rng.normal(0, 1.0 / bw_h, size=(n_rff, inst_h.shape[1])); b_h = rng.uniform(0, 2 * np.pi, n_rff)
    om_q = rng.normal(0, 1.0 / bw_q, size=(n_rff, inst_q.shape[1])); b_q = rng.uniform(0, 2 * np.pi, n_rff)
    return MomentBank(om_h, b_h, mh, sh, bw_h, om_q, b_q, mq, sq, bw_q, n_rff)


def _rff(inst, om, b, m, s):
    z = standardize_apply(inst.astype(np.float64), m, s)
    return np.cos(z @ om.T + b)                                   # (n, J)


def _mmr(resid, inst, bw, m, s, max_pairs=3000, seed=0):
    """sqrt(max(U-stat,0)) kernel discrepancy, subsampled for O(max_pairs)."""
    z = standardize_apply(inst.astype(np.float64), m, s)
    n = len(resid)
    rng = np.random.default_rng(seed)
    if n * (n - 1) // 2 <= max_pairs:
        i, j = np.triu_indices(n, 1)
    else:
        i = rng.integers(0, n, max_pairs); j = rng.integers(0, n, max_pairs)
        keep = i != j; i, j = i[keep], j[keep]
    d2 = np.sum((z[i] - z[j]) ** 2, axis=1)
    k = np.exp(-d2 / (2 * bw * bw))
    u = np.mean(resid[i] * resid[j] * k)
    return float(np.sqrt(max(u, 0.0)))


def evaluate_candidate(bank: MomentBank, candidate, view, kinds=("rff_l2", "rff_max", "mmr")) -> dict:
    """Held-out moment discrepancies D_h, D_q for a candidate, using the SHARED bank."""
    h = candidate.predict_h(view.W, view.X)
    q = candidate.predict_q(view.V, view.X)
    return discrepancy_from_hq(bank, h, q, view, kinds)


def discrepancy_from_hq(bank: MomentBank, h: np.ndarray, q: np.ndarray, view,
                        kinds=("rff_l2", "rff_max", "mmr")) -> dict:
    """Moment discrepancies from raw h/q value arrays (n,K) — used by candidates AND by the
    exact-bridge mechanism study (which corrupts the population bridges directly)."""
    W, V, X, Y, T, K = view.W, view.V, view.X, view.Y, view.T, view.K
    inst_h = np.concatenate([V, X], axis=1)
    inst_q = np.concatenate([W, X], axis=1)
    gh = _rff(inst_h, bank.om_h, bank.b_h, bank.mh, bank.sh)      # (n,J) shared
    gq = _rff(inst_q, bank.om_q, bank.b_q, bank.mq, bank.sq)

    acc = {f"D_{s}_{k}": [] for s in ("h", "q") for k in kinds}
    for k in range(K):
        mask = (T == k).astype(float)
        R = mask * (Y - h[:, k]); S = mask * q[:, k] - 1.0
        Rn = R / (R.std() + EPS); Sn = S / (S.std() + EPS)       # within-arm normalization
        if "rff_l2" in kinds:
            acc["D_h_rff_l2"].append(np.sqrt(np.mean((Rn[:, None] * gh).mean(0) ** 2)))
            acc["D_q_rff_l2"].append(np.sqrt(np.mean((Sn[:, None] * gq).mean(0) ** 2)))
        if "rff_max" in kinds:
            acc["D_h_rff_max"].append(np.abs((Rn[:, None] * gh).mean(0)).max())
            acc["D_q_rff_max"].append(np.abs((Sn[:, None] * gq).mean(0)).max())
        if "mmr" in kinds:
            acc["D_h_mmr"].append(_mmr(Rn, inst_h, bank.bw_h, bank.mh, bank.sh, seed=k))
            acc["D_q_mmr"].append(_mmr(Sn, inst_q, bank.bw_q, bank.mq, bank.sq, seed=k))
    return {key: float(np.mean(v)) for key, v in acc.items()}
