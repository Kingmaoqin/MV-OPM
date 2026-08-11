"""Prespecified selection scores over held-out moment discrepancies (MV-OPM §1.3, §6).

The PRIMARY proposed criterion is the PRODUCT score, motivated by the orthogonal/doubly-robust
remainder ~ ||h_err|| * ||q_err||. Alternatives (h-only, q-only, sum, max) are prespecified
here so that any post-hoc winner is clearly labelled exploratory (never silently promoted).
All scores are LOWER = better; select = argmin. The product is ranked via
log(D_h+eps)+log(D_q+eps) (monotone in D_h*D_q, numerically safe).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

EPS = 1e-8
PRIMARY = "product"
SCORE_NAMES = ["product", "h_only", "q_only", "sum", "max"]


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / (s + EPS)


def compute_scores(cand_D: Dict[str, dict], kind: str = "rff_l2") -> Dict[str, Dict[str, float]]:
    """cand_D: name -> {'D_h_<kind>':.., 'D_q_<kind>':..}. Returns score_name -> {cand: value}."""
    names = list(cand_D)
    Dh = np.array([cand_D[n][f"D_h_{kind}"] for n in names])
    Dq = np.array([cand_D[n][f"D_q_{kind}"] for n in names])
    zh, zq = _z(Dh), _z(Dq)
    out = {
        "product": np.log(Dh + EPS) + np.log(Dq + EPS),
        "h_only": Dh,
        "q_only": Dq,
        "sum": zh + zq,
        "max": np.maximum(zh, zq),
    }
    return {s: {n: float(v[i]) for i, n in enumerate(names)} for s, v in out.items()}


def select(score_map: Dict[str, float]) -> str:
    """argmin (lower = better)."""
    return min(score_map, key=score_map.get)
