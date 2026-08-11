"""Statistical protocol (spec 6.3): paired tests, BH correction, seed-bootstrap CIs.

Every headline table generator must route through here.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import stats as sps


def paired_tests(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired comparison of method a vs b over seeds (lower metric = better).

    Returns Wilcoxon signed-rank and paired t-test p-values plus the mean difference
    (a - b); negative mean_diff means a is better.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = {"mean_diff": float(np.mean(d)), "n": int(len(d))}
    if np.allclose(d, 0):
        out["wilcoxon_p"], out["ttest_p"] = 1.0, 1.0
        return out
    try:
        out["wilcoxon_p"] = float(sps.wilcoxon(a, b).pvalue)
    except Exception:
        out["wilcoxon_p"] = float("nan")
    try:
        out["ttest_p"] = float(sps.ttest_rel(a, b).pvalue)
    except Exception:
        out["ttest_p"] = float("nan")
    return out


def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05):
    """Return (rejected_bool_array, adjusted_pvalues) under BH FDR control."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj_full = np.empty(m)
    adj_full[order] = np.clip(adj, 0, 1)
    rejected = adj_full <= alpha
    return rejected, adj_full


def seed_bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Bootstrap 95% CI on the MEAN across seeds."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1) if len(v) > 1 else 0.0),
            "ci_low": float(lo), "ci_high": float(hi)}


def summarize(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(values, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1) if len(v) > 1 else 0.0), "n": int(len(v))}
