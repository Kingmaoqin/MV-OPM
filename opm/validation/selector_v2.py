"""Round-2 Moment-Screened Efficiency Selection (MSES).

This module is deliberately separate from the frozen Round-1 scalar selectors and contains no
oracle imports. All uncertainty inputs must come from out-of-fold pseudo-outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

ABSTAIN = "ABSTAIN_LIBRARY_INADEQUATE"


@dataclass(frozen=True)
class ScreeningResult:
    passes: bool
    p_dr: np.ndarray
    threshold: float
    failed_arms: tuple[int, ...]


def screen_candidate(p_h, p_q, *, alpha: float = 0.05, n_required_arms: int | None = None) -> ScreeningResult:
    """Bonferroni screen using the proximal DR union-null p-value ``max(p_h,p_q)``."""
    ph = np.asarray(p_h, dtype=float).reshape(-1)
    pq = np.asarray(p_q, dtype=float).reshape(-1)
    if ph.shape != pq.shape or ph.size == 0:
        raise ValueError("p_h and p_q must be nonempty and have the same shape")
    K = int(n_required_arms or ph.size)
    if K != ph.size:
        raise ValueError("n_required_arms must equal the number of tested arm p-values")
    threshold = float(alpha / K)
    p_dr = np.maximum(ph, pq)
    failed = tuple(int(k) for k in np.flatnonzero(~np.isfinite(p_dr) | (p_dr < threshold)))
    return ScreeningResult(len(failed) == 0, p_dr, threshold, failed)


def pseudo_outcome_uncertainty(phi: np.ndarray, *, reference_arm: int = 0) -> dict:
    """Compute contrast variances from a complete OOF pseudo-outcome matrix."""
    x = np.asarray(phi, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("phi must have shape (n>=2, K>=2)")
    if not np.all(np.isfinite(x)):
        raise ValueError("OOF phi contains nonfinite or uncovered entries")
    arms = [k for k in range(x.shape[1]) if k != reference_arm]
    psi = np.column_stack([x[:, k] - x[:, reference_arm] for k in arms])
    contrast_var = np.var(psi, axis=0, ddof=1)
    n = x.shape[0]
    se = np.sqrt(contrast_var / n)
    return {
        "contrast_arms": arms,
        "contrast_variance": contrast_var,
        "variance_mean": float(np.mean(contrast_var)),
        "variance_max": float(np.max(contrast_var)),
        "ate_se": se,
        "ate_ci_width": 3.92 * se,
    }


def select_mses(
    tests: Dict[str, dict],
    uncertainties: Dict[str, dict],
    *,
    alpha: float = 0.05,
) -> dict:
    """Screen on moment adequacy, then minimize mean OOF contrast variance."""
    if set(tests) != set(uncertainties):
        raise ValueError("tests and uncertainties must contain identical candidates")
    screening = {}
    for name in sorted(tests):
        t = tests[name]
        screening[name] = screen_candidate(t["p_h"], t["p_q"], alpha=alpha)
    survivors = [name for name in sorted(screening) if screening[name].passes]
    if not survivors:
        selected = ABSTAIN
    else:
        selected = min(survivors, key=lambda n: (float(uncertainties[n]["variance_mean"]), n))
    return {
        "selected": selected,
        "survivors": survivors,
        "survivor_count": len(survivors),
        "screening": screening,
        "abstained": selected == ABSTAIN,
    }
