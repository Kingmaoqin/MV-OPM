"""Secondary cross-fitted relative CATE-risk evaluator for Round 2.

This is not the primary MSES selector. Its interpretation requires the conditions documented in
``docs/mvopm_round2/CATE_SELECTION_ANALYSIS.md``.
"""
from __future__ import annotations

import numpy as np


def relative_cate_risk(tau_a: np.ndarray, tau_b: np.ndarray, signal: np.ndarray) -> dict:
    """Estimate risk(a)-risk(b) using a conditionally unbiased proximal signal.

    Inputs must be predictions/signals on an evaluation fold excluded from all corresponding
    nuisance and candidate fits. Columns represent non-reference treatment contrasts.
    """
    a = np.asarray(tau_a, dtype=float)
    b = np.asarray(tau_b, dtype=float)
    d = np.asarray(signal, dtype=float)
    if a.shape != b.shape or a.shape != d.shape or a.ndim != 2:
        raise ValueError("tau_a, tau_b, and signal must share shape (n, K-1)")
    if a.shape[0] < 2 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.all(np.isfinite(d)):
        raise ValueError("relative-risk inputs must be finite with at least two rows")
    per_contrast = a * a - b * b - 2.0 * (a - b) * d
    per_observation = np.mean(per_contrast, axis=1)
    estimate = float(np.mean(per_observation))
    se = float(np.std(per_observation, ddof=1) / np.sqrt(len(per_observation)))
    return {
        "risk_a_minus_b": estimate,
        "se": se,
        "ci_low": estimate - 1.96 * se,
        "ci_high": estimate + 1.96 * se,
        "n": int(len(per_observation)),
        "n_contrasts": int(a.shape[1]),
    }
