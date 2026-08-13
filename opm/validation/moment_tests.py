"""Formal studentized identifying-moment tests for MV-OPM Round 2.

The descriptive discrepancies in :mod:`opm.validation.moments` are intentionally retained.
This module adds a candidate-comparable hypothesis test using a shared RFF bank and a centered
Gaussian multiplier bootstrap. It contains no oracle or DGP imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from ..utils import standardize_apply
from .moments import MomentBank

SE_EPS = 1e-12


@dataclass(frozen=True)
class MaxTestResult:
    statistic: float
    p_value: float
    n_effective_moments: int
    warning: str | None
    bootstrap_seed: int
    n_boot: int

    def as_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n_effective_moments": self.n_effective_moments,
            "warning": self.warning,
            "bootstrap_seed": self.bootstrap_seed,
            "n_boot": self.n_boot,
        }


def rff_features(bank: MomentBank, inst_h: np.ndarray, inst_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the two fixed candidate-independent RFF design matrices."""
    zh = standardize_apply(np.asarray(inst_h, dtype=np.float64), bank.mh, bank.sh)
    zq = standardize_apply(np.asarray(inst_q, dtype=np.float64), bank.mq, bank.sq)
    gh = np.cos(zh @ bank.om_h.T + bank.b_h)
    gq = np.cos(zq @ bank.om_q.T + bank.b_q)
    return gh, gq


def studentized_rff_max_test(
    residual: np.ndarray,
    features: np.ndarray,
    *,
    n_boot: int = 999,
    bootstrap_seed: int,
    eps: float = SE_EPS,
) -> MaxTestResult:
    """Test ``E[residual * feature_j] = 0`` for all j with a max statistic.

    The bootstrap uses centered observation-level moments and Gaussian multipliers. The observed
    and bootstrap statistics use the same sample standard errors. A plus-one p-value prevents a
    zero Monte Carlo p-value and makes the calculation deterministic for a fixed seed.
    """
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    g = np.asarray(features, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] != r.shape[0]:
        raise ValueError("features must have shape (n, J) matching residual")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    n = len(r)
    if n < 2:
        return MaxTestResult(float("nan"), float("nan"), 0, "n_lt_2", bootstrap_seed, n_boot)

    z = r[:, None] * g
    means = np.mean(z, axis=0)
    sd = np.std(z, axis=0, ddof=1)
    se = sd / np.sqrt(n)
    keep = np.isfinite(means) & np.isfinite(se) & (se > eps)
    n_eff = int(np.sum(keep))
    if n_eff == 0:
        return MaxTestResult(float("nan"), float("nan"), 0, "no_effective_moments", bootstrap_seed, n_boot)

    z = z[:, keep]
    means = means[keep]
    se = se[keep]
    observed = float(np.max(np.abs(means / se)))
    centered = z - means[None, :]
    rng = np.random.default_rng(int(bootstrap_seed))
    exceed = 0
    # Chunking avoids a B x n x J tensor while remaining exactly deterministic.
    chunk = min(128, n_boot)
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        xi = rng.standard_normal((b, n))
        boot_means = xi @ centered / n
        boot_stats = np.max(np.abs(boot_means / se[None, :]), axis=1)
        exceed += int(np.sum(boot_stats >= observed))
    p = float((1 + exceed) / (n_boot + 1))
    warning = None if n_eff == g.shape[1] else f"excluded_{g.shape[1] - n_eff}_degenerate_moments"
    return MaxTestResult(observed, p, n_eff, warning, int(bootstrap_seed), int(n_boot))


def test_candidate_hq(
    bank: MomentBank,
    h: np.ndarray,
    q: np.ndarray,
    view,
    *,
    n_boot: int = 999,
    bootstrap_seed: int,
) -> dict:
    """Run the h- and q-side max tests for every arm of one candidate."""
    h = np.asarray(h, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if h.shape != (view.n, view.K) or q.shape != (view.n, view.K):
        raise ValueError("h and q must both have shape (view.n, view.K)")
    inst_h = np.concatenate([view.V, view.X], axis=1)
    inst_q = np.concatenate([view.W, view.X], axis=1)
    gh, gq = rff_features(bank, inst_h, inst_q)
    h_results, q_results = [], []
    for k in range(view.K):
        mask = (view.T == k).astype(np.float64)
        R = mask * (view.Y - h[:, k])
        S = mask * q[:, k] - 1.0
        # Fixed offsets make every arm/side reproducible and keep the h/q streams isolated.
        h_results.append(studentized_rff_max_test(
            R, gh, n_boot=n_boot, bootstrap_seed=int(bootstrap_seed) + 2 * k
        ))
        q_results.append(studentized_rff_max_test(
            S, gq, n_boot=n_boot, bootstrap_seed=int(bootstrap_seed) + 2 * k + 1
        ))
    p_h = np.array([x.p_value for x in h_results], dtype=float)
    p_q = np.array([x.p_value for x in q_results], dtype=float)
    return {
        "h": [x.as_dict() for x in h_results],
        "q": [x.as_dict() for x in q_results],
        "p_h": p_h,
        "p_q": p_q,
        "p_DR": np.maximum(p_h, p_q),
    }


def test_many_candidates(
    bank: MomentBank,
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    view,
    *,
    n_boot: int = 999,
    bootstrap_seed: int,
) -> Dict[str, dict]:
    """Apply one shared bank to all candidates with stable per-candidate bootstrap streams."""
    out = {}
    for i, name in enumerate(sorted(predictions)):
        h, q = predictions[name]
        out[name] = test_candidate_hq(
            bank, h, q, view, n_boot=n_boot, bootstrap_seed=int(bootstrap_seed) + 10000 * i
        )
    return out
