"""Exploratory Round-3 selector variants (pure functions of NON-oracle candidate features).

Motivation. Round-2 concluded that the frozen MSES rule (moment screen, then minimum out-of-fold
variance) is not a reliable near-oracle selector: in the nonlinear D2 regime the identifying-moment
screen rejects the oracle candidate (~11.5% survival) while flexible sieve2/3 candidates pass and
dominate the catastrophic survivor set. The lesson is that *moment compatibility is anti-correlated
with true causal error in strongly misspecified regimes*, so any selector that trusts the screen
inherits that failure, whereas variance-only selection is more robust.

This module implements a family of alternative rules that vary HOW MUCH the moment screen is
trusted -- from not at all (variance-only) to hard-gated (MSES) -- plus hybrids that keep variance
primary and use moments only to break ties, cap flexibility, or add a parsimony prior. They are
designed for exploratory comparison on FRESH seeds; per the Round-2 protocol none of them is tuned
on the confirmatory final seeds and none replaces the frozen confirmatory verdict.

Oracle isolation. Every selector consumes a ``feat`` mapping {candidate_name -> feature dict} whose
keys are strictly observed / OOF quantities:

    p_h, p_q   : per-arm moment compatibility p-values (arrays of length K)
    d_h, d_q   : RFF-L2 identifying-moment discrepancies (nonnegative scalars)
    var        : mean OOF pseudo-outcome contrast variance  (the MSES efficiency criterion)
    var_max    : max OOF contrast variance
    ci_width   : mean descriptive 95% Wald ATE CI width
    q_balance  : mean |E_n[1{T=k} q_k] - 1| -- an OBSERVABLE q-normalization moment violation
    max_q      : largest fitted treatment-bridge weight (extreme-weight red flag)
    ess        : mean effective sample size of the q-weights
    complexity : fixed solver-flexibility code (kernel/sieve1 = 1, sieve2 = 2, sieve3 = 3), summed
                 over the (h, q) pair; a structural constant, not data

The ``q_balance`` / ``max_q`` / ``ess`` triple is a SECOND observable adequacy signal, distinct
from the RFF identifying-moment p-values: it checks the Hajek self-normalization the treatment
bridge must satisfy on the observed data, with no counterfactual truth. Round-3 tests whether
gating on it is a more reliable screen than the (regime-unreliable) RFF moment screen.

No causal-truth key (``ate_err`` / ``pehe``) is ever read; ``test_selector_variants_oracle_blind``
asserts this by feeding poisoned features.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from ..candidates.library import CANDIDATE_SPECS

ABSTAIN = "ABSTAIN_LIBRARY_INADEQUATE"
EPS = 1e-8

# Structural solver-flexibility codes (NOT data): higher = more flexible / higher variance risk.
_SOLVER_COMPLEXITY = {"kernel": 1, "sieve1": 1, "sieve2": 2, "sieve3": 3}


def candidate_complexity(name: str) -> int:
    """Fixed flexibility code summed over the (h, q) solver pair. Structural, oracle-free."""
    h_solver, q_solver = CANDIDATE_SPECS[name]
    return _SOLVER_COMPLEXITY[h_solver] + _SOLVER_COMPLEXITY[q_solver]


# ---------------------------------------------------------------------------- helpers
def _names(feat: Dict[str, dict]) -> List[str]:
    return sorted(feat)


def _p_dr_min(row: dict) -> float:
    """Worst-arm DR-union compatibility p-value; small => strong moment violation."""
    p_dr = np.maximum(np.asarray(row["p_h"], float), np.asarray(row["p_q"], float))
    return float(np.min(p_dr))


def _screen_survivors(feat: Dict[str, dict], alpha: float) -> List[str]:
    """Bonferroni DR-union screen: keep candidates whose every-arm max(p_h,p_q) >= alpha/K."""
    survivors = []
    for name in _names(feat):
        p_h = np.asarray(feat[name]["p_h"], float)
        p_q = np.asarray(feat[name]["p_q"], float)
        K = p_h.size
        p_dr = np.maximum(p_h, p_q)
        thresh = alpha / K
        if np.all(np.isfinite(p_dr)) and np.all(p_dr >= thresh):
            survivors.append(name)
    return survivors


def _argmin(feat, key_fn) -> str:
    names = _names(feat)
    return min(names, key=lambda n: (key_fn(n), n))


# ---------------------------------------------------------------------------- selectors
def variance_only(feat: Dict[str, dict]) -> str:
    """Round-2's most robust baseline: minimum OOF pseudo-outcome variance, no screen."""
    return _argmin(feat, lambda n: feat[n]["var"])


def min_ci_width(feat: Dict[str, dict]) -> str:
    """Minimum descriptive Wald ATE CI width (a variance proxy; sanity comparator)."""
    return _argmin(feat, lambda n: feat[n]["ci_width"])


def mses(feat: Dict[str, dict], *, alpha: float = 0.05) -> str:
    """Frozen Round-2 rule: hard moment screen, then minimum variance; ABSTAIN if none survive."""
    survivors = _screen_survivors(feat, alpha)
    if not survivors:
        return ABSTAIN
    return min(survivors, key=lambda n: (feat[n]["var"], n))


def mses_varcap(feat: Dict[str, dict], *, alpha: float = 0.05, tau: float = 3.0) -> str:
    """MSES survivors, but reject any whose variance exceeds ``tau x median(var)`` over ALL
    candidates, then take minimum variance. If the cap (or the screen) empties the set, fall back
    to the global minimum-variance candidate instead of abstaining.

    Rationale: this keeps MSES's ability to flag a detectably inadequate library, but refuses to
    deploy a high-variance flexible survivor (the D2 catastrophe), degrading gracefully to
    variance-only exactly where the screen is untrustworthy.
    """
    names = _names(feat)
    med = float(np.median([feat[n]["var"] for n in names]))
    survivors = _screen_survivors(feat, alpha)
    capped = [n for n in survivors if feat[n]["var"] <= tau * med + EPS]
    pool = capped or survivors
    if not pool:
        return variance_only(feat)  # graceful fallback, never abstain
    return min(pool, key=lambda n: (feat[n]["var"], n))


def product_moment(feat: Dict[str, dict]) -> str:
    """Minimum log-space identifying-moment product ``log D_h + log D_q`` (Round-1 'product')."""
    return _argmin(feat, lambda n: np.log(feat[n]["d_h"] + EPS) + np.log(feat[n]["d_q"] + EPS))


def biasvar_mult(feat: Dict[str, dict]) -> str:
    """Minimum ``log D_h + log D_q + log var``: multiplicative bias x variance surrogate."""
    return _argmin(
        feat,
        lambda n: np.log(feat[n]["d_h"] + EPS) + np.log(feat[n]["d_q"] + EPS)
        + np.log(feat[n]["var"] + EPS),
    )


def biasvar_add(feat: Dict[str, dict]) -> str:
    """Minimum ``z(D_h*D_q) + z(var)`` with library-standardized log scores (Round-1 additive)."""
    names = _names(feat)
    prod = np.array([feat[n]["d_h"] * feat[n]["d_q"] for n in names], float)
    var = np.array([feat[n]["var"] for n in names], float)

    def z(x):
        lx = np.log(x + EPS)
        return (lx - lx.mean()) / (lx.std() + EPS)

    score = dict(zip(names, z(prod) + z(var)))
    return min(names, key=lambda n: (score[n], n))


def soft_screen(feat: Dict[str, dict], *, lam: float = 1.0) -> str:
    """Graded (never hard-rejecting) rule: minimize ``log var + lam * penalty`` where the penalty
    ``-mean log p_DR`` grows as moment violations sharpen. lam -> 0 recovers variance-only.

    Included to demonstrate a *failure direction*: because moment compatibility is anti-correlated
    with true error in D2, penalizing violation there hurts, so this should NOT beat variance-only
    in nonlinear regimes -- a diagnostic control, not a proposed fix.
    """
    def score(n):
        p_dr = np.maximum(np.asarray(feat[n]["p_h"], float), np.asarray(feat[n]["p_q"], float))
        penalty = float(-np.mean(np.log(np.clip(p_dr, EPS, 1.0))))
        return np.log(feat[n]["var"] + EPS) + lam * penalty

    return _argmin(feat, score)


def var_primary_moment_tiebreak(feat: Dict[str, dict], *, tol: float = 0.10) -> str:
    """Variance-primary hybrid: restrict to candidates within ``(1+tol) x min var``, then among
    that near-optimal-variance band prefer the most moment-compatible (largest worst-arm p_DR).

    Keeps variance's D2 robustness (moments cannot override a decisively lower-variance choice) but
    lets moments improve selection when variance is close to tied.
    """
    names = _names(feat)
    vmin = min(feat[n]["var"] for n in names)
    band = [n for n in names if feat[n]["var"] <= vmin * (1.0 + tol) + EPS]
    return max(band, key=lambda n: (_p_dr_min(feat[n]), -feat[n]["var"], n))


def complexity_penalized_var(feat: Dict[str, dict], *, gamma: float = 0.25) -> str:
    """Minimum ``var * (1 + gamma * complexity)``: variance with a structural parsimony prior that
    charges flexible sieve2/3 pairs (the D2 catastrophe source) without touching the moment tests.
    """
    return _argmin(feat, lambda n: feat[n]["var"] * (1.0 + gamma * candidate_complexity(n)))


def rank_sum(feat: Dict[str, dict]) -> str:
    """Scale-free Borda aggregation of the variance rank and the moment-product rank."""
    names = _names(feat)
    var = np.array([feat[n]["var"] for n in names], float)
    prod = np.array([feat[n]["d_h"] * feat[n]["d_q"] for n in names], float)
    r_var = var.argsort().argsort()
    r_prod = prod.argsort().argsort()
    score = dict(zip(names, r_var + r_prod))
    return min(names, key=lambda n: (score[n], n))


def _zlog(names: List[str], vals) -> np.ndarray:
    """Library-standardized log score (over candidates within one task), as in biasvar_add."""
    lx = np.log(np.asarray(vals, float) + EPS)
    return (lx - lx.mean()) / (lx.std() + EPS)


def min_q_balance(feat: Dict[str, dict]) -> str:
    """Minimize the observable q-normalization violation ``|E_n[1{T=k} q_k] - 1|`` alone.

    Round-3 found this signal is positively informative about true error in EVERY tested regime,
    unlike variance (useless in the hamd semi-synthetic) or the RFF moment screen (anti-informative
    in nonlinear).
    """
    return _argmin(feat, lambda n: feat[n]["q_balance"])


def varq_combo(feat: Dict[str, dict]) -> str:
    """Minimize ``z(var) + z(q_balance)``: a two-signal soft fusion that strictly improves on
    variance-only pooled (Round-3 median oracle-ratio 1.77 vs 2.51) and is never worse than
    variance-only in any single tested regime -- it repairs the hamd variance-only failure and
    avoids the nonlinear moment-screen catastrophe. Weaker than the three-signal
    :func:`varq_moment_combo` where the moment signal is informative (exact, hamd).
    """
    names = _names(feat)
    score = dict(zip(names, _zlog(names, [feat[n]["var"] for n in names])
                     + _zlog(names, [feat[n]["q_balance"] for n in names])))
    return min(names, key=lambda n: (score[n], n))


def varq_moment_combo(feat: Dict[str, dict]) -> str:
    """Minimize ``z(var) + z(q_balance) + z(moment_violation)``: soft fusion of all three observable
    adequacy signals. Among the most robust rules in Round-3 (worst-regime median ~1.69), statistically
    indistinguishable from the other top soft fusions (``soft_screen`` var+moment ~1.51,
    ``biasvar_mult`` var+discrepancy ~1.60) on 16 seeds/regime.

    The moment signal is catastrophic as a HARD screen (MSES: nonlinear 44x) but net-beneficial as ONE
    standardized term among three: the regime-robust terms buffer it in its bad regime (nonlinear
    1.175 vs varq_combo 1.157 -- only marginally worse) while it rescues the fusion where variance and
    q_balance are both weak (exact 1.514 vs varq_combo 5.408). No single soft fusion is claimed to be
    THE best; the supported claim is the class. Not tuned; does not reach a deployable target.
    """
    names = _names(feat)
    viol = [-np.log(np.clip(_p_dr_min(feat[n]), EPS, 1.0)) for n in names]  # higher = worse
    zm = (np.asarray(viol) - np.mean(viol)) / (np.std(viol) + EPS)
    score = dict(zip(names, _zlog(names, [feat[n]["var"] for n in names])
                     + _zlog(names, [feat[n]["q_balance"] for n in names]) + zm))
    return min(names, key=lambda n: (score[n], n))


def q_gated_variance(feat: Dict[str, dict], *, qb_max: float = 0.10, max_q_cap: float = 20.0) -> str:
    """Screen on OBSERVABLE q-normalization health (``q_balance <= qb_max`` and
    ``max_q <= max_q_cap``), then take minimum variance. Falls back to the global minimum-variance
    candidate if the q-screen empties the set.

    Unlike the RFF moment screen, this gate targets the exact pathology behind the D2 catastrophe:
    the flexible survivors that fool the identifying-moment test carry extreme, poorly self-
    normalized weights, which the q-diagnostics expose without any counterfactual truth.
    """
    names = _names(feat)
    ok = [n for n in names if feat[n]["q_balance"] <= qb_max and feat[n]["max_q"] <= max_q_cap]
    pool = ok or names
    return min(pool, key=lambda n: (feat[n]["var"], n))


def q_and_moment_gated_variance(feat: Dict[str, dict], *, alpha: float = 0.05,
                                qb_max: float = 0.10, max_q_cap: float = 20.0) -> str:
    """Require BOTH the RFF moment screen and the q-normalization screen, then minimum variance;
    fall back to the global minimum-variance candidate if the joint screen empties the set."""
    names = _names(feat)
    survivors = set(_screen_survivors(feat, alpha))
    ok = [n for n in names if n in survivors
          and feat[n]["q_balance"] <= qb_max and feat[n]["max_q"] <= max_q_cap]
    pool = ok or [n for n in names
                  if feat[n]["q_balance"] <= qb_max and feat[n]["max_q"] <= max_q_cap] or names
    return min(pool, key=lambda n: (feat[n]["var"], n))


def fixed_kernel(feat: Dict[str, dict]) -> str:
    names = _names(feat)
    return "kernel_kernel" if "kernel_kernel" in names else names[0]


def fixed_sieve1(feat: Dict[str, dict]) -> str:
    names = _names(feat)
    return "sieve1_sieve1" if "sieve1_sieve1" in names else names[0]


# ---------------------------------------------------------------------------- registry
def build_registry(*, alpha: float = 0.05) -> Dict[str, Callable[[Dict[str, dict]], str]]:
    """Return the exploratory selector library, including hyperparameter variants.

    Keys are stable selector identifiers; values are callables taking only ``feat``.
    """
    reg: Dict[str, Callable[[Dict[str, dict]], str]] = {
        "variance_only": variance_only,
        "min_ci_width": min_ci_width,
        "MSES": lambda f: mses(f, alpha=alpha),
        "product_moment": product_moment,
        "biasvar_mult": biasvar_mult,
        "biasvar_add": biasvar_add,
        "rank_sum": rank_sum,
        "min_q_balance": min_q_balance,
        "varq_combo": varq_combo,
        "varq_moment_combo": varq_moment_combo,
        "q_gated_variance": lambda f: q_gated_variance(f),
        "q_and_moment_gated_variance": lambda f: q_and_moment_gated_variance(f, alpha=alpha),
        "fixed_kernel": fixed_kernel,
        "fixed_sieve1": fixed_sieve1,
    }
    for tau in (1.5, 3.0):
        reg[f"mses_varcap[tau={tau}]"] = (lambda t: lambda f: mses_varcap(f, alpha=alpha, tau=t))(tau)
    for lam in (0.5, 1.0):
        reg[f"soft_screen[lam={lam}]"] = (lambda l: lambda f: soft_screen(f, lam=l))(lam)
    for tol in (0.05, 0.15):
        reg[f"var_tiebreak[tol={tol}]"] = (lambda t: lambda f: var_primary_moment_tiebreak(f, tol=t))(tol)
    for gamma in (0.25, 0.5):
        reg[f"complexity_var[gamma={gamma}]"] = (lambda g: lambda f: complexity_penalized_var(f, gamma=g))(gamma)
    return reg
