"""Invariants for the exploratory Round-3 selector variants.

The central guarantee is oracle-blindness: no selector may read a causal-truth key. We assert this
behaviorally by poisoning every candidate's feature dict with adversarial ``ate_err`` / ``pehe``
values that, if consulted, would flip the choice, and checking the decision is unchanged.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from opm.validation import selector_variants as sv


def _toy_feat():
    """Four candidates with a clear variance ordering and mixed moment compatibility."""
    return {
        "kernel_kernel": {"p_h": [0.40, 0.55], "p_q": [0.60, 0.50], "d_h": 0.10, "d_q": 0.12,
                          "var": 0.020, "var_max": 0.030, "ci_width": 0.18,
                          "q_balance": 0.02, "max_q": 8.0, "ess": 800.0},
        "sieve1_sieve1": {"p_h": [0.30, 0.20], "p_q": [0.25, 0.35], "d_h": 0.15, "d_q": 0.14,
                          "var": 0.025, "var_max": 0.040, "ci_width": 0.20,
                          "q_balance": 0.03, "max_q": 10.0, "ess": 700.0},
        "sieve2_sieve2": {"p_h": [0.90, 0.88], "p_q": [0.92, 0.80], "d_h": 0.05, "d_q": 0.06,
                          "var": 0.090, "var_max": 0.150, "ci_width": 0.40,
                          "q_balance": 0.05, "max_q": 15.0, "ess": 500.0},
        # sieve3 is jointly incompatible on arm 0 (BOTH p_h and p_q below the DR-union threshold),
        # so the doubly-robust screen rejects it despite its low variance; it also carries a
        # pathological q-normalization (q_balance 0.4, max_q 60) that the q-gate exposes.
        "sieve3_sieve3": {"p_h": [0.001, 0.50], "p_q": [0.002, 0.50], "d_h": 0.30, "d_q": 0.28,
                          "var": 0.012, "var_max": 0.020, "ci_width": 0.14,
                          "q_balance": 0.40, "max_q": 60.0, "ess": 120.0},
    }


def test_registry_returns_valid_candidates():
    feat = _toy_feat()
    reg = sv.build_registry()
    valid = set(feat) | {sv.ABSTAIN}
    assert reg, "registry must be nonempty"
    for name, fn in reg.items():
        choice = fn(feat)
        assert choice in valid, f"{name} returned {choice!r}"


def test_variance_only_picks_min_variance():
    feat = _toy_feat()
    assert sv.variance_only(feat) == "sieve3_sieve3"   # var 0.012 is smallest


def test_mses_screens_out_failed_arm():
    feat = _toy_feat()
    # sieve3 fails arm-0 (p_DR = max(0.001,0.50) = 0.001 < 0.05/2); it must not be MSES's pick
    # even though it has the lowest variance.
    choice = sv.mses(feat, alpha=0.05)
    assert choice != "sieve3_sieve3"
    assert choice in ("kernel_kernel", "sieve1_sieve1", "sieve2_sieve2")


def test_mses_abstains_when_all_fail():
    feat = {n: copy.deepcopy(r) for n, r in _toy_feat().items()}
    for r in feat.values():
        r["p_h"] = [0.0001, 0.0001]
        r["p_q"] = [0.0001, 0.0001]
    assert sv.mses(feat, alpha=0.05) == sv.ABSTAIN


def test_varcap_falls_back_not_abstain():
    """When the screen empties, mses_varcap degrades to variance-only rather than abstaining."""
    feat = {n: copy.deepcopy(r) for n, r in _toy_feat().items()}
    for r in feat.values():
        r["p_h"] = [0.0001, 0.0001]
        r["p_q"] = [0.0001, 0.0001]
    assert sv.mses_varcap(feat, alpha=0.05, tau=3.0) == sv.variance_only(feat)


def test_q_gate_rejects_pathological_q_despite_low_variance():
    """sieve3 has the lowest variance but a pathological q-normalization; the q-gate must avoid it
    even though variance_only would take it."""
    feat = _toy_feat()
    assert sv.variance_only(feat) == "sieve3_sieve3"          # min variance is the bad one
    assert sv.q_gated_variance(feat) != "sieve3_sieve3"       # q-gate rejects it
    assert sv.q_gated_variance(feat) == "kernel_kernel"       # then min variance among healthy


def test_q_gate_falls_back_when_all_pathological():
    feat = {n: copy.deepcopy(r) for n, r in _toy_feat().items()}
    for r in feat.values():
        r["q_balance"] = 0.9
        r["max_q"] = 99.0
    # all fail the q-screen -> fall back to global minimum variance, never abstain
    assert sv.q_gated_variance(feat) == sv.variance_only(feat)


def test_min_q_balance_picks_lowest_violation():
    feat = _toy_feat()
    assert sv.min_q_balance(feat) == "kernel_kernel"   # q_balance 0.02 is smallest


def test_varq_combo_avoids_pathological_q_candidate():
    """varq_combo must not pick the low-variance-but-q-pathological candidate that variance_only
    would take."""
    feat = _toy_feat()
    assert sv.variance_only(feat) == "sieve3_sieve3"
    assert sv.varq_combo(feat) != "sieve3_sieve3"


def test_complexity_penalty_charges_flexible_pairs():
    assert sv.candidate_complexity("kernel_kernel") == 2
    assert sv.candidate_complexity("sieve3_sieve3") == 6
    assert sv.candidate_complexity("kernel_sieve1") == 2


@pytest.mark.parametrize("name", list(sv.build_registry().keys()))
def test_oracle_blind(name):
    """Poison each candidate with adversarial truth keys; the decision must not move."""
    reg = sv.build_registry()
    fn = reg[name]
    feat = _toy_feat()
    baseline = fn(feat)
    poisoned = copy.deepcopy(feat)
    # Make the *worst* observed choice look oracle-perfect and vice versa; a selector that peeks
    # at these keys would be tempted to change its pick.
    for i, cand in enumerate(sorted(poisoned)):
        poisoned[cand]["ate_err"] = -1000.0 if cand == baseline else 1000.0
        poisoned[cand]["pehe"] = -1000.0 if cand == baseline else 1000.0
        poisoned[cand]["oracle_best"] = (cand != baseline)
    assert fn(poisoned) == baseline, f"{name} appears to read a causal-truth key"
