"""Nested cross-fitting for MV-OPM estimator selection (MV-OPM §2).

Selection adds a layer of adaptivity, so ordinary OOF is insufficient. For each OUTER fold:
INNER folds of OUTER_TRAIN fit every candidate, score held-out identifying-moment violations,
and select a candidate; the selected candidate is REFIT on the full OUTER_TRAIN and predicts
h/q on OUTER_TEST to build outer-OOF (STAR) pseudo-outcomes. The downstream CATE head is fit
only after all outer folds. OUTER_TEST never influences candidate fitting, selection, the
validation bank, bandwidth, or standardization. Every split is audited (see the anti-leakage
tests in tests/test_mvopm_invariants.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..candidates.library import CANDIDATE_NAMES, CandidateSet
from ..estimator.crossfit import make_folds
from ..estimator.pseudo_outcome import compute_phi
from ..estimator.tau_head import fit_cate_and_ate
from ..validation.moments import build_bank, evaluate_candidate
from ..validation.selector import PRIMARY, compute_scores, select


@dataclass
class NestedConfig:
    K: int
    outer_folds: int = 4
    inner_folds: int = 3
    seed: int = 0
    device: str = "cpu"
    score_name: str = PRIMARY
    kind: str = "rff_l2"
    n_rff: int = 200
    bank_seed: int = 20240
    candidate_names: list = None
    bridge_kwargs: dict = field(default_factory=dict)
    head_kwargs: dict = field(default_factory=dict)


def _inst_h(v):
    return np.concatenate([v.V, v.X], axis=1)


def _inst_q(v):
    return np.concatenate([v.W, v.X], axis=1)


def nested_mvopm(view, cfg: NestedConfig) -> dict:
    """Run nested MV-OPM on an ObservedDatasetView. Returns phi (outer-OOF), per-fold selected
    candidate, fitted tau head, ATE CIs, and an index-audit log."""
    n, K = view.n, view.K
    names = sorted(cfg.candidate_names) if cfg.candidate_names else list(CANDIDATE_NAMES)
    outer = make_folds(n, cfg.outer_folds, cfg.seed)
    phi = np.zeros((n, K))
    selected, audit = [], []

    all_idx = np.arange(n)
    for of, outer_test in enumerate(outer):
        outer_train = np.setdiff1d(all_idx, outer_test, assume_unique=False)
        # INNER folds *within* OUTER_TRAIN
        inner = make_folds(len(outer_train), cfg.inner_folds, cfg.seed + 101 * of + 1)
        D_acc = {nm: [] for nm in names}
        inner_audit = []
        for inv_local in inner:
            inner_valid = outer_train[inv_local]                       # map local -> global
            inner_train = np.setdiff1d(outer_train, inner_valid)
            cs = CandidateSet(K, seed=cfg.seed + 1000 * of + 1, device=cfg.device,
                              bridge_kwargs=cfg.bridge_kwargs, candidate_names=names).fit(
                view.subset(inner_train))
            ivv = view.subset(inner_valid)
            bank = build_bank(_inst_h(ivv), _inst_q(ivv), cfg.n_rff, cfg.bank_seed)  # held-out only
            for nm, cand in cs:
                D_acc[nm].append(evaluate_candidate(bank, cand, ivv, kinds=(cfg.kind,)))
            inner_audit.append({
                "inner_train_valid_disjoint": len(np.intersect1d(inner_train, inner_valid)) == 0,
                "inner_subset_of_outer_train": np.all(np.isin(np.concatenate([inner_train, inner_valid]), outer_train)),
                "outer_test_absent_from_inner": len(np.intersect1d(np.concatenate([inner_train, inner_valid]), outer_test)) == 0,
            })
        cand_D = {nm: {key: float(np.mean([d[key] for d in D_acc[nm]])) for key in D_acc[nm][0]}
                  for nm in names}
        scores = compute_scores(cand_D, cfg.kind)[cfg.score_name]
        sel = select(scores)
        selected.append(sel)

        # refit SELECTED candidate on the FULL OUTER_TRAIN, predict on OUTER_TEST
        cs_full = CandidateSet(K, seed=cfg.seed + 7 * of + 3, device=cfg.device,
                               bridge_kwargs=cfg.bridge_kwargs, candidate_names=[sel]).fit(
            view.subset(outer_train))
        cand = cs_full.get(sel)
        otev = view.subset(outer_test)
        h = cand.predict_h(otev.W, otev.X); q = cand.predict_q(otev.V, otev.X)
        phi[outer_test] = compute_phi(h, q, otev)
        audit.append({
            "fold": of, "selected": sel,
            "outer_train_test_disjoint": len(np.intersect1d(outer_train, outer_test)) == 0,
            "inner": inner_audit,
        })

    tau_head, g0_head, ate = fit_cate_and_ate(view.X, phi, K, cfg.device, cfg.seed, cfg.head_kwargs)
    return {"phi": phi, "selected": selected, "tau_head": tau_head, "g0_head": g0_head,
            "ate": ate, "audit": audit,
            "predict_cate": lambda X: tau_head.predict(X).reshape(X.shape[0], K - 1),
            "ate_vector": np.array([ate[k]["ate"] for k in range(1, K)])}
