"""Candidate-independent inner-OOF validation population for Round-2 selection."""
from __future__ import annotations

import numpy as np

from ..candidates.library import CANDIDATE_NAMES, CandidateSet
from ..estimator.crossfit import make_folds
from ..estimator.pseudo_outcome import compute_phi
from .moment_tests import test_many_candidates
from .moments import build_bank, discrepancy_from_hq
from .selector_v2 import pseudo_outcome_uncertainty, select_mses


def collect_candidate_oof(
    view,
    *,
    n_folds: int = 3,
    seed: int = 0,
    device: str = "cpu",
    candidate_names=None,
    bridge_kwargs=None,
    n_rff: int = 200,
    bank_seed: int = 20240,
    n_boot: int = 999,
    bootstrap_seed: int = 7020240,
) -> dict:
    """Fit candidates foldwise and evaluate all OOF residuals with one shared bank.

    Every observation receives exactly one prediction from a model that excluded it. The RFF
    bank is built only after OOF prediction collection, on the common validation population.
    """
    names = sorted(candidate_names) if candidate_names else list(CANDIDATE_NAMES)
    n, K = view.n, view.K
    h = {nm: np.full((n, K), np.nan) for nm in names}
    q = {nm: np.full((n, K), np.nan) for nm in names}
    coverage = np.zeros(n, dtype=np.int64)
    fold_audit = []
    all_idx = np.arange(n)
    folds = make_folds(n, n_folds, seed)
    for fi, valid in enumerate(folds):
        train = np.setdiff1d(all_idx, valid, assume_unique=False)
        cs = CandidateSet(
            K,
            seed=seed + 1000 * fi + 1,
            device=device,
            bridge_kwargs=bridge_kwargs or {},
            candidate_names=names,
        ).fit(view.subset(train))
        vv = view.subset(valid)
        for nm, cand in cs:
            h[nm][valid] = cand.predict_h(vv.W, vv.X)
            q[nm][valid] = cand.predict_q(vv.V, vv.X)
        coverage[valid] += 1
        fold_audit.append({
            "fold": fi,
            "train": train,
            "valid": valid,
            "disjoint": len(np.intersect1d(train, valid)) == 0,
        })
    if not np.all(coverage == 1):
        raise RuntimeError("inner OOF coverage invariant failed")
    for nm in names:
        if not np.all(np.isfinite(h[nm])) or not np.all(np.isfinite(q[nm])):
            raise RuntimeError(f"incomplete OOF predictions for {nm}")

    inst_h = np.concatenate([view.V, view.X], axis=1)
    inst_q = np.concatenate([view.W, view.X], axis=1)
    bank = build_bank(inst_h, inst_q, n_rff=n_rff, seed=bank_seed)
    predictions = {nm: (h[nm], q[nm]) for nm in names}
    tests = test_many_candidates(
        bank, predictions, view, n_boot=n_boot, bootstrap_seed=bootstrap_seed
    )
    phi = {nm: compute_phi(h[nm], q[nm], view) for nm in names}
    uncertainty = {nm: pseudo_outcome_uncertainty(phi[nm]) for nm in names}
    discrepancy = {
        nm: discrepancy_from_hq(bank, h[nm], q[nm], view, kinds=("rff_l2", "rff_max", "mmr"))
        for nm in names
    }
    return {
        "h": h,
        "q": q,
        "phi": phi,
        "tests": tests,
        "uncertainty": uncertainty,
        "discrepancy": discrepancy,
        "bank": bank,
        "bank_identity": id(bank),
        "coverage": coverage,
        "fold_audit": fold_audit,
    }


def fit_mses_on_oof(view, **kwargs) -> dict:
    """Convenience wrapper returning OOF artifacts plus the pure MSES decision."""
    alpha = float(kwargs.pop("alpha", 0.05))
    out = collect_candidate_oof(view, **kwargs)
    out["mses"] = select_mses(out["tests"], out["uncertainty"], alpha=alpha)
    return out
