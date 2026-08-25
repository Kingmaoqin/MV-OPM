"""Observed-data-only convergence audit for the kernel candidate.

Oracle metrics are intentionally absent from this module so they cannot determine the budget.
"""
from __future__ import annotations

import numpy as np

from ...candidates.library import CandidateSet
from ...data.views import observed
from ...estimator.crossfit import make_folds
from ...validation.moments import build_bank, evaluate_candidate
from ..mvopm.scenarios import generate

BUDGETS = (45, 90, 180, 300)


def audit_one(
    scenario: str,
    seed: int,
    *,
    n: int = 1500,
    device: str = "cpu",
    budgets: tuple[int, ...] = BUDGETS,
    batch_size: int = 1024,
) -> list[dict]:
    """Evaluate held-out moments/objective/prediction stability across fixed epoch caps."""
    if not budgets or any(int(budget) < 1 for budget in budgets):
        raise ValueError("budgets must be nonempty positive integers")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ds = generate(scenario, n, seed, c_U=1.0)
    view = observed(ds)
    valid = make_folds(view.n, 5, seed)[0]
    train = np.setdiff1d(np.arange(view.n), valid)
    tv, vv = view.subset(train), view.subset(valid)
    inst_h = np.concatenate([vv.V, vv.X], axis=1)
    inst_q = np.concatenate([vv.W, vv.X], axis=1)
    bank = build_bank(inst_h, inst_q, n_rff=200, seed=880000000 + seed)
    out, previous = [], None
    for budget in budgets:
        cs = CandidateSet(
            view.K,
            seed=seed,
            device=device,
            candidate_names=["kernel_kernel"],
            bridge_kwargs={
                "max_epochs": budget,
                "patience": max(15, int(round(0.15 * budget))),
                "eval_every": 3,
                "batch_size": int(batch_size),
            },
        ).fit(tv)
        cand = cs.get("kernel_kernel")
        h, q = cand.predict_h(vv.W, vv.X), cand.predict_q(vv.V, vv.X)
        D = evaluate_candidate(bank, cand, vv, kinds=("rff_l2", "mmr"))
        if previous is None:
            h_stability = q_stability = float("nan")
        else:
            ph, pq = previous
            h_stability = float(np.sqrt(np.mean((h - ph) ** 2)) / (np.std(ph) + 1e-8))
            q_stability = float(np.sqrt(np.mean((q - pq) ** 2)) / (np.std(pq) + 1e-8))
        solver = cs._solvers["kernel"]
        parameter_l2 = float(np.sqrt(sum(
            float(np.sum(p.detach().cpu().numpy() ** 2))
            for model in (solver.net_h, solver.net_q)
            for p in model.parameters()
        )))
        out.append({
            "scenario": scenario,
            "seed": seed,
            "budget": budget,
            "D_total_rff_l2": D["D_h_rff_l2"] + D["D_q_rff_l2"],
            "D_total_mmr": D["D_h_mmr"] + D["D_q_mmr"],
            "h_prediction_change": h_stability,
            "q_prediction_change": q_stability,
            "optimizer_best_resid": float(solver.best_resid),
            "optimizer_min_training_objective": float(solver.min_training_objective),
            "optimizer_final_training_objective": float(solver.final_training_objective),
            "optimizer_updates": len(solver.training_objective_trace),
            "training_objective_scope": "minibatch_vstat_with_diagonal_regularization",
            "training_batch_size_cap": int(solver.cfg.batch_size),
            "parameter_l2": parameter_l2,
            "h_prediction_min": float(np.min(h)),
            "h_prediction_max": float(np.max(h)),
            "h_prediction_rms": float(np.sqrt(np.mean(h ** 2))),
            "q_prediction_min": float(np.min(q)),
            "q_prediction_max": float(np.max(q)),
            "q_prediction_rms": float(np.sqrt(np.mean(q ** 2))),
        })
        previous = h, q
    return out


def choose_plateau_budget(rows: list[dict], *, rel_tol: float = 0.05, stability_tol: float = 0.05) -> int:
    """Smallest observed-data plateau; returns 300 when no earlier cap qualifies."""
    budgets = sorted({int(r["budget"]) for r in rows})
    for current, nxt in zip(budgets[:-1], budgets[1:]):
        cur = np.array([r["D_total_rff_l2"] for r in rows if r["budget"] == current])
        new = np.array([r["D_total_rff_l2"] for r in rows if r["budget"] == nxt])
        hchg = np.array([r["h_prediction_change"] for r in rows if r["budget"] == nxt])
        qchg = np.array([r["q_prediction_change"] for r in rows if r["budget"] == nxt])
        improvement = (np.median(cur) - np.median(new)) / max(abs(np.median(cur)), 1e-8)
        stable = max(float(np.nanmedian(hchg)), float(np.nanmedian(qchg))) < stability_tol
        if improvement < rel_tol and stable:
            return current
    return 300
