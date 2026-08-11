"""MV-OPM evaluation orchestrator (oracle-aware; lives OUTSIDE the pure validation package).

`evaluate_candidates_oof` computes, per candidate, held-out (OOF) identifying-moment
discrepancies AND the candidate's oracle causal error, in a single cross-fit pass. `selection_metrics`
applies every prespecified selector + competitor and returns ranking/regret/oracle-ratio metrics.

The selector functions it calls (`opm.validation.selector/moments`) never see oracle truth; this
module is the evaluation layer permitted to read `opm.eval.oracle`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import kendalltau, spearmanr

from ...candidates.library import CANDIDATE_NAMES, CandidateSet
from ...data.views import observed
from ...diagnostics.diagnostics import q_sanity
from ...estimator.crossfit import make_folds
from ...estimator.pseudo_outcome import compute_phi
from ...estimator.tau_head import fit_cate_and_ate
from ...eval import oracle
from ...validation.moments import build_bank, evaluate_candidate
from ...validation.selector import SCORE_NAMES, compute_scores, select


@dataclass
class SelCfg:
    K: int
    n_folds: int = 5
    seed: int = 0
    device: str = "cpu"
    kind: str = "rff_l2"
    n_rff: int = 200
    bank_seed: int = 20240
    candidate_names: list = None
    bridge_kwargs: dict = field(default_factory=dict)
    head_kwargs: dict = field(default_factory=dict)


def _inst_h(v): return np.concatenate([v.V, v.X], axis=1)
def _inst_q(v): return np.concatenate([v.W, v.X], axis=1)


def evaluate_candidates_oof(ds, cfg: SelCfg) -> dict:
    """Per-candidate OOF moment discrepancies + oracle causal error. Returns name -> metrics."""
    view = observed(ds)
    n, K = view.n, view.K
    names = sorted(cfg.candidate_names) if cfg.candidate_names else list(CANDIDATE_NAMES)
    folds = make_folds(n, cfg.n_folds, cfg.seed)
    phi = {nm: np.zeros((n, K)) for nm in names}
    q_oof = {nm: np.zeros((n, K)) for nm in names}
    D_acc = {nm: [] for nm in names}
    all_idx = np.arange(n)
    for fi, te in enumerate(folds):
        tr = np.setdiff1d(all_idx, te)
        cs = CandidateSet(K, seed=cfg.seed + 1000 * fi + 1, device=cfg.device,
                          bridge_kwargs=cfg.bridge_kwargs, candidate_names=names).fit(view.subset(tr))
        tev = view.subset(te)
        bank = build_bank(_inst_h(tev), _inst_q(tev), cfg.n_rff, cfg.bank_seed)  # held-out only
        for nm, cand in cs:
            h = cand.predict_h(tev.W, tev.X); q = cand.predict_q(tev.V, tev.X)
            phi[nm][te] = compute_phi(h, q, tev); q_oof[nm][te] = q
            D_acc[nm].append(evaluate_candidate(bank, cand, tev, kinds=(cfg.kind,)))

    has_truth = oracle.has_truth(ds)
    rows = {}
    for nm in names:
        tau_head, _, ate = fit_cate_and_ate(ds.X, phi[nm], K, cfg.device, cfg.seed, cfg.head_kwargs)
        if has_truth:
            tau_hat = tau_head.predict(ds.X)
            err = oracle.candidate_causal_error(tau_hat, [ate[k]["ate"] for k in range(1, K)], ds)
        else:
            err = {"pehe": float("nan"), "ate_err": float("nan")}      # real data: no oracle
        qd = q_sanity(q_oof[nm], ds.T, K)
        D = {key: float(np.mean([d[key] for d in D_acc[nm]])) for key in D_acc[nm][0]}
        rows[nm] = {**D, **err,
                    "ate1": ate[1]["ate"], "ate1_lo": ate[1]["ci_low"], "ate1_hi": ate[1]["ci_high"],
                    "q_balance": float(np.mean([abs(qd[k]["En_1Tk_qk"] - 1.0) for k in range(K)])),
                    "ess": float(np.mean([qd[k]["ess"] for k in range(K)])),
                    "max_q": float(np.max([qd[k]["max_q"] for k in range(K)]))}
    return rows


def selection_metrics(rows: dict, cfg: SelCfg, target: str = "pehe") -> dict:
    """Apply every selector + competitor. Returns a flat metrics dict for one (scenario, seed)."""
    names = list(rows)
    err = np.array([rows[n][target] for n in names])
    oracle_best = names[int(np.argmin(err))]
    best_err = float(err.min())
    scores = compute_scores(rows, cfg.kind)                     # validator scores

    # competitor selectors (need candidate identity / q-diagnostics; NOT oracle for selection)
    rng = np.random.default_rng(cfg.seed + 999)
    comp = {
        "fixed_kernel": "kernel_kernel" if "kernel_kernel" in names else names[0],
        "fixed_sieve1": "sieve1_sieve1" if "sieve1_sieve1" in names else names[0],
        "random": names[int(rng.integers(len(names)))],
        "ess": names[int(np.argmax([rows[n]["ess"] for n in names]))],
        "qbalance": names[int(np.argmin([rows[n]["q_balance"] for n in names]))],
        "oracle": oracle_best,
    }
    selectors = {s: select(scores[s]) for s in SCORE_NAMES}
    selectors.update(comp)

    out = {"target": target, "oracle_best": oracle_best, "best_err": best_err,
           "n_candidates": len(names)}
    for sname, chosen in selectors.items():
        sel_err = float(rows[chosen][target])
        out[f"sel[{sname}]"] = chosen
        out[f"err[{sname}]"] = sel_err
        out[f"regret[{sname}]"] = sel_err - best_err
        out[f"ratio[{sname}]"] = sel_err / (best_err + 1e-12)
        out[f"cat[{sname}]"] = int(sel_err > 1.5 * best_err)          # catastrophic (>1.5x oracle)
        out[f"top1[{sname}]"] = int(chosen == oracle_best)
    # ranking quality of validator scores vs causal error (across candidates)
    err_rank = err
    for s in SCORE_NAMES:
        sv = np.array([scores[s][n] for n in names])
        out[f"spearman[{s}]"] = float(spearmanr(sv, err_rank).correlation)
        out[f"kendall[{s}]"] = float(kendalltau(sv, err_rank).correlation)
        # top-2 inclusion: is oracle-best among the 2 lowest-score candidates?
        top2 = set(np.array(names)[np.argsort(sv)[:2]])
        out[f"top2[{s}]"] = int(oracle_best in top2)
    return out
