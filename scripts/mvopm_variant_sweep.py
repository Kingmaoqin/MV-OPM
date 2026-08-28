"""Round-3 EXPLORATORY selector-variant sweep.

For each (scenario, fresh seed) this fits the six-candidate library once via oracle-isolated OOF
cross-fitting (``evaluate_candidates_round2``), extracts strictly non-oracle candidate features,
applies the full exploratory selector registry, and then -- separately, using DGP truth held out
of selection -- scores each selector on primary and NON-PRIMARY targets:

    ATE       : oracle_ratio, regret, catastrophic(>1.5x), top1
    CATE/PEHE : pehe_ratio, pehe_catastrophic         (never used for selection)
    coverage  : does the selected candidate's 95% Wald ATE CI cover the true ATE
    safety    : deployed max_q, ESS, q_balance
    abstention: for screen-based rules

Fresh seeds only (default base 77_000_000), disjoint from the Round-2 confirmatory final seeds; no
selector is tuned here and the frozen confirmatory verdict is untouched. Writes one JSONL row per
(scenario, seed, selector) plus a per-task candidate summary line.

    python scripts/mvopm_variant_sweep.py --scenario nonlinear --seed-start 77000000 \
        --seed-count 16 --n 2000 --n-boot 299 --out results/mvopm_variants/raw_nonlinear.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from opm.experiments.mvopm.scenarios import generate  # noqa: E402
from opm.experiments.mvopm_round2.core import Round2Config, evaluate_candidates_round2  # noqa: E402
from opm.validation.selector_variants import ABSTAIN, build_registry, candidate_complexity  # noqa: E402

CATASTROPHIC = 1.5  # error > 1.5 x oracle-best  (Round-2 preregistered definition)


def _features(rows: dict) -> dict:
    """Extract ONLY non-oracle candidate features for the selectors."""
    feat = {}
    for name, r in rows.items():
        feat[name] = {
            "p_h": list(r["p_h"]),
            "p_q": list(r["p_q"]),
            "d_h": float(r["D_h_rff_l2"]),
            "d_q": float(r["D_q_rff_l2"]),
            "var": float(r["po_var_mean"]),
            "var_max": float(r["po_var_max"]),
            "ci_width": float(np.mean(r["ate_ci_width_by_contrast"])),
            "q_balance": float(r["q_balance"]),
            "max_q": float(r["max_q"]),
            "ess": float(r["ess"]),
        }
    return feat


def _covers(row: dict, ate_true: np.ndarray) -> float:
    lo = np.asarray(row["ate_lo"], float)
    hi = np.asarray(row["ate_hi"], float)
    t = np.asarray(ate_true, float).reshape(-1)
    return float(np.mean((t >= lo) & (t <= hi)))


def run_seed(scenario: str, n: int, seed: int, n_boot: int, alpha: float) -> list:
    ds = generate(scenario, n, seed)
    cfg = Round2Config(K=ds.K, n_folds=4, inner_folds=3, seed=seed, n_boot=n_boot,
                       alpha=alpha, device="cpu")
    rows = evaluate_candidates_round2(ds, cfg)["rows"]
    names = sorted(rows)
    ate_err = np.array([rows[n_]["ate_err"] for n_ in names], float)
    pehe = np.array([rows[n_]["pehe"] for n_ in names], float)
    best_err = float(np.nanmin(ate_err))
    best_pehe = float(np.nanmin(pehe))
    oracle_best = names[int(np.nanargmin(ate_err))]
    ate_true = np.asarray(ds.ate_true, float).reshape(-1)
    # Effect-size scale for NORMALIZED severity (Round-3 review fix): the raw oracle_ratio divides
    # by a near-zero oracle-best error and is unnormalized by the true effect, so it inflates
    # severity. abs regret / mean|ATE| is comparable across scenarios of different effect size.
    effect_scale = float(np.mean(np.abs(ate_true))) or 1.0

    feat = _features(rows)
    registry = build_registry(alpha=alpha)
    out = []
    for sname, fn in registry.items():
        chosen = fn(feat)
        rec = {"scenario": scenario, "seed": int(seed), "n": int(n), "K": int(ds.K),
               "selector": sname, "chosen": chosen, "oracle_best": oracle_best,
               "oracle_best_error": best_err, "oracle_best_pehe": best_pehe,
               "effect_scale": effect_scale, "abstained": chosen == ABSTAIN}
        if chosen == ABSTAIN:
            rec.update({"error": None, "regret": None, "oracle_ratio": None,
                        "norm_regret": None, "norm_error": None, "norm_catastrophic": None,
                        "catastrophic": None, "top1": 0, "pehe": None, "pehe_ratio": None,
                        "pehe_catastrophic": None, "coverage": None, "max_q": None,
                        "ess": None, "q_balance": None, "complexity": None})
        else:
            r = rows[chosen]
            err = float(r["ate_err"])
            ph = float(r["pehe"])
            rec.update({
                "error": err,
                "regret": err - best_err,
                "oracle_ratio": err / (best_err + 1e-12),
                # effect-size-normalized severity (primary, review-recommended headline)
                "norm_error": err / effect_scale,
                "norm_regret": (err - best_err) / effect_scale,
                "norm_catastrophic": int((err - best_err) / effect_scale > 0.25),
                "catastrophic": int(err > CATASTROPHIC * best_err),
                "top1": int(chosen == oracle_best),
                "pehe": ph,
                "pehe_ratio": ph / (best_pehe + 1e-12),
                "pehe_catastrophic": int(ph > CATASTROPHIC * best_pehe),
                "coverage": _covers(r, ate_true),
                "max_q": float(r["max_q"]),
                "ess": float(r["ess"]),
                "q_balance": float(r["q_balance"]),
                "complexity": candidate_complexity(chosen),
            })
        out.append(rec)
    # one candidate-level provenance line (oracle truth kept separate from selection)
    out.append({"scenario": scenario, "seed": int(seed), "n": int(n), "K": int(ds.K),
                "record": "candidates",
                "candidate_error": {nm: float(rows[nm]["ate_err"]) for nm in names},
                "candidate_pehe": {nm: float(rows[nm]["pehe"]) for nm in names},
                "candidate_var": {nm: float(rows[nm]["po_var_mean"]) for nm in names},
                "candidate_screen_pass": {nm: bool(rows[nm]["screen_pass"]) for nm in names},
                "candidate_p_dr_min": {nm: float(np.min(np.maximum(rows[nm]["p_h"], rows[nm]["p_q"])))
                                       for nm in names},
                "candidate_q_balance": {nm: float(rows[nm]["q_balance"]) for nm in names},
                "candidate_max_q": {nm: float(rows[nm]["max_q"]) for nm in names},
                "ate_true": ate_true.tolist(), "effect_scale": effect_scale,
                "oracle_best": oracle_best})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed-start", type=int, default=77_000_000)
    ap.add_argument("--seed-count", type=int, default=16)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=299)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    done = 0
    with open(args.out, "w") as f:
        for i in range(args.seed_count):
            seed = args.seed_start + i
            try:
                for rec in run_seed(args.scenario, args.n, seed, args.n_boot, args.alpha):
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                done += 1
                print(f"[{args.scenario}] seed {seed} ok ({done}/{args.seed_count}) "
                      f"{time.time()-t0:.0f}s", flush=True)
            except Exception as e:  # keep the sweep alive; record the failure
                import traceback
                f.write(json.dumps({"scenario": args.scenario, "seed": int(seed),
                                    "record": "error", "error": f"{type(e).__name__}: {e}",
                                    "traceback": traceback.format_exc()}) + "\n")
                f.flush()
                print(f"[{args.scenario}] seed {seed} FAILED: {e}", flush=True)
    print(f"[{args.scenario}] FINISHED {done}/{args.seed_count} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
