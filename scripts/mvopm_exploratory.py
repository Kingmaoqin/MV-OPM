"""EXPLORATORY (post-hoc, NON-confirmatory) probe of a bias x variance validator.

MV-OPM §15/§29: the product score is variance-blind; this probe tests whether adding an
observed-data VARIANCE signal (pseudo-outcome contrast variance / ATE-CI width) to the moment
diagnostics recovers reliable selection. Invented AFTER the pilot, so it CANNOT be confirmatory.
Run on FRESH seeds disjoint from the pilot dev bank (0..19) so the hypothesis is not tested on
the data that generated it. Primary target = ATE error (theory-aligned).

    python scripts/mvopm_exploratory.py           # runs fresh seeds x scenarios, writes report
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "mvopm")

FRESH_SEEDS = list(range(500, 515))          # 15 fresh seeds, disjoint from pilot (0..19) & final (100000+)
SCEN = {"S1": 4, "S2": 4, "nonlinear": 2, "hamd": 4}
EPS = 1e-8


def _scores(sub):
    """sub: DataFrame of candidates for one (scenario,seed). Return dict score_name -> {cand:val} (lower=better)."""
    Dh = sub["D_h_rff_l2"].to_numpy(); Dq = sub["D_q_rff_l2"].to_numpy(); pv = sub["po_var"].to_numpy()
    z = lambda x: (np.log(x + EPS) - np.log(x + EPS).mean())
    names = sub["candidate"].tolist()
    S = {
        "product": np.log(Dh + EPS) + np.log(Dq + EPS),                       # prespecified (bias only)
        "biasvar_mult": np.log(Dh + EPS) + np.log(Dq + EPS) + np.log(pv + EPS),  # bias x variance
        "biasvar_add": z(Dh * Dq) + z(pv),                                    # bias + variance (z-summed)
        "variance_only": np.log(pv + EPS),
        "fixed_sieve1": np.array([0 if n == "sieve1_sieve1" else 1 for n in names]),
        "fixed_kernel": np.array([0 if n == "kernel_kernel" else 1 for n in names]),
    }
    return {k: dict(zip(names, v)) for k, v in S.items()}


def run():
    from opm.experiments.mvopm.core import SelCfg, evaluate_candidates_oof
    from opm.experiments.mvopm.scenarios import generate
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
    bridge = dict(max_epochs=50, patience=15, eval_every=3, batch_size=512)
    head = dict(max_epochs=90, patience=10)
    recs = []
    for scen, K in SCEN.items():
        for seed in FRESH_SEEDS:
            t0 = time.time()
            ds = generate(scen, 1600, seed, c_U=1.0)
            cfg = SelCfg(K=K, n_folds=3, seed=seed, kind="rff_l2", n_rff=160,
                         bridge_kwargs=bridge, head_kwargs=head)
            rows = evaluate_candidates_oof(ds, cfg)
            for nm, d in rows.items():
                recs.append({"scenario": scen, "seed": seed, "candidate": nm,
                             "D_h_rff_l2": d["D_h_rff_l2"], "D_q_rff_l2": d["D_q_rff_l2"],
                             "po_var": d["po_var"], "ate_err": d["ate_err"], "pehe": d["pehe"]})
            print(f"[explore] {scen} seed{seed} {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(OUT, "explore_candidates.csv"), index=False)
    return df


def analyze(df):
    target = "ate_err"
    score_names = ["product", "biasvar_mult", "biasvar_add", "variance_only", "fixed_sieve1", "fixed_kernel"]
    per = {scen: {s: {"spearman": [], "ratio": [], "cat": [], "top1": []} for s in score_names} for scen in SCEN}
    for (scen, seed), sub in df.groupby(["scenario", "seed"]):
        err = sub.set_index("candidate")[target]
        best = err.min(); oracle_best = err.idxmin()
        sc = _scores(sub)
        for s in score_names:
            order = sorted(sc[s], key=sc[s].get)
            chosen = order[0]
            per[scen][s]["ratio"].append(float(err[chosen] / (best + 1e-12)))
            per[scen][s]["cat"].append(int(err[chosen] > 1.5 * best))
            per[scen][s]["top1"].append(int(chosen == oracle_best))
            sv = np.array([sc[s][c] for c in sub["candidate"]])
            per[scen][s]["spearman"].append(float(spearmanr(sv, sub[target]).correlation))
    lines = ["# MV-OPM — EXPLORATORY bias×variance probe (POST-HOC, NON-confirmatory)", "",
             "Fresh seeds 500–514 (disjoint from pilot). Primary target = ATE error. This was invented",
             "after the pilot HOLD and **cannot be reported as confirmatory** (MV-OPM §15/§29); it only",
             "tests whether the prereg direction is worth a future confirmatory study.", "",
             "## Median oracle-ratio (ATE) and mean rank-Spearman(score, ATE err), by scenario", ""]
    rows = []
    for scen in SCEN:
        for s in score_names:
            d = per[scen][s]
            rows.append({"scenario": scen, "score": s,
                         "median_ratio": round(float(np.median(d["ratio"])), 3),
                         "mean_spearman": round(float(np.nanmean(d["spearman"])), 3),
                         "cat_rate": round(float(np.mean(d["cat"])), 3),
                         "top1_rate": round(float(np.mean(d["top1"])), 3)})
    tab = pd.DataFrame(rows)
    lines += [tab.to_markdown(index=False), ""]
    # headline comparison
    def agg(s, col):
        return float(np.median([r[col] for r in rows if r["score"] == s]))
    lines += ["## Exploratory read (labelled exploratory)",
              f"- product (prereg, bias only): median oracle-ratio {agg('product','median_ratio'):.2f}, "
              f"mean Spearman {agg('product','mean_spearman'):.2f}",
              f"- biasvar_mult (bias×var):     median oracle-ratio {agg('biasvar_mult','median_ratio'):.2f}, "
              f"mean Spearman {agg('biasvar_mult','mean_spearman'):.2f}",
              f"- biasvar_add  (bias+var):     median oracle-ratio {agg('biasvar_add','median_ratio'):.2f}, "
              f"mean Spearman {agg('biasvar_add','mean_spearman'):.2f}",
              f"- fixed_sieve1 (strong baseline): median oracle-ratio {agg('fixed_sieve1','median_ratio'):.2f}", ""]
    with open(os.path.join(OUT, "EXPLORATORY_BIASVAR.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(OUT, "EXPLORATORY_BIASVAR.md"))
    print(tab.to_string(index=False))


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    df = run()
    analyze(df)
