"""Aggregate the Round-3 selector-variant sweep into summary tables and a markdown report.

    python scripts/mvopm_variant_aggregate.py results/mvopm_variants

Reads every ``raw_*.jsonl`` in the directory, pools the selector rows, and reports primary
(ATE oracle-ratio / regret / catastrophic / top1) and NON-PRIMARY (PEHE ratio, CI coverage,
deployed q-safety) targets, per scenario and pooled. Also reports the structural ATE-vs-PEHE
oracle-agreement rate from the candidate provenance lines. Nothing here is tuned; it only scores
selections against held-out DGP truth.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CATASTROPHIC = 1.5


def _load(d: str):
    sel, cand, errs = [], [], []
    for path in sorted(glob.glob(os.path.join(d, "raw_*.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            if r.get("selector"):
                sel.append(r)
            elif r.get("record") == "candidates":
                cand.append(r)
            elif r.get("record") == "error":
                errs.append(r)
    return pd.DataFrame(sel), cand, errs


def _agg(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    ab = int(df["abstained"].sum())
    ev = df[~df["abstained"]]                       # evaluable (non-abstain) rows
    r = ev["oracle_ratio"].astype(float)
    reg = ev["regret"].astype(float)
    pr = ev["pehe_ratio"].astype(float)
    # effect-size-normalized severity (primary, review-recommended); fall back gracefully if absent
    nreg = ev["norm_regret"].astype(float) if "norm_regret" in ev else pd.Series(dtype=float)
    nerr = ev["norm_error"].astype(float) if "norm_error" in ev else pd.Series(dtype=float)
    ncat = ev["norm_catastrophic"].astype(float) if "norm_catastrophic" in ev else pd.Series(dtype=float)
    return pd.Series({
        "n": n,
        "abstain_rate": ab / n if n else np.nan,
        # --- primary: normalized by true effect size ---
        "median_norm_regret": nreg.median() if len(nreg) else np.nan,
        "mean_norm_regret": nreg.mean() if len(nreg) else np.nan,
        "median_norm_error": nerr.median() if len(nerr) else np.nan,
        "norm_catastrophic_rate": ncat.mean() if len(ncat) else np.nan,
        "median_abs_regret": reg.median(),
        "median_abs_error": ev["error"].astype(float).median(),
        # --- legacy oracle-ratio (denominator-inflated; kept for continuity) ---
        "median_oracle_ratio": r.median(),
        "mean_oracle_ratio": r.mean(),
        "p90_oracle_ratio": r.quantile(0.90),
        "catastrophic_rate": ev["catastrophic"].astype(float).mean(),
        "top1_rate": ev["top1"].astype(float).mean(),
        "median_pehe_ratio": pr.median(),
        "pehe_catastrophic_rate": ev["pehe_catastrophic"].astype(float).mean(),
        "mean_ate_ci_coverage": ev["coverage"].astype(float).mean(),
        "mean_max_q": ev["max_q"].astype(float).mean(),
        "mean_ess": ev["ess"].astype(float).mean(),
        "mean_q_balance": ev["q_balance"].astype(float).mean(),
    })


def _fmt(df: pd.DataFrame, cols) -> str:
    d = df.copy()
    for c in d.columns:
        if c in cols:
            d[c] = d[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    return d.to_markdown()


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/mvopm_variants"
    sel, cand, errs = _load(d)
    if sel.empty:
        raise SystemExit(f"no selector rows found in {d}")

    # rank by effect-size-normalized regret when available (review fix), else legacy ratio
    order_by = "median_norm_regret" if "norm_regret" in sel.columns else "median_oracle_ratio"
    pooled = sel.groupby("selector").apply(_agg, include_groups=False).sort_values(order_by)
    pooled.to_csv(os.path.join(d, "summary_pooled.csv"))

    by = (sel.groupby(["scenario", "selector"]).apply(_agg, include_groups=False)
          .reset_index().sort_values(["scenario", order_by]))
    by.to_csv(os.path.join(d, "summary_by_scenario.csv"), index=False)

    # structural ATE-vs-PEHE oracle agreement (candidate provenance)
    agree = []
    for c in cand:
        ce, cp = c["candidate_error"], c["candidate_pehe"]
        ate_best = min(ce, key=ce.get)
        pehe_best = min(cp, key=cp.get)
        agree.append({"scenario": c["scenario"], "match": int(ate_best == pehe_best)})
    agree_df = pd.DataFrame(agree)
    agree_by = agree_df.groupby("scenario")["match"].mean() if not agree_df.empty else pd.Series(dtype=float)

    # Mechanistic crux: per scenario, pool all (candidate, task) points and correlate each SELECTION
    # SCORE with true ATE error. A useful score is POSITIVELY correlated with error (low score ->
    # low error -> you pick it). rho(variance) should be > 0; rho(moment-compatibility) near 0 or
    # wrong-signed in nonlinear regimes is exactly why moment-trusting selectors fail there.
    corr_rows = []
    scen_pts: dict = {}
    for c in cand:
        s = c["scenario"]
        ce, cv, cp = c["candidate_error"], c["candidate_var"], c["candidate_p_dr_min"]
        cqb = c.get("candidate_q_balance", {})
        cmq = c.get("candidate_max_q", {})
        for nm in ce:
            scen_pts.setdefault(s, {"err": [], "var": [], "pdr": [], "qb": [], "mq": []})
            scen_pts[s]["err"].append(ce[nm])
            scen_pts[s]["var"].append(cv[nm])
            scen_pts[s]["pdr"].append(cp[nm])
            scen_pts[s]["qb"].append(cqb.get(nm, np.nan))
            scen_pts[s]["mq"].append(cmq.get(nm, np.nan))
    for s, p in sorted(scen_pts.items()):
        err = np.array(p["err"], float)
        rv = spearmanr(np.array(p["var"], float), err).correlation
        # moment VIOLATION = -compatibility, oriented like variance so LOW = what the rule prefers;
        # positive rho therefore means "informative" for both scores.
        rp = spearmanr(-np.array(p["pdr"], float), err).correlation
        qb = np.array(p["qb"], float)
        mq = np.array(p["mq"], float)
        rqb = spearmanr(qb, err).correlation if np.isfinite(qb).all() else np.nan
        rmq = spearmanr(mq, err).correlation if np.isfinite(mq).all() else np.nan
        corr_rows.append({"scenario": s, "rho_variance_vs_error": rv,
                          "rho_moment_violation_vs_error": rp,
                          "rho_q_balance_vs_error": rqb, "rho_max_q_vs_error": rmq,
                          "n_points": len(err)})
    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty:
        corr_df.to_csv(os.path.join(d, "score_error_correlations.csv"), index=False)

    metric_cols = [c for c in pooled.columns if c != "n"]
    lines = []
    lines.append("# Round-3 exploratory selector-variant sweep\n")
    lines.append("Fresh seeds (base 77,000,000), disjoint from the Round-2 confirmatory final "
                 "seeds. No selector is tuned here; every number scores a selection against "
                 "held-out DGP truth. This does not alter the frozen Round-2 verdict.\n")
    scen = sorted(sel["scenario"].unique())
    seeds = sorted(sel["seed"].unique())
    lines.append(f"- Scenarios: {', '.join(scen)}")
    lines.append(f"- Seeds per scenario: {len(seeds)} (fresh block)")
    lines.append(f"- Selectors: {sel['selector'].nunique()}")
    if errs:
        lines.append(f"- Task failures recorded: {len(errs)}")
    lines.append("")
    lines.append("## Pooled over all scenarios (sorted by median ATE oracle-ratio; 1.000 = oracle)\n")
    lines.append(_fmt(pooled.reset_index(), metric_cols))
    lines.append("")
    lines.append("## Nonlinear regime (the Round-2 decisive failure D2/E9)\n")
    nl = by[by["scenario"] == "nonlinear"].drop(columns="scenario").set_index("selector")
    if not nl.empty:
        lines.append(_fmt(nl.reset_index(), metric_cols))
    lines.append("")
    lines.append("## ATE-oracle vs PEHE-oracle agreement (structural; candidate level)\n")
    if not agree_by.empty:
        lines.append("| scenario | P(ATE-best == PEHE-best) |")
        lines.append("|---|---:|")
        for s, v in agree_by.items():
            lines.append(f"| {s} | {v:.3f} |")
    lines.append("")
    lines.append("## Why moment-trusting selectors fail: score-vs-error rank correlation\n")
    lines.append("Spearman rho between a candidate's selection score and its true ATE error, pooled "
                 "over candidates and seeds within a scenario. Positive rho = informative (low "
                 "score picks low error). Variance stays informative everywhere; moment "
                 "compatibility does not.\n")
    if not corr_df.empty:
        lines.append("| scenario | rho(variance, err) | rho(moment-violation, err) | "
                     "rho(q_balance, err) | rho(max_q, err) | n |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in corr_df.iterrows():
            qb = row.get("rho_q_balance_vs_error", float("nan"))
            mq = row.get("rho_max_q_vs_error", float("nan"))
            qbs = f"{qb:.3f}" if pd.notna(qb) else "-"
            mqs = f"{mq:.3f}" if pd.notna(mq) else "-"
            lines.append(f"| {row['scenario']} | {row['rho_variance_vs_error']:.3f} | "
                         f"{row['rho_moment_violation_vs_error']:.3f} | {qbs} | {mqs} | "
                         f"{int(row['n_points'])} |")
    lines.append("")
    with open(os.path.join(d, "EXPLORATORY_VARIANTS_REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote summary_pooled.csv, summary_by_scenario.csv, EXPLORATORY_VARIANTS_REPORT.md")
    print("\nPOOLED (PRIMARY = effect-size-normalized regret; legacy oracle-ratio for reference):")
    cols = [c for c in ["median_norm_regret", "norm_catastrophic_rate", "median_abs_regret",
                        "median_oracle_ratio", "catastrophic_rate", "top1_rate",
                        "median_pehe_ratio"] if c in pooled.columns]
    print(pooled[cols].round(3).to_string())


if __name__ == "__main__":
    main()
