"""Analyze MV-OPM raw results: mechanism (Study A), selection studies, and the frozen Go/No-Go gate.

    python scripts/mvopm_analyze.py [pilot|final]

Reads results/mvopm/raw.csv (+ raw_candidates.csv), writes results/mvopm/GATE_REPORT.md (pilot) or
results/mvopm/FINAL_ANALYSIS.md (final). PRIMARY causal-reliability target = ATE error (theory-
aligned; see docs/reviews). PEHE reported as secondary/stress metric.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "mvopm")
SCEN_STUDY = {"B": "S1", "C": "S2", "D": "nonlinear", "E": "hamd"}
SCORES = ["product", "h_only", "q_only", "sum", "max"]
COMPET = ["fixed_kernel", "fixed_sieve1", "random", "ess", "qbalance"]


def _f(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)


def mechanism(df):
    a = df[df.study == "A"].copy()
    if not len(a):
        return None, []
    for c in ["true_err_product", "moment_product", "D_h", "D_q", "true_h_err", "true_q_err", "abs_ate_bias", "r_h", "r_q"]:
        a[c] = _f(a, c)
    lines = ["## Study A — exact product mechanism", ""]
    sp = lambda x, y: float(spearmanr(a[x], a[y]).correlation)
    res = {
        "spearman(moment_product, true_err_product)": sp("moment_product", "true_err_product"),
        "spearman(D_h, true_h_err)": sp("D_h", "true_h_err"),
        "spearman(D_q, true_q_err)": sp("D_q", "true_q_err"),
        "spearman(abs_ate_bias, moment_product)": sp("abs_ate_bias", "moment_product"),
    }
    for k, v in res.items():
        lines.append(f"- {k} = **{v:.3f}**")
    # single-side vs product corruption: mean abs ATE bias
    def mb(cond):
        s = a[cond]["abs_ate_bias"]
        return float(s.mean()) if len(s) else float("nan")
    lines += ["", "Mean |ATE bias| by corruption pattern (DR should keep single-side small):",
              f"- neither (r_h=0,r_q=0): {mb((a.r_h==0)&(a.r_q==0)):.4f}",
              f"- h-only  (r_h>0,r_q=0): {mb((a.r_h>0)&(a.r_q==0)):.4f}",
              f"- q-only  (r_h=0,r_q>0): {mb((a.r_h==0)&(a.r_q>0)):.4f}",
              f"- both    (r_h>0,r_q>0): {mb((a.r_h>0)&(a.r_q>0)):.4f}", ""]
    mech_ok = res["spearman(moment_product, true_err_product)"] > 0 and res["spearman(abs_ate_bias, moment_product)"] > 0
    return mech_ok, lines


def selection(df, target="ate"):
    lines = [f"## Selection studies — primary target = {'ATE error' if target=='ate' else 'PEHE'}", ""]
    per_scen = {}
    rows = []
    for study, scen in SCEN_STUDY.items():
        sub = df[df.study == study]
        if not len(sub):
            continue
        sp = _f(sub, f"{target}::spearman[product]").mean()
        per_scen[scen] = float(sp)
        rec = {"scenario": scen, "n": len(sub), "spearman_product": round(float(sp), 3)}
        for sel in ["oracle", "product", "h_only", "q_only", "fixed_kernel", "fixed_sieve1", "random"]:
            r = _f(sub, f"{target}::ratio[{sel}]")
            c = _f(sub, f"{target}::cat[{sel}]")
            rec[f"ratio[{sel}]"] = round(float(np.nanmedian(r)), 3) if len(r) else None
            rec[f"cat[{sel}]"] = round(float(np.nanmean(c)), 3) if len(c) else None
        rows.append(rec)
    tab = pd.DataFrame(rows)
    if len(tab):
        lines += [tab.to_markdown(index=False), ""]
    return per_scen, tab, lines


def gate(mech_ok, per_scen_ate, tab_ate):
    fams = ["S1", "S2", "nonlinear", "hamd"]
    n_pos = sum(1 for f in fams if per_scen_ate.get(f, -1) > 0.2)
    c1 = n_pos >= 3
    # criterion 2: product median ratio < fixed_kernel median ratio overall
    if len(tab_ate):
        med_prod = np.nanmedian([r for r in tab_ate.get("ratio[product]", []) if r is not None])
        med_ker = np.nanmedian([r for r in tab_ate.get("ratio[fixed_kernel]", []) if r is not None])
    else:
        med_prod = med_ker = float("nan")
    c2 = med_prod < med_ker
    c4 = bool(mech_ok)
    passed = c1 and c4  # gate: (1) rank association in >=3/4 AND (4) mechanism; (2) is supportive
    lines = ["## Go / No-Go gate (frozen criteria; primary target = ATE error)", "",
             f"- C1 rank assoc Spearman(product, ATE err) > 0.2 in >=3/4 families: "
             f"**{'PASS' if c1 else 'FAIL'}** ({n_pos}/4 families; per-family "
             f"{ {k: round(v,2) for k,v in per_scen_ate.items()} })",
             f"- C2 product median oracle-ratio < fixed-kernel: "
             f"**{'PASS' if c2 else 'FAIL'}** (product={med_prod:.3f} vs kernel={med_ker:.3f})",
             f"- C4 mechanism: moment product tracks true error product & |ATE bias|: "
             f"**{'PASS' if c4 else 'FAIL'}**",
             "", f"### GATE DECISION: **{'GO' if passed else 'HOLD'}**", ""]
    return passed, lines


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    df = pd.read_csv(os.path.join(OUT, "raw.csv"))
    mech_ok, mlines = mechanism(df)
    per_scen_ate, tab_ate, slines_ate = selection(df, "ate")
    _, _, slines_pehe = selection(df, "pehe")
    passed, glines = gate(mech_ok, per_scen_ate, tab_ate)
    doc = [f"# MV-OPM {phase.upper()} analysis", "",
           f"Rows analyzed: {len(df)}. Primary target = **ATE error** (theory-aligned); PEHE secondary.",
           ""] + glines + mlines + slines_ate + ["### Secondary (PEHE) — variance stress metric", ""] + slines_pehe[1:]
    name = "GATE_REPORT.md" if phase == "pilot" else "FINAL_ANALYSIS.md"
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(doc) + "\n")
    print("wrote", os.path.join(OUT, name), " GATE:", "GO" if passed else "HOLD")


if __name__ == "__main__":
    main()
