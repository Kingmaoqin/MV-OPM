"""Emit a consolidated LaTeX table of headline results into results/tables_all.tex.

Reads each experiment's raw.csv. Safe to run partially (skips missing experiments).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def e3_table():
    p = os.path.join(RES, "E3", "raw.csv")
    if not os.path.exists(p):
        return ""
    df = pd.read_csv(p)
    lines = ["% E3 main benchmark PEHE (mean +/- sd over seeds)",
             "\\begin{tabular}{l" + "c" * df.scenario.nunique() + "}", "\\hline",
             "Method & " + " & ".join(sorted(df.scenario.unique())) + " \\\\ \\hline"]
    for m in ["OPM"] + [x for x in df.method.unique() if x != "OPM"]:
        cells = []
        for scen in sorted(df.scenario.unique()):
            v = df[(df.scenario == scen) & (df.method == m)]["pehe"]
            cells.append(f"{v.mean():.3f}$\\pm${v.std():.3f}" if len(v) else "--")
        lines.append(f"{m} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", ""]
    return "\n".join(lines)


def e4_table():
    p = os.path.join(RES, "E4", "raw.csv")
    if not os.path.exists(p):
        return ""
    df = pd.read_csv(p)
    if "estimator" not in df.columns:
        return ""
    lines = ["% E4 RHC ATE (days), negative = RHC shortens survival",
             "\\begin{tabular}{lccc}", "\\hline",
             "Estimator & ATE & CI low & CI high \\\\ \\hline"]
    for _, r in df.iterrows():
        lines.append(f"{r['estimator']} & {r['ate']:.3f} & {r['ci_low']:.3f} & {r['ci_high']:.3f} \\\\")
    lines += ["\\hline", "\\end{tabular}", ""]
    return "\n".join(lines)


def main():
    parts = [e3_table(), e4_table()]
    out = os.path.join(RES, "tables_all.tex")
    with open(out, "w") as f:
        f.write("\n".join(p for p in parts if p))
    print("wrote", out)


if __name__ == "__main__":
    main()
