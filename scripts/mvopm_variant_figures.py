"""Figures for the Round-3 selector-variant sweep. Reads the aggregator CSVs; writes PNGs.

    python scripts/mvopm_variant_figures.py results/mvopm_variants
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# colorblind-safe (Okabe-Ito)
CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]


def fig_regime_ratio(by: pd.DataFrame, out: str):
    """Median ATE oracle-ratio per scenario for a focused selector panel (log-y)."""
    focus = ["variance_only", "MSES", "varq_combo", "varq_moment_combo", "q_gated_variance"]
    scen = sorted(by["scenario"].unique())
    focus = [s for s in focus if s in set(by["selector"])]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.8 / len(focus)
    x = np.arange(len(scen))
    for i, sel in enumerate(focus):
        vals = [by[(by.scenario == sc) & (by.selector == sel)]["median_oracle_ratio"].values
                for sc in scen]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + i * w, vals, w, label=sel, color=CB[i % len(CB)])
    ax.axhline(1.0, color="#444", lw=1, ls="--", label="oracle (=1.0)")
    ax.set_yscale("log")
    ax.set_xticks(x + w * (len(focus) - 1) / 2)
    ax.set_xticklabels(scen)
    ax.set_ylabel("median ATE oracle-ratio (log)")
    ax.set_title("Selector performance is regime-dependent (no rule dominates)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_correlations(corr: pd.DataFrame, out: str):
    """Score-vs-error rank correlations per scenario: the mechanistic crux."""
    cols = [("rho_variance_vs_error", "variance"),
            ("rho_moment_violation_vs_error", "moment-violation"),
            ("rho_q_balance_vs_error", "q_balance"),
            ("rho_max_q_vs_error", "max_q")]
    cols = [(c, lab) for c, lab in cols if c in corr.columns]
    scen = list(corr["scenario"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.8 / len(cols)
    x = np.arange(len(scen))
    for i, (c, lab) in enumerate(cols):
        ax.bar(x + i * w, corr[c].values, w, label=lab, color=CB[i % len(CB)])
    ax.axhline(0.0, color="#444", lw=1)
    ax.set_xticks(x + w * (len(cols) - 1) / 2)
    ax.set_xticklabels(scen)
    ax.set_ylabel("Spearman rho vs true ATE error")
    ax.set_title("Is the signal informative? (positive = low score picks low error)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/mvopm_variants"
    figdir = os.path.join(d, "figures")
    os.makedirs(figdir, exist_ok=True)
    by = pd.read_csv(os.path.join(d, "summary_by_scenario.csv"))
    fig_regime_ratio(by, os.path.join(figdir, "fig1_regime_oracle_ratio.png"))
    corr_path = os.path.join(d, "score_error_correlations.csv")
    if os.path.exists(corr_path):
        fig_correlations(pd.read_csv(corr_path), os.path.join(figdir, "fig2_score_error_corr.png"))
    print(f"wrote figures to {figdir}")


if __name__ == "__main__":
    main()
