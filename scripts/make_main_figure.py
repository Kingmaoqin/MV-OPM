"""Publication-quality 4-panel main figure from the experiment raw.csv files.

    python scripts/make_main_figure.py   ->  results/figures/main_figure.png

Panels: (a) E9 proximal necessity, (b) E6(f) proximal gain vs confounding,
(c) E10 coverage -> nominal, (d) E1 consistency/rates.
Skips any panel whose raw data is missing.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def _read(exp, name="raw.csv"):
    p = os.path.join(RES, exp, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) E9 — proximal necessity (ATE error, the clean win)
    ax = axes[0][0]; df = _read("E9")
    if df is not None:
        order = ["OPM-proximal", "OPM-dr_fallback", "sieve1-PDR", "DR-learner", "CausalForest", "DragonNet"]
        order = [m for m in order if m in df.method.unique()]
        ate = [df[df.method == m]["ate_err"].mean() for m in order]
        pehe = [df[df.method == m]["pehe"].mean() for m in order]
        x = np.arange(len(order)); w = 0.38
        ax.bar(x - w / 2, ate, w, label="ATE error", color="#1b7837")
        ax.bar(x + w / 2, pehe, w, label="PEHE", color="#a6dba0")
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("error"); ax.legend(fontsize=8)
        ax.set_title("(a) Proximal necessity under nonlinear confounding (E9)")
    else:
        ax.set_visible(False)

    # (b) E6(f) — proximal ATE-bias advantage grows with c_U
    ax = axes[0][1]; df = _read("E6")
    if df is not None and "ablation" in df.columns:
        f = df[df.ablation == "f_cu_sweep"].sort_values("setting")
        if len(f):
            ax.plot(f["setting"], f["prox_ate"], "o-", color="#1b7837", label="proximal ATE err")
            ax.plot(f["setting"], f["fb_ate"], "s--", color="#b2182b", label="dr_fallback ATE err")
            ax.set_xlabel("confounding strength c_U"); ax.set_ylabel("ATE error"); ax.legend(fontsize=8)
            ax.set_title("(b) Proximal gain grows with confounding (E6f)")
    else:
        ax.set_visible(False)

    # (c) E10 — coverage approaches nominal
    ax = axes[1][0]; sub = _read("E10")
    ptp = _read("E10", "raw_pointwise.csv")
    if sub is not None and "n" in sub.columns:
        ng = sorted(sub.n.unique())
        sc = [sub[sub.n == n]["cover"].mean() for n in ng]
        ax.axhline(0.95, color="k", ls="--", lw=1, label="nominal 0.95")
        ax.plot(ng, sc, "o-", color="#1b7837", label="subgroup-ATE")
        if ptp is not None:
            pc = [ptp[ptp.n == n]["cover"].mean() for n in ng]
            ax.plot(ng, pc, "s--", color="#762a83", label="pointwise CATE")
        ax.set_xscale("log"); ax.set_ylim(0.7, 1.02)
        ax.set_xlabel("n (log)"); ax.set_ylabel("95%-CI coverage"); ax.legend(fontsize=8)
        ax.set_title("(c) Valid effect inference: coverage -> nominal (E10)")
    else:
        ax.set_visible(False)

    # (d) E1 — consistency & rates
    ax = axes[1][1]; df = _read("E1")
    if df is not None:
        g = df.groupby("n").agg(bias=("ate", lambda s: abs(s.mean() - 2.0)), h_l2=("h_l2", "mean")).reset_index()
        ax.loglog(g["n"], g["bias"], "o-", color="#1b7837", label="|ATE bias|")
        ax.loglog(g["n"], g["h_l2"], "s-", color="#4393c3", label="bridge L2 error")
        ax.set_xlabel("n (log)"); ax.set_ylabel("error (log)"); ax.legend(fontsize=8)
        ax.set_title("(d) Consistency & rates (E1)")
    else:
        ax.set_visible(False)

    fig.suptitle("OPM v2 — proximal identification removes confounding bias ignorability cannot; "
                 "kernel bridge is robust, not superior", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    out = os.path.join(RES, "figures", "main_figure.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
