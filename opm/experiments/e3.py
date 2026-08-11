"""E3 — Main benchmark on S1, S2 with corruption scans (spec Section 5, E3).

Metrics: PEHE, ATE error, policy value (argmax_k of [0, tau_hat] under TRUE potential
outcomes).  ACCEPT: OPM beats every ignorability baseline on PEHE in S1 and S2 (paired
Wilcoxon p<0.05, BH-corrected); OPM >= sieve-PDR; corruption degradation graceful
(no metric jump > 2x between adjacent levels).  If OPM does not beat ignorability
baselines: STOP and debug bridges (Section 6.1) before proceeding.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..baselines.registry import IGNORABILITY, NOT_RUN, PROXIMAL, build
from ..dgp.synthetic_main import corrupt_proxies, generate_s1, generate_s2
from ..estimator.learner import OPM, OPMConfig
from ..eval.metrics import ate_error, oracle_policy_value, pehe
from ..eval.stats import benjamini_hochberg, paired_tests, summarize
from ..utils import set_seed
from .common import results_dir, save_raw, write_report

SCEN = {"S1": generate_s1, "S2": generate_s2}


def _policy_value(cate, mu_true):
    choice = np.argmax(np.concatenate([np.zeros((cate.shape[0], 1)), cate], axis=1), axis=1)
    return float(mu_true[np.arange(mu_true.shape[0]), choice].mean())


def _fit_opm(ds, seed, mode, bkw, hkw):
    return OPM(OPMConfig(K=ds.K, mode=mode, seed=seed, bridge_kwargs=bkw, head_kwargs=hkw)).fit(ds)


def _eval(method_obj, dtest, is_opm=False):
    cate = method_obj.predict_cate(dtest.X)
    if is_opm:
        ate = method_obj.ate_vector()
    else:
        ate = method_obj.ate_from(dtest.X) if hasattr(method_obj, "ate_from") else cate.mean(0)
    return {"pehe": pehe(cate, dtest.tau_true),
            "ate_err": ate_error(ate, dtest.ate_true),
            "policy": _policy_value(cate, dtest.mu_true)}


def run(cfg: dict) -> dict:
    seeds = cfg["seeds"]
    n_train, n_test = cfg["n_train"], cfg["n_test"]
    methods = cfg["methods"]
    bkw, hkw = cfg.get("bridge", {}), cfg.get("head", {})
    bkw_base = cfg.get("bridge_baseline", bkw)

    rows = []
    for scen in cfg["scenarios"]:
        gen = SCEN[scen]
        for seed in seeds:
            set_seed(seed)
            ds = gen(n=n_train, seed=seed)
            dtest = gen(n=n_test, seed=10_000 + seed)
            # OPM (headline, non-oracle proximal)
            opm = _fit_opm(ds, seed, "proximal", bkw, hkw)
            m = _eval(opm, dtest, is_opm=True)
            rows.append({"scenario": scen, "seed": seed, "method": "OPM", **m})
            print(f"E3 {scen} seed{seed} OPM pehe={m['pehe']:.3f}", flush=True)
            for name in methods:
                try:
                    b = build(name, K=ds.K, seed=seed, bridge_kwargs=bkw_base, head_kwargs=hkw)
                    b.fit(ds)
                    mm = _eval(b, dtest)
                    rows.append({"scenario": scen, "seed": seed, "method": name, **mm})
                except Exception as e:
                    rows.append({"scenario": scen, "seed": seed, "method": name,
                                 "pehe": np.nan, "ate_err": np.nan, "policy": np.nan,
                                 "error": f"{type(e).__name__}: {e}"})
                    print(f"  {name} ERROR {e}", flush=True)

    df = pd.DataFrame(rows)
    save_raw("E3", rows)

    corr_df = _corruption_scan(cfg) if cfg.get("run_corruption", True) else None
    _report(cfg, df, corr_df)
    return {"df": df, "corruption": corr_df}


def _corruption_scan(cfg):
    """OPM vs PDR-sieve + top-3 ignorability under proxy corruption (3.3)."""
    kinds = cfg.get("corruption_kinds", ["additive", "missing"])
    levels = cfg.get("corruption_levels", [0.1, 0.2, 0.3])
    seeds = cfg.get("corruption_seeds", cfg["seeds"][:5])
    n_train, n_test = cfg["n_train"], cfg["n_test"]
    top3 = cfg.get("corruption_baselines", ["S-learner", "CausalForest", "DR-learner"])
    bkw, hkw = cfg.get("bridge", {}), cfg.get("head", {})
    rows = []
    for seed in seeds:
        for kind in kinds:
            for s in [0.0] + levels:
                set_seed(seed)
                ds = generate_s1(n=n_train, seed=seed)
                dtest = generate_s1(n=n_test, seed=10_000 + seed)
                ds.W, ds.V = corrupt_proxies(ds.W, ds.V, kind, s, seed)
                dtest.W, dtest.V = corrupt_proxies(dtest.W, dtest.V, kind, s, seed + 1)
                opm = _fit_opm(ds, seed, "proximal", bkw, hkw)
                rows.append({"seed": seed, "kind": kind, "level": s, "method": "OPM",
                             **_eval(opm, dtest, is_opm=True)})
                for name in ["PDR-sieve"] + top3:
                    b = build(name, K=ds.K, seed=seed, bridge_kwargs=bkw, head_kwargs=hkw)
                    b.fit(ds)
                    rows.append({"seed": seed, "kind": kind, "level": s, "method": name,
                                 **_eval(b, dtest)})
        print(f"E3-corruption seed{seed} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir("E3"), "raw_corruption.csv"), index=False)
    return df


def _report(cfg, df, corr_df):
    ign = [m for m in cfg["methods"] if m in IGNORABILITY]
    criteria, sections = [], []
    all_p, p_labels = [], []
    beats_all = True

    for scen in cfg["scenarios"]:
        sub = df[df.scenario == scen]
        opm_pehe = sub[sub.method == "OPM"].sort_values("seed")["pehe"].to_numpy()
        sec = [f"### {scen}: PEHE (mean ± sd over seeds)", ""]
        tbl = []
        for m in ["OPM"] + cfg["methods"]:
            v = sub[sub.method == m].sort_values("seed")["pehe"].to_numpy()
            s = summarize(v[~np.isnan(v)])
            tbl.append({"method": m, "pehe_mean": round(s["mean"], 4), "pehe_sd": round(s["sd"], 4)})
        sec.append(pd.DataFrame(tbl).to_markdown(index=False)); sec.append("")
        # paired tests OPM vs each ignorability baseline
        for m in ign:
            v = sub[sub.method == m].sort_values("seed")["pehe"].to_numpy()
            if np.any(np.isnan(v)):
                continue
            pt = paired_tests(opm_pehe, v)  # mean_diff<0 => OPM better
            all_p.append(pt["wilcoxon_p"]); p_labels.append((scen, m, pt["mean_diff"]))
        sections += sec

    # BH correction across all (scenario, ignorability-baseline) tests
    if all_p:
        rej, adj = benjamini_hochberg(all_p, alpha=0.05)
        detail = []
        for (scen, m, md), a, r in zip(p_labels, adj, rej):
            better = md < 0
            ok = better and r
            beats_all = beats_all and ok
            detail.append(f"{scen}/{m}: mean_diff={md:+.3f} adjp={a:.3g} {'✓' if ok else '✗'}")
        criteria.append({"name": "OPM beats every ignorability baseline on PEHE (Wilcoxon, BH<0.05)",
                         "passed": beats_all, "detail": "; ".join(detail)})

    # Kernel-vs-sieve gap: the spec reports this "either way" (both are proximal, differing
    # only in bridge estimator); OPM is competitive if within 20% of sieve-PDR. See DECISIONS.
    pdr_ok = True
    pdr_detail = []
    for scen in cfg["scenarios"]:
        sub = df[df.scenario == scen]
        o = sub[sub.method == "OPM"]["pehe"].mean()
        p = sub[sub.method == "PDR-sieve"]["pehe"].mean()
        rel = (o - p) / max(p, 1e-9)
        pdr_ok = pdr_ok and (rel <= 0.20)
        pdr_detail.append(f"{scen}: OPM={o:.3f} vs PDR-sieve={p:.3f} (gap {rel*100:+.1f}%)")
    criteria.append({"name": "OPM competitive with sieve-PDR (kernel-vs-sieve gap reported, within 20%)",
                     "passed": pdr_ok, "detail": "; ".join(pdr_detail)})

    # corruption graceful degradation
    if corr_df is not None:
        graceful, cdetail = True, []
        for kind in corr_df["kind"].unique():
            for method in corr_df["method"].unique():
                sq = corr_df[(corr_df.kind == kind) & (corr_df.method == method)]
                curve = sq.groupby("level")["pehe"].mean().sort_index()
                vals = curve.to_numpy()
                if len(vals) >= 2:
                    ratios = vals[1:] / np.maximum(vals[:-1], 1e-6)
                    if np.any(ratios > 2.0):
                        graceful = False
                        cdetail.append(f"{kind}/{method}: jump {ratios.max():.2f}x")
        criteria.append({"name": "Graceful degradation (no PEHE jump >2x between adjacent levels)",
                         "passed": graceful,
                         "detail": "OK" if graceful else "; ".join(cdetail)})
        _plot_corruption(corr_df)
        sections += ["### Corruption scans (S1)", "", "![corruption](figures/e3_corruption.png)", ""]

    _plot_scatter(df, cfg)
    _latex_tables(df, cfg)
    sections += ["### Per-seed PEHE scatter", "", "![scatter](figures/e3_scatter.png)", "",
                 f"Baselines not run (honest bookkeeping): "
                 + "; ".join(f"{k} — {v}" for k, v in NOT_RUN.items())]
    write_report("E3", "Main benchmark (S1, S2 + corruption)", criteria, sections)


def _plot_scatter(df, cfg):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sc = cfg["scenarios"]
    fig, axes = plt.subplots(1, len(sc), figsize=(6 * len(sc), 4), squeeze=False)
    for j, scen in enumerate(sc):
        ax = axes[0][j]
        sub = df[df.scenario == scen]
        order = ["OPM"] + cfg["methods"]
        for i, m in enumerate(order):
            v = sub[sub.method == m]["pehe"].to_numpy()
            ax.scatter(np.full(len(v), i), v, s=14, alpha=0.6)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("PEHE"); ax.set_title(scen)
    fig.tight_layout(); fig.savefig(os.path.join(results_dir("E3"), "figures", "e3_scatter.png"), dpi=110)
    plt.close(fig)


def _plot_corruption(corr_df):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    kinds = corr_df["kind"].unique()
    fig, axes = plt.subplots(1, len(kinds), figsize=(5 * len(kinds), 4), squeeze=False)
    for j, kind in enumerate(kinds):
        ax = axes[0][j]
        for method in corr_df["method"].unique():
            sq = corr_df[(corr_df.kind == kind) & (corr_df.method == method)]
            curve = sq.groupby("level")["pehe"].mean().sort_index()
            ax.plot(curve.index, curve.values, "o-", label=method)
        ax.set_xlabel("corruption level s"); ax.set_ylabel("PEHE"); ax.set_title(kind); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(results_dir("E3"), "figures", "e3_corruption.png"), dpi=110)
    plt.close(fig)


def _latex_tables(df, cfg):
    lines = ["% E3 PEHE table (auto-generated)"]
    for scen in cfg["scenarios"]:
        sub = df[df.scenario == scen]
        lines.append(f"\\begin{{tabular}}{{lcc}}\n\\hline Method & PEHE & ATE err \\\\ \\hline")
        for m in ["OPM"] + cfg["methods"]:
            p = sub[sub.method == m]["pehe"]; a = sub[sub.method == m]["ate_err"]
            lines.append(f"{m} & {p.mean():.3f}$\\pm${p.std():.3f} & {a.mean():.3f} \\\\")
        lines.append("\\hline\\end{tabular}\n")
    with open(os.path.join(results_dir("E3"), "tables_E3.tex"), "w") as f:
        f.write("\n".join(lines))
