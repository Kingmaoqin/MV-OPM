"""E5 — Semi-synthetic on real (HAMD) covariates (spec Section 5, E5).

Realistic covariates + known truth: real HAMD covariates enter theta_k'X and g0 via
meta-seeded projections; U/T/Y/W/V are simulated as in S1 (proxies present -> proximal
mode is identified).  Non-oracle OPM vs baselines; same statistical criteria as E3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..baselines.registry import IGNORABILITY, NOT_RUN, build
from ..data.hamd import hamd_covariates
from ..dgp.semisynth import generate_semisynth
from ..estimator.learner import OPM, OPMConfig
from ..eval.stats import benjamini_hochberg, paired_tests, summarize
from ..utils import set_seed
from .common import results_dir, save_raw, write_report
from .e3 import _eval, _fit_opm


def run(cfg: dict) -> dict:
    seeds = cfg["seeds"]
    methods = cfg["methods"]
    bkw, hkw = cfg.get("bridge", {}), cfg.get("head", {})
    c_U = cfg.get("c_U", 1.0)
    Xreal = hamd_covariates()
    n = Xreal.shape[0]

    rows = []
    for seed in seeds:
        set_seed(seed)
        # Separate full-size train/test draws on the SAME real covariates (rather than a
        # 50/50 split) so the kernel bridge gets the full n on these high-dim (p=67) covariates.
        ds = generate_semisynth(Xreal, seed=seed, c_U=c_U)
        dtest = generate_semisynth(Xreal, seed=5000 + seed, c_U=c_U)
        opm = _fit_opm(ds, seed, "proximal", bkw, hkw)
        rows.append({"seed": seed, "method": "OPM", **_eval(opm, dtest, is_opm=True)})
        print(f"E5 seed{seed} OPM pehe={rows[-1]['pehe']:.3f}", flush=True)
        for name in methods:
            try:
                b = build(name, K=ds.K, seed=seed, bridge_kwargs=bkw, head_kwargs=hkw); b.fit(ds)
                rows.append({"seed": seed, "method": name, **_eval(b, dtest)})
            except Exception as e:
                rows.append({"seed": seed, "method": name, "pehe": np.nan,
                             "ate_err": np.nan, "policy": np.nan, "error": str(e)})
    df = pd.DataFrame(rows)
    save_raw("E5", rows)
    _report(cfg, df)
    return {"df": df}


def _report(cfg, df):
    ign = [m for m in cfg["methods"] if m in IGNORABILITY]
    opm_pehe = df[df.method == "OPM"].sort_values("seed")["pehe"].to_numpy()
    tbl, all_p, labels = [], [], []
    for m in ["OPM"] + cfg["methods"]:
        v = df[df.method == m].sort_values("seed")["pehe"].to_numpy()
        s = summarize(v[~np.isnan(v)])
        tbl.append({"method": m, "pehe_mean": round(s["mean"], 4), "pehe_sd": round(s["sd"], 4)})
    for m in ign:
        v = df[df.method == m].sort_values("seed")["pehe"].to_numpy()
        if np.any(np.isnan(v)):
            continue
        pt = paired_tests(opm_pehe, v)
        all_p.append(pt["wilcoxon_p"]); labels.append((m, pt["mean_diff"]))
    beats = True; detail = []
    if all_p:
        rej, adj = benjamini_hochberg(all_p)
        for (m, md), a, r in zip(labels, adj, rej):
            ok = (md < 0) and r; beats = beats and ok
            detail.append(f"{m}: mean_diff={md:+.3f} adjp={a:.3g} {'✓' if ok else '✗'}")
    o = df[df.method == "OPM"]["pehe"].mean(); p = df[df.method == "PDR-sieve"]["pehe"].mean()
    rel = (o - p) / max(p, 1e-9)
    criteria = [
        {"name": "OPM beats every ignorability baseline on PEHE (Wilcoxon, BH<0.05)",
         "passed": beats, "detail": "; ".join(detail)},
        {"name": "OPM competitive with sieve-PDR (kernel-vs-sieve gap reported, within 20%)",
         "passed": rel <= 0.20, "detail": f"OPM={o:.3f} vs PDR-sieve={p:.3f} (gap {rel*100:+.1f}%)"},
    ]
    body = ["## PEHE on semi-synthetic HAMD covariates (mean ± sd over seeds)", "",
            pd.DataFrame(tbl).to_markdown(index=False), "",
            "Baselines not run: " + "; ".join(f"{k}" for k in NOT_RUN)]
    write_report("E5", "Semi-synthetic on real HAMD covariates", criteria, body)
