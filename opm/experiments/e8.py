"""E8 — Diagnostics validity study (spec Section 5, E8).

Across corruption levels/seeds, correlate the Section 6.1 diagnostics (proxy strength
rho_min, bridge residual) with realized PEHE (Spearman, per scenario family).

ACCEPT: rho_min and bridge residual each achieve |Spearman| > 0.4 with PEHE across
corruption levels in at least one scenario family; report all correlations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..diagnostics.diagnostics import bridge_residual_score, proxy_strength
from ..dgp.synthetic_main import corrupt_proxies, generate_s1
from ..estimator.learner import OPM, OPMConfig
from ..eval.metrics import pehe
from ..utils import median_bandwidth, set_seed, standardize_apply, standardize_fit
from .common import results_dir, save_raw, write_report


def _bridge_resid(opm, ds):
    """Mean over arms of the RFF bridge-residual score on OOF h (instrument (V,X))."""
    inst = np.concatenate([ds.V, ds.X], axis=1)
    bw = median_bandwidth(standardize_apply(inst, *standardize_fit(inst)))
    scores = []
    for k in range(ds.K):
        R = (ds.T == k).astype(float) * (ds.Y - opm.h_oof[:, k])
        scores.append(bridge_residual_score(R, inst, bw, seed=k))
    return float(np.mean(scores))


def run(cfg: dict) -> dict:
    seeds = cfg["seeds"]
    bkw, hkw = cfg.get("bridge", {}), cfg.get("head", {})
    rows = []
    for kind in cfg["corruption_kinds"]:
        for s in cfg["corruption_levels"]:
            for seed in seeds:
                set_seed(seed)
                ds = generate_s1(n=cfg["n_train"], seed=seed)
                dtest = generate_s1(n=cfg["n_test"], seed=50_000 + seed)
                ds.W, ds.V = corrupt_proxies(ds.W, ds.V, kind, s, seed)
                dtest.W, dtest.V = corrupt_proxies(dtest.W, dtest.V, kind, s, seed + 1)
                opm = OPM(OPMConfig(K=ds.K, mode="proximal", seed=seed,
                                    bridge_kwargs=bkw, head_kwargs=hkw)).fit(ds)
                ps = proxy_strength(ds.W, ds.V, ds.X)
                rows.append({"kind": kind, "level": s, "seed": seed,
                             "pehe": pehe(opm.predict_cate(dtest.X), dtest.tau_true),
                             "rho_min": ps["rho_min"], "rho_top": ps["rho_top"],
                             "bridge_resid": _bridge_resid(opm, ds)})
            print(f"E8 {kind} level={s} done", flush=True)
    df = pd.DataFrame(rows)
    save_raw("E8", rows)
    _report(df)
    return {"df": df}


def _report(df):
    # The spec correlates the diagnostics with PEHE "across corruption levels". Per-(level,
    # seed) points are dominated by seed noise (PEHE has large seed variance while the method
    # degrades gracefully), so we correlate the per-LEVEL means over seeds — the quantity that
    # actually tracks proxy strength as corruption varies. Per-point correlations are reported
    # too for transparency. See DECISIONS.md.
    corr_rows, passed_any_rho, passed_any_resid = [], False, False
    for kind in df["kind"].unique():
        sub = df[df.kind == kind]
        g = sub.groupby("level").agg(rho_min=("rho_min", "mean"), bridge_resid=("bridge_resid", "mean"),
                                     pehe=("pehe", "mean")).reset_index()
        sp_rho = spearmanr(g["rho_min"], g["pehe"]).correlation
        sp_res = spearmanr(g["bridge_resid"], g["pehe"]).correlation
        sp_rho_pt = spearmanr(sub["rho_min"], sub["pehe"]).correlation
        sp_res_pt = spearmanr(sub["bridge_resid"], sub["pehe"]).correlation
        passed_any_rho = passed_any_rho or abs(sp_rho) > 0.4
        passed_any_resid = passed_any_resid or abs(sp_res) > 0.4
        corr_rows.append({"family": kind, "spearman(rho_min,PEHE)": round(sp_rho, 3),
                          "spearman(bridge_resid,PEHE)": round(sp_res, 3),
                          "rho_min(per-point)": round(sp_rho_pt, 3),
                          "bridge_resid(per-point)": round(sp_res_pt, 3)})

    criteria = [
        {"name": "|Spearman(rho_min, PEHE)| > 0.4 across corruption levels in >=1 family",
         "passed": passed_any_rho,
         "detail": "; ".join(f"{r['family']}:{r['spearman(rho_min,PEHE)']}" for r in corr_rows)},
        {"name": "|Spearman(bridge_resid, PEHE)| > 0.4 across corruption levels in >=1 family",
         "passed": passed_any_resid,
         "detail": "; ".join(f"{r['family']}:{r['spearman(bridge_resid,PEHE)']}" for r in corr_rows)},
    ]
    body = ["## Diagnostic–PEHE Spearman correlations (per-level means over seeds)", "",
            pd.DataFrame(corr_rows).to_markdown(index=False), "",
            "Expected signs: rho_min ↓ (weaker proxies) and bridge_resid ↑ as corruption "
            "grows, both tracking higher PEHE. Per-point columns include seed scatter."]
    write_report("E8", "Diagnostics validity study", criteria, body)
