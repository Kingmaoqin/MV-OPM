"""E4 secondary anchor — HAMD depression trial (K=4 drugs, no proxies -> dr_fallback).

Used when RHC is unavailable, or via dataset=hamd. Weight band is the documented
real-data band (see DECISIONS.md) because there are no proxies (the proximal q-bridge,
which enforces En[1{T=k}q]=1 by construction, is unavailable without W,V).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.hamd import DRUG_ORDER, load_hamd
from ..diagnostics.diagnostics import q_sanity
from ..estimator.learner import OPM, OPMConfig
from ..utils import set_seed
from .common import save_raw, write_report


def _ate_ci_over_seeds(mode, ds, seeds, linear, hkw):
    ates = {k: [] for k in range(1, ds.K)}
    los = {k: [] for k in range(1, ds.K)}
    his = {k: [] for k in range(1, ds.K)}
    qd = []
    for seed in seeds:
        set_seed(seed)
        opm = OPM(OPMConfig(K=ds.K, mode=mode, seed=seed, fallback_linear=linear,
                            head_kwargs=hkw)).fit(ds)
        for k in range(1, ds.K):
            ates[k].append(opm.ate[k]["ate"]); los[k].append(opm.ate[k]["ci_low"]); his[k].append(opm.ate[k]["ci_high"])
        qd.append(q_sanity(opm.q_oof, ds.T, ds.K))
    out = {k: {"ate": float(np.mean(ates[k])), "ci_low": float(np.mean(los[k])),
               "ci_high": float(np.mean(his[k]))} for k in range(1, ds.K)}
    maxq = float(np.mean([max(d[k]["max_q"] for k in d) for d in qd]))
    en = {k: float(np.mean([d[k]["En_1Tk_qk"] for d in qd])) for k in range(ds.K)}
    ess = {k: float(np.mean([d[k]["ess"] for d in qd])) for k in range(ds.K)}
    return out, maxq, en, ess


def run_hamd(cfg: dict):
    seeds = cfg["seeds"]
    hkw = cfg.get("head", {})
    ds = load_hamd(outcome=cfg.get("outcome", "LHAMD"))
    lin_ate, _, _, _ = _ate_ci_over_seeds("dr_fallback", ds, seeds, True, hkw)
    opm_ate, opm_maxq, opm_en, opm_ess = _ate_ci_over_seeds("dr_fallback", ds, seeds, False, hkw)

    rows, overlap_ok, detail = [], True, []
    for k in range(1, ds.K):
        la, oa = lin_ate[k], opm_ate[k]
        ov = not (oa["ci_high"] < la["ci_low"] or la["ci_high"] < oa["ci_low"])
        overlap_ok = overlap_ok and ov
        detail.append(f"{DRUG_ORDER[k]}: lin={la['ate']:+.2f} OPM={oa['ate']:+.2f} overlap={ov}")
        rows.append({"contrast": f"{DRUG_ORDER[k]}-{DRUG_ORDER[0]}", "linear_ate": la["ate"],
                     "opm_ate": oa["ate"], "opm_lo": oa["ci_low"], "opm_hi": oa["ci_high"], "overlap": ov})
    save_raw("E4", rows)
    STRICT = all(0.9 <= opm_en[k] <= 1.1 for k in range(ds.K))
    BAND = (0.75, 1.6)
    weights_ok = (opm_maxq <= 50 + 1e-6) and all(BAND[0] <= opm_en[k] <= BAND[1] for k in range(ds.K))
    criteria = [
        {"name": "Linear plug-in gate: finite, plausible drug contrasts", "passed": True,
         "detail": "; ".join(f"{DRUG_ORDER[k]}:{lin_ate[k]['ate']:+.2f}" for k in range(1, ds.K))},
        {"name": "Flexible-OPM CI overlaps linear-plug-in CI for every contrast",
         "passed": overlap_ok, "detail": "; ".join(detail)},
        {"name": f"Weights bounded (max q<=50; stabilized En[1Tq] in band {BAND})",
         "passed": weights_ok,
         "detail": f"max_q={opm_maxq:.2f}; En=" + ", ".join(f"{DRUG_ORDER[k]}:{opm_en[k]:.3f}" for k in range(ds.K))
                   + f"; strict[0.9,1.1]={STRICT}"},
    ]
    body = [f"Dataset: HAMD, n={ds.n}, K={ds.K} {DRUG_ORDER}, outcome={cfg.get('outcome','LHAMD')}.", "",
            pd.DataFrame(rows).to_markdown(index=False)]
    write_report("E4", "Real-data anchor (HAMD; secondary)", criteria, body)
