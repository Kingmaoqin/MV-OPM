"""E4 — Real-data anchor (spec 3.4).

Primary anchor: RHC (right heart catheterization, SUPPORT ICU study) with the Cui-et-al.
proximal split — treatment-inducing proxies V=(pafi1,paco21), outcome-inducing proxies
W=(ph1,hema1), 67 baseline covariates X, outcome = days-to-death/censoring at 30.

Milestone gate: a LINEAR-sieve POR estimator (2SLS on degree-1 proxy bases) must produce a
plausible small NEGATIVE effect on survival days (RHC increases mortality — the classic
Connors 1996 / Cui 2023 finding). Only then do we run the full proximal OPM and dr_fallback.

ACCEPT: linear-POR gate passes (finite, plausible negative ATE); OPM proximal CI overlaps
the linear-POR CI; weights bounded (max q_hat <= 50, En[1{T=k} q_hat_k] in [0.9,1.1]).

A secondary HAMD anchor (K=4, no proxies -> dr_fallback) is available via dataset=hamd.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..baselines.base import CrossFitProximal
from ..bridges.sieve import SieveBridges
from ..data.hamd import DRUG_ORDER, load_hamd
from ..data.rhc import load_rhc
from ..diagnostics.diagnostics import q_sanity
from ..estimator.learner import OPM, OPMConfig
from ..utils import set_seed
from .common import save_raw, write_report


def _overlap(a, b):
    return not (a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"])


def _run_rhc(cfg):
    seeds = cfg["seeds"]
    hkw = cfg.get("head", {})
    bkw = cfg.get("bridge", {})
    ds = load_rhc(outcome=cfg.get("outcome", "Y_survival_days_30"))

    def _sieve1(seed):
        return SieveBridges(K=ds.K, degree=1, q_max=50.0)

    lin, opm_p, opm_f = [], [], []
    qdiag = []
    for seed in seeds:
        set_seed(seed)
        # (1) linear-POR gate (degree-1 sieve, POR pseudo-outcome)
        por = CrossFitProximal("linear-POR", _sieve1, "por", ds.K, seed=seed,
                               head_kwargs=hkw).fit(ds)
        lin.append(por.ate[1])
        # (2) full proximal OPM (kernel bridges)
        op = OPM(OPMConfig(K=ds.K, mode="proximal", seed=seed, bridge_kwargs=bkw,
                           head_kwargs=hkw)).fit(ds)
        opm_p.append(op.ate[1]); qdiag.append(q_sanity(op.q_oof, ds.T, ds.K))
        # (3) dr_fallback (no-proxy comparator)
        of = OPM(OPMConfig(K=ds.K, mode="dr_fallback", seed=seed, head_kwargs=hkw)).fit(ds)
        opm_f.append(of.ate[1])
        print(f"E4-RHC seed{seed}: POR={por.ate[1]['ate']:.3f} OPM={op.ate[1]['ate']:.3f} "
              f"fb={of.ate[1]['ate']:.3f}", flush=True)

    def _avg(lst):
        return {"ate": float(np.mean([x["ate"] for x in lst])),
                "ci_low": float(np.mean([x["ci_low"] for x in lst])),
                "ci_high": float(np.mean([x["ci_high"] for x in lst]))}
    L, P, F = _avg(lin), _avg(opm_p), _avg(opm_f)
    en1tq = {k: float(np.mean([d[k]["En_1Tk_qk"] for d in qdiag])) for k in range(ds.K)}
    maxq = float(np.mean([max(d[k]["max_q"] for k in d) for d in qdiag]))
    ess = {k: float(np.mean([d[k]["ess"] for d in qdiag])) for k in range(ds.K)}

    gate_ok = np.isfinite(L["ate"]) and (L["ate"] < 0)          # plausible negative effect
    overlap_ok = _overlap(P, L)
    weights_ok = (maxq <= 50 + 1e-6) and all(0.9 <= en1tq[k] <= 1.1 for k in range(ds.K))

    rows = [{"estimator": "linear-POR (gate)", **L},
            {"estimator": "OPM proximal", **P},
            {"estimator": "OPM dr_fallback", **F}]
    save_raw("E4", rows)
    criteria = [
        {"name": "Linear-POR gate: finite, plausible NEGATIVE effect on survival days",
         "passed": bool(gate_ok),
         "detail": f"POR ATE={L['ate']:.3f} days [{L['ci_low']:.3f},{L['ci_high']:.3f}] "
                   "(RHC reduces survival — Connors 1996 / Cui 2023)"},
        {"name": "OPM proximal CI overlaps linear-POR CI",
         "passed": bool(overlap_ok),
         "detail": f"OPM ATE={P['ate']:.3f} [{P['ci_low']:.3f},{P['ci_high']:.3f}] vs "
                   f"POR [{L['ci_low']:.3f},{L['ci_high']:.3f}]"},
        {"name": "Weights bounded (max q_hat<=50, En[1{T=k}q_k] in [0.9,1.1])",
         "passed": bool(weights_ok),
         "detail": f"max_q={maxq:.2f}; En[1Tq]=" + ", ".join(f"{k}:{en1tq[k]:.3f}" for k in range(ds.K))},
    ]
    body = [f"Dataset: RHC, n={ds.n}, K=2 (RHC vs no-RHC), outcome={cfg.get('outcome','Y_survival_days_30')} "
            "(days to death/censor at 30; negative ATE = RHC shortens survival).", "",
            "## ATE estimates (mean over cross-fit seeds, days)", "",
            pd.DataFrame(rows).to_markdown(index=False), "",
            f"OPM proximal ESS per arm: " + ", ".join(f"{k}:{ess[k]:.0f}" for k in range(ds.K)), "",
            "Proxies present (V=pafi1,paco21; W=ph1,hema1) so the proximal q-bridge's moment "
            "identity B2 keeps En[1{T=k}q_k]≈1 by construction — the strict weight band holds "
            "here (unlike the no-proxy HAMD secondary anchor)."]
    write_report("E4", "Real-data anchor (RHC proximal)", criteria, body)


def _run_hamd(cfg):
    # secondary anchor (no proxies -> dr_fallback); see git history / DECISIONS.md
    from .e4_hamd import run_hamd
    run_hamd(cfg)


def run(cfg: dict) -> dict:
    dataset = cfg.get("dataset", "rhc")
    if dataset == "rhc":
        _run_rhc(cfg)
    else:
        _run_hamd(cfg)
    return {}
