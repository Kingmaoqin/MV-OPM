"""MV-OPM study rows: one call = one (study, config, seed) unit of work.

- Study A (mechanism): corrupt the EXACT population bridges by known rates and test whether the
  downstream ATE bias and the observed moment product track the true bridge-error product.
- Selection studies (B–J): fit the candidate library, score held-out moments, and evaluate every
  selector/competitor against the oracle-best candidate.
"""
from __future__ import annotations

import numpy as np

from ...data.views import observed
from ...dgp import finite_proxy_exact as fpe
from ...estimator.pseudo_outcome import compute_phi
from ...validation.moments import build_bank, discrepancy_from_hq
from .core import SelCfg, evaluate_candidates_oof, selection_metrics
from .scenarios import apply_proxy_mod, generate

ATE_TRUE = 2.0


# ---------------- Study A: exact product mechanism ----------------
def _delta_h(X):  # structured h-perturbation (function of X, well-defined on the exact DGP)
    return np.sin(X[:, 0])


def _delta_q(X):  # structured q-perturbation
    return np.cos(X[:, 1])


def mechanism_row(seed: int, r_h: float, r_q: float, n: int = 4000, c_U: float = 1.0,
                  n_rff: int = 200) -> dict:
    """Corrupt exact bridges by (r_h, r_q); return true errors, moment discrepancies, ATE bias."""
    ds = fpe.generate(n, seed, c_U=c_U)
    view = observed(ds)
    h_true = ds.meta["h_true_val"].copy()          # (n,2) oracle (mechanism study is oracle-side)
    q_true = ds.meta["q_true_val"].copy()
    dh = r_h * _delta_h(ds.X); dq = r_q * _delta_q(ds.X)
    h_hat = h_true + dh[:, None]                    # additive corruption both arms
    q_hat = np.clip(q_true * (1.0 + dq[:, None]), 0.0, 50.0)

    true_h_err = float(np.sqrt(np.mean((h_hat - h_true) ** 2)))
    true_q_err = float(np.sqrt(np.mean((q_hat - q_true) ** 2)))
    # moment discrepancies of the corrupted bridges (shared bank on the same data)
    bank = build_bank(np.concatenate([ds.V, ds.X], 1), np.concatenate([ds.W, ds.X], 1), n_rff, 20240)
    D = discrepancy_from_hq(bank, h_hat, q_hat, view, kinds=("rff_l2",))
    phi = compute_phi(h_hat, q_hat, view)
    ate_bias = float((phi[:, 1] - phi[:, 0]).mean() - ATE_TRUE)
    po_bias = float(phi.mean() - (h_true.mean()))    # rough pseudo-outcome level bias proxy
    return {"study": "A", "seed": seed, "n": n, "c_U": c_U, "r_h": r_h, "r_q": r_q,
            "true_h_err": true_h_err, "true_q_err": true_q_err,
            "true_err_product": true_h_err * true_q_err,
            "D_h": D["D_h_rff_l2"], "D_q": D["D_q_rff_l2"],
            "moment_product": D["D_h_rff_l2"] * D["D_q_rff_l2"],
            "ate_bias": ate_bias, "abs_ate_bias": abs(ate_bias), "po_bias": po_bias}


# ---------------- Selection studies (B–J) ----------------
def selection_row(study: str, scenario: str, n: int, seed: int, cfg: SelCfg,
                  c_U: float = 1.0, mod: dict = None) -> dict:
    """One selection unit: generate (with optional confounding/proxy mod), evaluate candidates,
    apply all selectors. Returns metrics + provenance. Also emits an abstention signal."""
    ds = generate(scenario, n, seed, c_U=c_U)
    if mod:
        ds = apply_proxy_mod(ds, mod, seed)
    rows = evaluate_candidates_oof(ds, cfg)
    m = selection_metrics(rows, cfg, target="pehe")
    m_ate = selection_metrics(rows, cfg, target="ate_err")
    # abstention signal: smallest total moment discrepancy across candidates (high => unsupported)
    kind = cfg.kind
    min_total_D = float(min(rows[nm][f"D_h_{kind}"] + rows[nm][f"D_q_{kind}"] for nm in rows))
    out = {"study": study, "scenario": scenario, "n": n, "seed": seed, "c_U": c_U,
           "mod_kind": (mod or {}).get("kind"), "mod_level": (mod or {}).get("level"),
           "min_total_D": min_total_D}
    out.update({f"pehe::{k}": v for k, v in m.items()})
    out.update({f"ate::{k}": v for k, v in m_ate.items()})
    # keep per-candidate raw for provenance/reconstruction
    out["_candidates"] = {nm: {kk: rows[nm][kk] for kk in rows[nm]} for nm in rows}
    return out


# ---------------- Study J: RHC real data (no oracle) ----------------
def rhc_row(seed: int, cfg: SelCfg) -> dict:
    """Repeated-split stability on RHC: select via product moments (no oracle), report the
    selected candidate + its ATE/CI + q diagnostics + moment scores. Splits are NOT independent
    replications (reported as split-to-split algorithmic stability)."""
    from ...data.rhc import load_rhc
    from ...validation.selector import compute_scores, select
    ds = load_rhc()
    # high-dim covariates (p=67): high-degree polynomial sieves are inapplicable (feature blow-up),
    # so the candidate library for real high-dim X excludes sieve2/sieve3.
    cfg.candidate_names = ["kernel_kernel", "sieve1_sieve1", "kernel_sieve1", "sieve1_kernel"]
    rows = evaluate_candidates_oof(ds, cfg)
    scores = compute_scores(rows, cfg.kind)["product"]
    sel = select(scores)
    return {"study": "J", "scenario": "rhc", "seed": seed, "selected": sel,
            "ate1": rows[sel]["ate1"], "ate1_lo": rows[sel]["ate1_lo"], "ate1_hi": rows[sel]["ate1_hi"],
            "ess": rows[sel]["ess"], "max_q": rows[sel]["max_q"], "q_balance": rows[sel]["q_balance"],
            "min_total_D": float(min(rows[nm][f"D_h_{cfg.kind}"] + rows[nm][f"D_q_{cfg.kind}"] for nm in rows)),
            "_candidates": {nm: {kk: rows[nm][kk] for kk in rows[nm]} for nm in rows}}
