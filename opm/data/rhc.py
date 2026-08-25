"""RHC (right heart catheterization) loader — spec 3.4.

Prefers the prepared ``data/raw/rhc_opm_ready.csv`` (treatment, outcomes, four proxies,
67 baseline X covariates), falling back to a raw ``rhc.csv`` with the Cui-et-al. split.

Provisional proxy split attributed to Cui et al. 2023, JASA, Section 6:
  V = (pafi1, paco21)  treatment-inducing proxies
  W = (ph1, hema1)     outcome-inducing proxies

Confirmatory causal use requires a verified ``<csv>.provenance.json`` sidecar containing the
dataset checksum, exact paper citation/page, and this proxy mapping. Without it the loader can
only be used as a provisional engineering demonstration.
"""
from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd

from ..dgp.base import CausalDataset
from ..provenance import sha256_file

PROXY_V = ["pafi1", "paco21"]
PROXY_W = ["ph1", "hema1"]
# Populated only after an external source review commits the exact sidecar and its digest.
# A sidecar cannot authorize itself by setting a boolean field.
APPROVED_RHC_SIDECAR_SHA256 = frozenset()
READY = os.path.join("data", "raw", "rhc_opm_ready.csv")
RAW = os.path.join("data", "raw", "rhc.csv")


def _proxy_provenance(path: str) -> dict:
    sidecar = path + ".provenance.json"
    if not os.path.exists(sidecar):
        return {"verified": False, "reason": "missing_provenance_sidecar", "sidecar": sidecar}
    with open(sidecar, encoding="utf-8") as handle:
        record = json.load(handle)
    expected_mapping = {"V": PROXY_V, "W": PROXY_W}
    sidecar_sha256 = sha256_file(sidecar)
    checks = {
        "dataset_sha256": record.get("dataset_sha256") == sha256_file(path),
        "proxy_mapping": record.get("proxy_mapping") == expected_mapping,
        "citation": bool(
            record.get("paper_citation")
            and record.get("paper_doi")
            and record.get("paper_page_or_table")
            and record.get("paper_variable_definition")
        ),
        "preparation": bool(record.get("preparation_script_sha256")),
        "external_review": bool(
            record.get("reviewed_by") and record.get("reviewed_on")
        ),
    }
    schema_complete = all(checks.values())
    externally_approved = sidecar_sha256 in APPROVED_RHC_SIDECAR_SHA256
    return {
        "verified": schema_complete and externally_approved,
        "schema_complete": schema_complete,
        "externally_approved": externally_approved,
        "sidecar_sha256": sidecar_sha256,
        "checks": checks,
        "sidecar": sidecar,
        "record": record,
    }


def load_rhc(
    csv_path: str = None,
    outcome: str = "Y_survival_days_30",
    *,
    require_verified_proxy_mapping: bool = False,
) -> CausalDataset:
    """Load RHC as a binary-treatment proximal dataset (K=2).

    outcome: 'Y_survival_days_30' (days to death/censor at 30; the JASA endpoint, use for
    the linear-POR gate) or 'Y_survived_30' (binary alive-at-30).
    """
    path = csv_path or (READY if os.path.exists(READY) else RAW)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"RHC dataset not found at {path}. Place rhc_opm_ready.csv (or rhc.csv) in data/raw/.")
    provenance = _proxy_provenance(path)
    if require_verified_proxy_mapping and not provenance["verified"]:
        raise RuntimeError(
            "RHC proxy mapping is not verified for causal use: "
            f"{provenance}. Add an externally reviewed sidecar and register its exact digest."
        )
    df = pd.read_csv(path)

    if "A_rhc" in df.columns:                     # prepared file
        T = df["A_rhc"].astype(int).to_numpy()
        Y = df[outcome].astype(float).to_numpy()
        V = df[["V_pafi1", "V_paco21"]].apply(pd.to_numeric, errors="coerce").to_numpy()
        W = df[["W_ph1", "W_hema1"]].apply(pd.to_numeric, errors="coerce").to_numpy()
        Xcols = [c for c in df.columns if c.startswith("X_")]
        X = df[Xcols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        meta_out = outcome
    else:                                          # raw fallback (Cui split)
        T = (df["swang1"].astype(str).str.contains("RHC|Yes|1", case=False)).astype(int).to_numpy()
        Y = df["t3d30"].astype(float).to_numpy() if "t3d30" in df.columns else df["dth30"].astype(float).to_numpy()
        V = df[PROXY_V].apply(pd.to_numeric, errors="coerce").to_numpy()
        W = df[PROXY_W].apply(pd.to_numeric, errors="coerce").to_numpy()
        drop = set(PROXY_V + PROXY_W + ["swang1", "t3d30", "dth30"])
        Xc = [c for c in df.columns if c not in drop]
        X = pd.get_dummies(df[Xc], drop_first=True).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
        meta_out = "t3d30"

    keep = np.isfinite(Y) & np.isfinite(V).all(1) & np.isfinite(W).all(1)
    return CausalDataset(X=X[keep], T=T[keep], Y=Y[keep], K=2, W=W[keep], V=V[keep],
                         meta={"dataset": "rhc", "outcome": meta_out,
                               "proxy_V": PROXY_V, "proxy_W": PROXY_W, "n": int(keep.sum()),
                               "proxy_mapping_verified": provenance["verified"],
                               "causal_inference_authorized": provenance["verified"],
                               "proxy_provenance_sidecar": provenance["sidecar"]})
