"""RHC (right heart catheterization) loader — spec 3.4.

Prefers the prepared ``data/raw/rhc_opm_ready.csv`` (treatment, outcomes, four proxies,
67 baseline X covariates), falling back to a raw ``rhc.csv`` with the Cui-et-al. split.

Proxy split (Cui et al. 2023, JASA, Section 6):
  V = (pafi1, paco21)  treatment-inducing proxies
  W = (ph1, hema1)     outcome-inducing proxies
  # VERIFY against the published paper; if your copy differs, edit PROXY_V / PROXY_W.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..dgp.base import CausalDataset

PROXY_V = ["pafi1", "paco21"]
PROXY_W = ["ph1", "hema1"]
READY = os.path.join("data", "raw", "rhc_opm_ready.csv")
RAW = os.path.join("data", "raw", "rhc.csv")


def load_rhc(csv_path: str = None, outcome: str = "Y_survival_days_30") -> CausalDataset:
    """Load RHC as a binary-treatment proximal dataset (K=2).

    outcome: 'Y_survival_days_30' (days to death/censor at 30; the JASA endpoint, use for
    the linear-POR gate) or 'Y_survived_30' (binary alive-at-30).
    """
    path = csv_path or (READY if os.path.exists(READY) else RAW)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"RHC dataset not found at {path}. Place rhc_opm_ready.csv (or rhc.csv) in data/raw/.")
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
                               "proxy_V": PROXY_V, "proxy_W": PROXY_W, "n": int(keep.sum())})
