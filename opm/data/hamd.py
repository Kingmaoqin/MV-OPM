"""HAMD depression-trial loader — local real-covariate anchor (substitute for RHC).

Source: /home/xqin5/DpressionTreatmenteffect/data-merged-def-1.csv (~1.9k patients).
Discrete treatment DRUG (K=4: Duloxetine/Venlafaxine/Paroxetine/Fluoxetine), pre-treatment
covariates = baseline HAMD01..17 items + demographics + severity, outcome = follow-up HAMD
total (lower is better).  No designated proxies -> exercises the dr_fallback path (E4) and
provides real covariates for the semi-synthetic DGP (E5).  See DECISIONS.md.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..dgp.base import CausalDataset

DEFAULT_CSV = "/home/xqin5/DpressionTreatmenteffect/data-merged-def-1.csv"
DRUG_ORDER = ["Duloxetine", "Venlafaxine", "Paroxetine", "Fluoxetine"]  # 0 = reference
BASE_ITEMS = [f"V1-HAMD{i:02d}" for i in range(1, 18)]


def _covariate_frame(df: pd.DataFrame) -> pd.DataFrame:
    num = df[["AGE"] + BASE_ITEMS].apply(pd.to_numeric, errors="coerce")
    cats = []
    for c in ["SEX", "SEVERITY", "PROTOCOL"]:
        if c in df.columns:
            cats.append(pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True))
    X = pd.concat([num] + cats, axis=1)
    return X.apply(pd.to_numeric, errors="coerce")


def load_hamd(csv_path: str = DEFAULT_CSV, outcome: str = "LHAMD") -> CausalDataset:
    """outcome: 'LHAMD' (last HAMD total) or 'V2'/'V3' (that visit's HAMD total)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"HAMD csv not found at {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["DRUG"].isin(DRUG_ORDER)].copy()

    if outcome == "LHAMD" and "LHAMD" in df.columns:
        y = pd.to_numeric(df["LHAMD"], errors="coerce")
    else:
        vis = outcome if outcome.startswith("V") else "V2"
        cols = [f"{vis}-HAMD{i:02d}" for i in range(1, 18)]
        y = df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=17)

    X = _covariate_frame(df)
    T = df["DRUG"].map({d: i for i, d in enumerate(DRUG_ORDER)}).to_numpy()

    keep = y.notna() & X.notna().all(axis=1)
    X = X[keep].to_numpy(dtype=float)
    y = y[keep].to_numpy(dtype=float)
    T = T[keep.to_numpy()].astype(np.int64)
    return CausalDataset(X=X, T=T, Y=y, K=len(DRUG_ORDER), W=None, V=None,
                         meta={"dataset": "hamd", "outcome": outcome,
                               "drug_order": DRUG_ORDER, "n": len(y)})


def hamd_covariates(csv_path: str = DEFAULT_CSV, standardize: bool = True) -> np.ndarray:
    """Real covariate matrix for the semi-synthetic DGP (E5)."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["DRUG"].isin(DRUG_ORDER)].copy()
    X = _covariate_frame(df)
    X = X[X.notna().all(axis=1)].to_numpy(dtype=float)
    if standardize:
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X
