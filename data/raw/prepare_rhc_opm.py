#!/usr/bin/env python3
"""Prepare the public SUPPORT right-heart-catheterization data for OPM E4.

Input:
    data/raw/rhc.csv

Outputs:
    data/processed/rhc_opm_ready.csv

Definitions:
    A = 1 if right-heart catheterization was used within the first 24 hours.
    Main Y = t3d30, days from admission to death or censoring at day 30.
    Optional binary Y = 1 if alive at day 30.
    V = (pafi1, paco21), treatment-inducing proxies (called Z in Cui et al.).
    W = (ph1, hema1), outcome-inducing proxies.
    X = the other 67 variables from the reconstructed 71-covariate
        Hirano–Imbens-style baseline set.

Do not use dates, patient IDs, post-treatment variables, death indicators, or treatment
as X covariates.
"""

from pathlib import Path
import pandas as pd


def prepare_rhc(input_csv: str | Path, output_csv: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(input_csv)
    required = {
        "ptid", "swang1", "t3d30", "dth30", "pafi1", "paco21", "ph1", "hema1",
        "cat1", "cat2", "ca", "age", "sex", "edu", "income", "ninsclas",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Input RHC file is missing required columns: {sorted(missing)}")

    raw["cat2"] = raw["cat2"].fillna("Missing")
    cov = pd.DataFrame(index=raw.index)

    cov["age"] = raw["age"]
    cov["sex_female"] = (raw["sex"] == "Female").astype(int)
    cov["race_black"] = (raw["race"] == "black").astype(int)
    cov["race_other"] = (raw["race"] == "other").astype(int)
    cov["edu"] = raw["edu"]

    for name, label in [
        ("income_11_25k", "$11-$25k"),
        ("income_25_50k", "$25-$50k"),
        ("income_gt_50k", "> $50k"),
    ]:
        cov[name] = (raw["income"] == label).astype(int)

    for name, label in [
        ("ins_medicare", "Medicare"),
        ("ins_private_medicare", "Private & Medicare"),
        ("ins_medicaid", "Medicaid"),
        ("ins_none", "No insurance"),
        ("ins_medicare_medicaid", "Medicare & Medicaid"),
    ]:
        cov[name] = (raw["ninsclas"] == label).astype(int)

    for name, label in [
        ("cat1_copd", "COPD"),
        ("cat1_mosf_sepsis", "MOSF w/Sepsis"),
        ("cat1_mosf_malignancy", "MOSF w/Malignancy"),
        ("cat1_chf", "CHF"),
        ("cat1_coma", "Coma"),
        ("cat1_cirrhosis", "Cirrhosis"),
        ("cat1_lung_cancer", "Lung Cancer"),
        ("cat1_colon_cancer", "Colon Cancer"),
    ]:
        cov[name] = (raw["cat1"] == label).astype(int)

    for name, label in [
        ("cat2_mosf_sepsis", "MOSF w/Sepsis"),
        ("cat2_coma", "Coma"),
        ("cat2_mosf_malignancy", "MOSF w/Malignancy"),
        ("cat2_lung_cancer", "Lung Cancer"),
        ("cat2_cirrhosis", "Cirrhosis"),
        ("cat2_colon_cancer", "Colon Cancer"),
    ]:
        cov[name] = (raw["cat2"] == label).astype(int)

    for col in ["resp", "card", "neuro", "gastr", "renal", "meta", "hema", "seps", "trauma", "ortho"]:
        cov[col] = (raw[col] == "Yes").astype(int)

    cov["das2d3pc"] = raw["das2d3pc"]
    cov["dnr1"] = (raw["dnr1"] == "Yes").astype(int)
    cov["ca_yes"] = (raw["ca"] == "Yes").astype(int)
    cov["ca_metastatic"] = (raw["ca"] == "Metastatic").astype(int)

    for col in [
        "surv2md1", "aps1", "scoma1", "wtkilo1", "temp1", "meanbp1",
        "resp1", "hrt1", "pafi1", "paco21", "ph1", "wblc1", "hema1",
        "sod1", "pot1", "crea1", "bili1", "alb1",
    ]:
        cov[col] = raw[col]

    for col in [
        "cardiohx", "chfhx", "dementhx", "psychhx", "chrpulhx", "renalhx",
        "liverhx", "gibledhx", "malighx", "immunhx", "transhx", "amihx",
    ]:
        cov[col] = raw[col]

    assert cov.shape[1] == 71
    assert not cov.isna().any().any()

    proxy_cols = {"pafi1", "paco21", "ph1", "hema1"}
    x_cols = [c for c in cov.columns if c not in proxy_cols]
    assert len(x_cols) == 67

    out = pd.DataFrame({
        "patient_id": raw["ptid"],
        "A_rhc": (raw["swang1"] == "RHC").astype(int),
        "Y_survival_days_30": raw["t3d30"].astype(float),
        "Y_survived_30": (raw["dth30"] == "No").astype(int),
        "V_pafi1": raw["pafi1"].astype(float),
        "V_paco21": raw["paco21"].astype(float),
        "W_ph1": raw["ph1"].astype(float),
        "W_hema1": raw["hema1"].astype(float),
    })
    for col in x_cols:
        out[f"X_{col}"] = cov[col]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


if __name__ == "__main__":
    prepared = prepare_rhc(
        input_csv="data/raw/rhc.csv",
        output_csv="data/processed/rhc_opm_ready.csv",
    )
    print(
        f"Saved {len(prepared)} rows and {prepared.shape[1]} columns; "
        f"treated={prepared['A_rhc'].sum()}, "
        f"30-day survivors={prepared['Y_survived_30'].sum()}."
    )
