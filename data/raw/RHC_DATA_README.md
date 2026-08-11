# RHC data bundle for OPM E4

## What “RHC” means

RHC here means **right heart catheterization**, not Rural Health Clinic. The observations
come from the SUPPORT study of critically ill ICU patients.

## Files

- `rhc.csv`: raw 5,735-row public RHC file. Place it at `data/raw/rhc.csv`.
- `rhc_opm_ready.csv`: prepared table with treatment, two outcomes, four proxies, and
  67 baseline X covariates.
- `prepare_rhc_opm.py`: reproducible preparation script.

## Main definitions

- `A_rhc = 1`: RHC was used within the first 24 hours; 2,184 treated and 3,551 controls.
- `Y_survival_days_30`: days from admission to death or censoring at day 30 (`t3d30`).
  Use this endpoint for reproducing the proximal-paper estimates measured in days.
- `Y_survived_30`: binary secondary endpoint, 1 if alive at day 30.
- `V_pafi1`, `V_paco21`: treatment-inducing proxies. The Cui paper denotes these by `Z`.
- `W_ph1`, `W_hema1`: outcome-inducing proxies.
- `X_*`: 67 remaining baseline covariates after reconstructing the 71-covariate
  Hirano–Imbens-style set and removing the four proxies.

## Important modeling note

The OPM prompt says “outcome: 30-day survival,” but the proximal JASA application defines
Y as the number of days between admission and death or censoring at day 30. Therefore,
the linear-POR reproduction gate should use `Y_survival_days_30`. The binary endpoint is
included only as an additional analysis unless the research protocol is explicitly changed.

## Basic validation

- Rows: 5,735
- Treated: 2,184
- Controls: 3,551
- Survived 30 days: 3,817
- Died within 30 days: 1,918
- Reconstructed baseline variables: 71
- Proximal X variables after removing proxies: 67
