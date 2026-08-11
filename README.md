# OPM v2 — Orthogonal Proximal Moments

A multi-treatment **proximal doubly-robust CATE learner** with an optional
diffusion-distillation module for distributional counterfactuals, built from the
specification in `执行意见`.

## What it is
Under unobserved confounding with proxies `(W, V)`, OPM estimates arm-wise CATEs
`tau_k(x)` via a strictly sequential pipeline:

1. **Stage 1 — bridges.** Outcome bridge `h_k(W,X)` and treatment bridge `q_k(V,X)`
   are estimated by kernel-moment (U-statistic) losses (NMMR-style, no adversary),
   cross-fitted over 5 folds, EMA-smoothed, early-stopped on a random-Fourier-feature
   residual diagnostic.
2. **Stage 2 — pseudo-outcomes.** Out-of-fold doubly-robust pseudo-outcomes
   `phi_k = 1{T=k} q_k (Y - h_k) + h_k` are regressed on `X` (MLP) to get `tau_hat`;
   ATE contrasts get influence-function CIs.
3. **Stage 3 — generator (optional).** A conditional DDPM whose sample MEAN is
   distilled toward the proximal `mu_hat_k` (one tunable knob, `lambda_distill`).

Honest degradation: with no proxies the method becomes a standard multi-treatment
DR-learner (`mode="dr_fallback"`).

## Layout
```
opm/            package: dgp/ bridges/ estimator/ generator/ baselines/ diagnostics/ eval/ experiments/
configs/        one YAML per experiment (E1..E8)
tests/          pytest unit tests (spec Section 7)
results/{Ex}/   REPORT.md (PASS/FAIL vs acceptance criteria) + raw.csv + figures + LaTeX tables
data/raw/       drop rhc.csv here (optional); HAMD anchor is loaded from a local path
DECISIONS.md    every design choice / documented deviation
```

## Run
```bash
pip install -e .                       # env: MDPC (torch 2.6, econml 0.16, sklearn 1.6)
pytest -q                              # unit tests 1..7
python -m opm.run experiment=E1        # consistency & rates (M1 hard gate)
python -m opm.run experiment=E3        # main benchmark (S1/S2 + corruption)
python -m opm.run experiment=E6        # ablations incl. the key (f) c_U sweep
# ... E2, E4, E5, E7, E8 likewise
```

## Milestones (spec Section 8)
- **M0** repo + DGPs + unit tests 1,5,6 — green.
- **M1 (hard gate)** bridges + estimator; unit tests 2,3,4,7; **E1 PASS** — required before benchmarks.
- **M2** E2 · **M3** baselines + E3 · **M4** E4 (HAMD anchor) + E5 · **M5** generator + E6/E7/E8.

See `results/SUMMARY.md` for the consolidated PASS/FAIL board and `DECISIONS.md` for
deviations (notably: RHC unavailable → local HAMD used as the real-data anchor).
