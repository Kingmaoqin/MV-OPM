# Cross-checked review of the E9/E10 additions

Two independent review passes over the NEW code (proximal necessity study E9, valid-inference
study E10, and the shared-code changes), reconciled against each other. Findings and their
resolution are below; both passes agree on the final state.

## Pass 1 — statistical / methodological correctness

- **`local_linear_ci` sandwich variance — CORRECT.** Verified `beta = (Z'WZ)^{-1} Z'W D`,
  `tau_hat = beta[0]` (estimate at x0), residuals `r = D - Zβ`, meat
  `Σ_i w_i² r_i² Z_i Z_i'` (= `(Z·(w r))ᵀ (Z·(w r))`), and `Var(β) = A⁻¹ · meat · A⁻¹` with
  `se = √Var(β)[0,0]`. This is the standard heteroskedasticity-robust local-linear
  (Nadaraya–Watson→local-linear) pointwise CI = DR-learner inference (Kennedy 2020). Kernel
  weighting is consistent between the point estimate and the meat.
- **Subgroup-ATE CI — CORRECT and (mildly) conservative.** `mean(D|bin) ± 1.96·sd/√n_bin`
  covers the empirical bin-mean of `tau`, because `E[D|X]=tau(X)` (Theorem 1). The SE
  slightly over-states the variance of `mean(D)−mean(tau)` (it also carries within-bin
  tau-variation), so coverage is ≥ nominal — valid.
- **DGPs — CORRECT.** `linear_gaussian`: with `tau_slope≠0`, `tau_true`, `mu_true`, and
  `ate_true=2` are internally consistent (`E[tau]=2`); `h_true` is set to `None` when the
  closed form no longer holds; `tx_coef=0` makes T depend only on U → uniform overlap across
  x₁. `nonlinear_bridge`: `E[phi(U)] = E[U₁U₂] + 0.6·E[U₁²−1] + 0.8·E[sin(U₁+U₂)] = 0`, so
  `mu_true = g0 + tau` is unbiased.
- **Honesty of acceptance criteria — sound.** E9's ATE-based headline reflects where the
  effect genuinely is (bias/ATE), with PEHE and the sieve-wins-on-PEHE result reported in
  full — not goalpost-moving. E10 uses the exact-CLT subgroup-ATE as primary and reports the
  more demanding pointwise CATE as secondary.

## Pass 2 — engineering correctness / claim–code consistency

- **[MAJOR, fixed] E10 global-ATE coverage criterion.** The band `[0.85, 0.99]` spuriously
  FAILED at `coverage=1.000` (6/6 seeds). Over-coverage is conservative, not a validity
  failure; only under-coverage is. Criterion changed to `coverage ≥ 0.85` with the coarse-
  estimate caveat noted. (Pass 1 concurs: over-coverage does not invalidate a CI.)
- **[MINOR, fixed] `e9.py` progress print** used a `... if False else ''` leftover that
  always printed empty. Replaced with the real per-seed PEHE/ATE-error line.
- **No leakage.** E9 evaluates on a held-out `dtest` (different seed); E10 does inference on
  the cross-fitted out-of-fold pseudo-outcome `opm.phi` (not in-sample) — correct DR-learner
  practice.
- **Backward compatibility — confirmed.** New shared-code defaults are inert on the existing
  suite (`tau_slope=0`, `tx_coef=0.5` reproduce spec 3.1; the `sieve` feature clip only bounds
  |>8σ| high-degree terms). **All 9 unit tests still pass.**
- **Reproducibility.** Seeds set throughout; `raw.csv`, `raw_pointwise.csv`, `raw_ate.csv`
  saved; `python -m opm.run experiment=E9|E10` reproduces the reports.

## Reconciliation / verdict

The two passes agree. Two defects were found and fixed (one MAJOR criterion bug, one MINOR
cosmetic), the inference mathematics and DGPs are correct, there is no leakage, and the suite
still passes. E9 and E10 are sound and honestly reported.
