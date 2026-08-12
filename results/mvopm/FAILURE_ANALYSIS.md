# MV-OPM Pilot — Failure Analysis (Gate = HOLD)

The development pilot (394 completed rows across Studies A–I; dev seeds) was scored against the
Go/No-Go gate **frozen before the aggregate was inspected** (`docs/reviews/CROSS_RECONCILIATION.md`,
primary target = ATE error). Decision: **HOLD.** This document diagnoses why, honestly, and states
the strongest evidence-supported conclusion. No prespecified score was changed; no new score is
promoted to confirmatory (MV-OPM §11/§15/§29).

## What passed
- **Mechanism (Study A) — STRONG.** On the exact-bridge DGP (real `h_true`, `q_true`), the held-out
  moment discrepancies track the TRUE bridge errors:
  `Spearman(D_h, ‖ĥ−h‖)=0.86`, `Spearman(D_q, ‖q̂−q‖)=0.90`,
  `Spearman(moment_product, true_err_product)=0.83`.
  → **Held-out identifying-moment violations are a valid observed-data measure of proximal BRIDGE
  ERROR (bias).** This is the load-bearing Observation C/D, confirmed at the mechanism level.
- **Selection vs naive baselines (C2) — STRONG.** Median oracle-ratio (ATE): product ≈ **1.2–1.5**
  vs **fixed-kernel ≈ 3–14** (kernel catastrophic; cat-rate ≈ 1.0), and product beats h-only
  (1.6–20), q-only (1.3–6.4), and random (1.4–2.4). Product is the best of the prespecified scores.

## What failed (the gate)
- **Per-instance ranking (C1) — FAIL.** `Spearman(product score, ATE error)` across the 6
  candidates, averaged over seeds: S1 0.13, S2 0.10, nonlinear 0.07, hamd 0.45 → only 1/4 families
  exceed 0.2. Product does not reliably beat the **strong fixed-sieve1** baseline (which is often
  the oracle-best), and cannot rank near oracle.
- **Abstention (Study I) — FAIL.** `min_total_D` barely separates valid vs deliberately broken
  proxies (0.076 vs 0.079). The normalized moment can still be driven low by *some* candidate even
  when proxies are shuffled/noised.

## Root cause (definitive, not a power artifact)
The identifying moment measures **bias** (`E[R|Z]=0`), and residuals are normalized by `sd(R)` so
the diagnostic is scale-free — which makes it **blind to estimation VARIANCE**. Over-flexible,
low-bias–high-variance candidates satisfy the held-out moment yet have catastrophic CATE error.
Direct evidence (nonlinear scenario, per-candidate means):

| candidate | D_h | D_q | product | **ATE err** | ATE CI width |
|---|---|---|---|---|---|
| sieve1_sieve1 (oracle-best) | 0.037 | 0.063 | 0.0023 | **0.20** | 0.75 |
| **sieve2_sieve2** | 0.034 | 0.070 | **0.0024** | **5.10** | 27.8 |
| **sieve3_sieve3** | 0.036 | 0.093 | 0.0033 | **11.22** | 35.6 |
| kernel_kernel | 0.188 | 0.085 | 0.016 | 0.95 | 0.45 |

`sieve2/3` have the **lowest** moment discrepancy (they fit the identifying moment out-of-sample)
but the **worst** CATE error — the product score cannot separate the oracle-best `sieve1` from the
catastrophic `sieve2`. The moment product tracks bias (Study A) but not the variance that dominates
these candidates' error.

## Exploratory finding (post-hoc — NOT confirmatory; MV-OPM §15/§29)
CATE error ≈ bias + variance. The moment measures bias; the **ATE-CI width** is a cheap observed-data
variance proxy that cleanly flags `sieve2/3` (width 28–36 vs 0.75). A post-hoc combined score
`log(product) + log(CI_width)` ranks candidates by ATE error at **Spearman 0.67–0.75 in all four
families** (vs 0.03–0.36 for product alone). This was invented after seeing the pilot and therefore
**cannot be reported as confirmatory**; it is a *prespecified hypothesis for a future study*:
> *held-out identifying-moment adequacy (bias) must be combined with an estimation-variance signal to
> select reliable proximal CATE estimators.*

## Strongest evidence-supported conclusion (this run)
1. **Confirmed:** held-out proximal identifying-moment discrepancies are a reliable observed-data
   measure of proximal **bridge error/bias** (mechanism Spearman 0.83–0.90) — a genuine
   bridge-**adequacy / falsification** tool that needs no counterfactual labels.
2. **Confirmed:** using it to select avoids the **catastrophic** failures of fixed-kernel and
   single-side heuristics.
3. **NOT established:** the prespecified product-moment score does not achieve near-oracle CATE
   selection nor beat a strong low-degree sieve, because it is variance-blind (it targets bias).
4. **Contribution type:** an empirical **model-validation / falsification** framework for proximal
   bridges — NOT a "near-oracle selector." (MV-OPM §1/§32 explicitly permits this framing.)

## Next authorized action (per protocol)
STATUS = **HOLD**. Do NOT run the expensive final matrix on this hypothesis. The prereg direction for
a follow-up confirmatory study (fresh seeds) is the **bias×variance** validation above, defined here
before any confirmatory run. Real-data (RHC, Study J) is reported for stability only, without oracle
claims.
