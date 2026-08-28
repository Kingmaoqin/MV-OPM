# Review A — MV-OPM Round-3 exploratory study: methods / DGP-validity audit

**Reviewer role:** independent skeptical methods/statistics/DGP reviewer. Verdict formed by reading
the code and *running the pipeline live* (env `/home/xqin5/.conda/envs/MDPC/bin/python`), not from docs.

---

## OVERALL VERDICT: **MIXED — the two headline failures are GENUINE, but the metric that reports them (oracle-ratio / "catastrophic > 1.5×") is badly miscalibrated, and the report's *mechanistic interpretation* of the nonlinear failure is partly wrong.**

The user's hypothesis — "these bad results are artifacts of bugs / bad design / degenerate datasets"
— is **only partially right**:

- **The DGPs are NOT degenerate.** All five scenarios have a well-defined ATE, real overlap, and
  proxies that genuinely carry latent-confounder information, and in every scenario at least one
  candidate recovers the ATE almost exactly. So these are *real selection problems*, not broken data.
- **The two specific failures are REAL in absolute terms**, not denominator illusions:
  - `nonlinear` MSES: median **absolute** ATE error **1.53 on a true ATE of 2.0 (76% of the effect)**.
  - `hamd` `variance_only`: median absolute regret **0.25 on mean|ATE| 0.377 (66% of the effect)**.
- **But the *metric* is an artifact.** The oracle-ratio (`err/(best+1e-12)`) and the "catastrophic =
  err > 1.5×best" flag are unnormalized by effect size and explode whenever `oracle_best_error` is
  small. This makes benign selectors look "catastrophic" (e.g. `variance_only` is "37.5% catastrophic"
  in `nonlinear` on a 0.033 error = 1.6% of the ATE; "94% catastrophic / 5.4×" in `exact` on a
  0.25 regret = 12.5% of the ATE). The specific numbers **44×** and **5.04×** overstate the severity.
- **The report's headline mechanism for the nonlinear MSES failure ("moment compatibility is
  anti-correlated with true causal error") is misleading.** The real driver is that the moment
  *screen is a studentized test with essentially no power against high-variance candidates*: the raw
  scale-free discrepancies are near-identical for the accurate and the catastrophic candidates, and
  the pass/fail flips purely on the candidate's residual variance.

Per-scenario:

| scenario | reported failure | genuine? | note |
|---|---|---|---|
| `nonlinear` | MSES 44× / 94% cat | **GENUINE failure, metric-inflated number, mis-explained mechanism** | abs err 1.53/ATE 2.0; screen power-artifact drives it |
| `hamd` | `variance_only` 100% cat / 5.04× | **GENUINE** | variance fooled by a collapsed-q-bridge low-variance candidate; abs regret 66% of effect |
| `exact` | `variance_only` 94% cat / 5.4× | **METRIC ARTIFACT (mostly)** | abs regret 0.25 on ATE 2.0 = 12.5%; "catastrophic" massively overstated |
| `nonlinear` | `variance_only` 37.5% cat | **METRIC ARTIFACT** | abs err 0.033 = 1.6% of ATE flagged "catastrophic" |
| `S1`/`S2` | (not flagged) | fine | selectors behave sensibly |

---

## Severity-ranked findings

### CRITICAL

#### C1. The moment SCREEN pass/fail is a studentization power artifact, not moment satisfaction — the report's central nonlinear mechanism is mis-stated.
- **Where:** `opm/validation/moment_tests.py::studentized_rff_max_test` (statistic
  `max_j |mean_j / se_j|`, `se_j = sd(1{T=k}(Y−h_k)·g_j)/√n`); consumed as the screen in
  `opm/validation/selector_v2.py::screen_candidate` and `selector_variants.py::_screen_survivors`.
- **What's wrong:** the test studentizes each candidate's moment by *that candidate's own residual
  scale*. A candidate whose bridge `h` is wild has an enormous residual variance, so even large
  moment violations are statistically indistinguishable from zero → high p-value → **passes**. A
  precise candidate (small residual) has tiny SE, so *even trivial* violations become "significant" →
  **fails**. The screen is therefore approximately a monotone function of (inverse) candidate
  variance, confounded with precision.
- **Evidence (live pipeline, `nonlinear` seed 77000003):** raw scale-free discrepancies are
  essentially equal across the good and the catastrophic candidates, but the screen flips:

  | cand | ATE err | po_var | D_h (raw) | D_q (raw) | min p_h | screen |
  |---|---:|---:|---:|---:|---:|:--:|
  | kernel_kernel | **0.000** | 11.9 | 0.0178 | 0.0295 | 0.003 | **False (rejected)** |
  | sieve2_sieve2 | **9.34** | 29181 | 0.0170 | 0.0413 | 0.060 | **True (passed)** |
  | sieve3_sieve3 | **11.49** | 61283 | 0.0246 | 0.0747 | 0.053 | True (passed) |

  Same raw D_h (0.017 vs 0.018 — sieve2 is *not* better at the moment), opposite screen decision,
  flipped only by the 2500× larger residual variance. Reproduce:
  `python /tmp/.../scratchpad/reviewA_screen.py` (writes `reviewA_screen.out`).
- **Impact:** MSES = "screen → then min-variance". In `nonlinear` the screen *removes every accurate
  low-variance candidate from the survivor set*, forcing the min-variance step to choose among only
  the catastrophic high-variance survivors. So the MSES failure is real, but it is caused by the
  *screen gate*, and the report's stated mechanism — "moment compatibility is anti-correlated with
  causal error" (ρ = −0.58) — mislabels a **test-calibration defect** (no power vs high-variance
  candidates) + **an underpowered RFF bank** (D barely separates err-0.0 from err-11 candidates) as
  an intrinsic property of identifying moments. The proposed fix ("soft-weight the same moment
  signal") works largely by *re-injecting the variance/precision information the studentized screen
  destroys*, not by using the moment better. This should be reframed before any paper claim.

### MAJOR

#### M2. `oracle_ratio` and the "catastrophic > 1.5× oracle_best" flag are not normalized by effect size and are unreliable whenever `oracle_best_error` is small.
- **Where:** `scripts/mvopm_variant_sweep.py:101-102`, `opm/experiments/mvopm_round2/core.py:189-190`,
  and the aggregation in `scripts/mvopm_variant_aggregate.py` (`CATASTROPHIC=1.5`).
- **What's wrong:** `oracle_ratio = err/(best+1e-12)` and `cat = 1[err > 1.5·best]` divide by a
  quantity that is often tiny because the library genuinely contains a near-exact candidate.
  `oracle_best_error` distribution (from raw rows):
  `nonlinear` median 0.0196 (5/16 seeds < 0.01), `exact` median 0.054 (3/16 < 0.01). With `best`≈0.02,
  the "catastrophic" cutoff is 0.03 — so a 0.033 absolute error (1.6% of the ATE) is "catastrophic".
- **Evidence:** the *same* absolute regret (~0.25) is reported as "5.4×, 94% catastrophic" in BOTH
  `exact` (ATE 2.0 → 12.5% of effect, benign) and `hamd` (mean|ATE| 0.377 → 66% of effect, real).
  The metric cannot tell them apart. `variance_only` `nonlinear` "37.5% catastrophic" corresponds to a
  median absolute error of **0.033** (essentially oracle). Reproduce:
  `python /tmp/.../scratchpad/reviewA_parse.py`.
- **Impact:** every "N× / X% catastrophic" headline is inflated to an unknown degree. The *rankings*
  survive (see D1), and the two big failures survive on absolute error, but the reported magnitudes
  should be replaced by **absolute regret normalized by |ATE|** (or by a floor on the denominator,
  e.g. `max(best, c·|ATE|)`). Recommended replacement numbers already computed:

  | case | oracle-ratio (reported) | median abs err | abs err / mean|ATE| | verdict |
  |---|---:|---:|---:|---|
  | MSES nonlinear | 44.06 | 1.53 | **0.76** | genuine catastrophe |
  | variance_only nonlinear | 1.16 | 0.033 | 0.016 | benign (metric said 37.5% cat) |
  | variance_only hamd | 5.04 | 0.315 (regret 0.25) | **0.66** | genuine |
  | variance_only exact | 5.41 | ~0.30 (regret 0.25) | 0.13 | mostly benign |

#### M3. The `hamd` `variance_only` failure is genuine, but its mechanism is a *collapsed q-bridge* on one candidate, not a broad property of the DGP.
- **Where:** selection in `selector_variants.py::variance_only` (min `po_var`); variance defined in
  `opm/validation/selector_v2.py::pseudo_outcome_uncertainty` (per-sample contrast variance, **not**
  divided by n → an efficiency/spread criterion, bias-blind by construction).
- **What's wrong (but is a real finding):** in `hamd`, `sieve3_sieve3` is the min-variance candidate
  in **all 16/16 seeds** (po_var ≈ 1.9 vs ≥14 for the good candidates). Its q-bridge has collapsed —
  `max_q ≈ 4.4`, raw `D_q ≈ 1.87` (a massive treatment-moment violation), `q_balance ≈ 0.90` — so
  the DR correction term is near-zero, `phi ≈ plug-in h`, giving artificially low spread but a biased
  ATE (err ≈ 0.31). `variance_only` (bias-blind) always deploys it.
- **Evidence:** live pipeline `hamd` seed 77000000 (`reviewA_screen.out` line 21) and the logged
  per-candidate diagnostics (`reviewA_parse.py` / raw `candidate_*` lines): the moment screen
  (`screen_pass=False`) AND `q_balance=0.90` both correctly reject `sieve3`; only pure variance is
  fooled. That is exactly why MSES (median 1.53) and the q-gated rules beat `variance_only` (5.04)
  here.
- **Impact:** the "symmetric failure" narrative holds *as an empirical fact*, but note the asymmetry
  of cause: in `hamd` the screen is *correct* and variance is fooled by a degenerate candidate; in
  `nonlinear` the screen is *broken* (C1) and variance is correct. A single collapsed candidate
  (sieve3's q-solve) drives the entire `hamd` `variance_only` catastrophe — worth confirming this is
  not itself a sieve-degree-3 fitting pathology (see Q1) before treating it as a fundamental result.

### MINOR

#### m4. `n_boot = 299` gives coarse p-value resolution near the screen threshold.
- **Where:** sweep default `--n-boot 299`; p-value `= (1+exceed)/(n_boot+1)` in
  `studentized_rff_max_test`. Resolution ≈ 1/300 ≈ 0.0033; the Bonferroni screen threshold is
  `alpha/K` = 0.025 (K=4) or 0.025 (K=2). The minimum attainable p is 0.0033 (`min p_h=0.003` seen
  throughout). Not the driver of the failures (C1 is robust to n_boot — it is about *scale*, not MC
  noise), but the exact survivor set near p≈0.025 has avoidable Monte-Carlo jitter. Recommend
  n_boot ≥ 999 for anything used to certify a decision.

#### m5. Sample size / folds are adequate, not a flaw. n=2000, K=4 → ~410–530 per arm; 4-fold OOF
leaves ~75% per fold. The good candidates achieve near-oracle error under exactly these settings, so
per-arm n is sufficient for the library; the failures are selection failures, not sample-starvation.
(hamd uses the full real table n=1892 with no subsampling, since 1892 < 2000 — confirmed, no
degenerate subsample.)

---

## What is CORRECT / holds up (so the user knows what is safe)

1. **Reported aggregate numbers are faithfully computed.** I reproduced MSES nonlinear median
   oracle-ratio **44.061** and variance_only hamd **5.040** exactly from the raw JSONL, and the
   live pipeline reproduces the per-candidate errors and screen decisions. No aggregation/bookkeeping
   bug found in `mvopm_variant_aggregate.py` or `selector_metrics`.
2. **All five DGPs are valid and non-degenerate** (`reviewA_dgp.py`):
   - ATE well-defined and non-trivial: arm-1 ATE = 1.0 in every K=4 scenario, ATE = 2.0 in
     exact/nonlinear. (S1/hamd `mean|ATE|`≈0.38 only because arms 2–3 are deliberately near-zero;
     that is a heterogeneity design choice, not a degenerate ATE.)
   - Overlap/positivity fine: min per-arm mean true propensity ≈ 0.23; 1st-percentile realized-arm
     propensity ≈ 0.05. No positivity violation.
   - Proxies carry real latent-U signal: canonical corr(W,V) ≈ 0.73 (S1/S2/nonlinear/hamd);
     corr(U,W_cat)=0.60, corr(U,V_cat)=0.48 in `exact`. A valid proximal bridge exists.
   - Arm counts healthy (min per arm 410–1026).
3. **The library is adequate in every scenario** — at least one candidate is near-oracle
   (kernel/sieve1 achieve ATE err 0.001–0.2 in nonlinear and 0.006–0.08 in hamd). So "selection
   failure" is a *meaningful* concept here; abstention correctly never fires. This rules out the
   "library too weak, selection is moot" alternative.
4. **`variance_only` is bias-blind by design and this is correctly demonstrated.** The min-variance
   criterion (spread of OOF pseudo-outcomes, undivided by n) can be captured by a collapsed
   low-variance candidate — the hamd result is a legitimate instance of that known failure mode.
5. **ATE error is measured off the OOF pseudo-outcomes directly** (`ate_with_ci(phi[:,k]−phi[:,0])`),
   not off the CATE head, so the ATE failures reflect genuine bridge/selection quality, not a
   Stage-2 head artifact.
6. **Oracle isolation is real** — selectors read only observed/OOF features; truth enters only in
   scoring (consistent with the `test_oracle_blind` claim).

---

## Open item to hand to the code-correctness reviewer (Q1)
Why does `sieve3_sieve3`'s q-bridge collapse to `max_q≈4`, `D_q≈1.87` (hamd) while degree-1/2 sieves
do not? A degree-3 polynomial sieve producing *smaller, more degenerate* weights than degree-1 is
counter-intuitive and may indicate an ill-conditioned/over-regularized q-solve in
`opm/bridges/sieve.py`. If sieve3's collapse is a numerical artifact rather than genuine
overfitting, the hamd `variance_only` catastrophe is partially manufactured by a broken candidate
(still a real hazard for variance-only, but weaker as a "fundamental" claim). This is code-behavior,
outside my DGP/metric mandate — flagging for cross-check.

## Reproduction index (scratchpad, prefix reviewA_)
- `reviewA_parse.py` — reproduces headline ratios + absolute-error/normalized views + oracle_best dist.
- `reviewA_dgp.py` — DGP validity: ATE, overlap, arm counts, U-proxy canonical correlations.
- `reviewA_screen.py` → `reviewA_screen.out` — live per-candidate D (raw) vs studentized p vs
  variance vs error (the C1 power-artifact evidence).
