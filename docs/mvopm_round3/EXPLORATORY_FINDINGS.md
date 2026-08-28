# Round-3 exploratory selector study — findings

> **POST-REVIEW CORRECTIONS (2026-08-28).** Two independent code/design reviews audited this study;
> see [CODE_REVIEW_RECONCILIATION.md](CODE_REVIEW_RECONCILIATION.md). Net: the pipeline and DGPs are
> correct (no result-corrupting bug; datasets not degenerate), and the failures are genuine in
> absolute terms — BUT three things in an earlier version of this doc were wrong and are corrected
> below: (1) **severity was overstated** — the `oracle_ratio` / "catastrophic>1.5×" metric is
> denominator-inflated and unnormalized by effect size; the PRIMARY metric is now
> `norm_regret = regret / mean|ATE|` and a `norm_catastrophic` flag (regret > 0.25×effect). Several
> "5×/90% catastrophic" cases are benign (e.g. `variance_only` in `exact`: norm_regret 0.126,
> norm-catastrophic 0.000). (2) **`mses_varcap` had a fallback BUG** (fixed): it now repairs the
> nonlinear catastrophe (norm_regret 0.001) rather than being ≡MSES. (3) the moment-screen mechanism
> is reframed: it is a **studentized-test power/variance confound used as a hard gate**, not an
> intrinsic anti-correlation of identifying moments (the test itself is correctly calibrated).

**Status: EXPLORATORY. Fresh seeds only (base 77,000,000), disjoint from the Round-2 confirmatory
final seeds. Nothing here is tuned on the confirmatory seeds, and nothing here alters the frozen
Round-2 verdict (`NOT_SUPPORTED`).** This study characterises *why* the Round-2 selector failed and
whether any alternative rule or signal does measurably better, using held-out DGP truth only for
scoring — never for selection.

## 1. What we ran

- **22 selector rules** (`opm/validation/selector_variants.py`), spanning the full trust spectrum:
  variance-only; the frozen MSES screen-then-variance; moment-only rules (product, biasvar, rank_sum,
  soft_screen); variance-primary hybrids (var_tiebreak, complexity_var); a variance-capped MSES
  (`mses_varcap`); and **new q-normalization rules** (`q_gated_variance`,
  `q_and_moment_gated_variance`, `min_q_balance`, `varq_combo`, `varq_moment_combo`).
- **5 scenarios × 16 fresh seeds** = 80 tasks, `n = 2000`, 4-fold OOF, `n_boot = 299`. Scenarios:
  `exact` (finite-proxy, K=2), `S1`/`S2` (synthetic main, K=4), `nonlinear` (the Round-2 D2/E9
  failure, K=2), `hamd` (semi-synthetic on real depression-trial covariates, K=4).
- Selection uses only observed / OOF features (moment p-values, RFF-L2 discrepancies, OOF
  pseudo-outcome variance, Wald CI width, and the observable q-normalization diagnostics
  `q_balance`, `max_q`, `ess`). A unit test (`test_oracle_blind`) proves the selectors never read a
  causal-truth key. Truth enters only in scoring.
- Metrics: primary ATE (oracle-ratio, regret, catastrophic > 1.5×, top1) plus **non-primary**
  targets — CATE/PEHE oracle-ratio, 95% Wald CI coverage, deployed q-safety (`max_q`, `ess`,
  `q_balance`), abstention, and per-regime score-vs-error rank correlations.

Tables and figures: [`results/mvopm_variants/EXPLORATORY_VARIANTS_REPORT.md`](../../results/mvopm_variants/EXPLORATORY_VARIANTS_REPORT.md),
`summary_pooled.csv`, `summary_by_scenario.csv`, `score_error_correlations.csv`,
`figures/fig1_regime_oracle_ratio.png`, `figures/fig2_score_error_corr.png`.

## 2. Finding 1 — the failure is regime-specific and *symmetric*

Round-2 concluded MSES fails and variance-only is "better overall". Round-3 sharpens this: **neither
the moment screen nor variance is universally reliable — each has its own catastrophe regime.**

| regime | MSES (screen→min-var) | variance-only |
|---|---:|---:|
| nonlinear | **median 44.06, catastrophic 0.94, PEHE-ratio 12.4** | 1.157, cat 0.375 |
| hamd | 1.53, cat 0.50 | **median 5.04, catastrophic 1.00** |

- In **nonlinear**, the RFF identifying-moment screen *rejects the oracle candidate* and retains
  flexible sieve candidates that pass the moment test but carry pathological weights; MSES then
  deploys one and is catastrophic at both the ATE *and* CATE level (PEHE-ratio 12.4).
- In **hamd**, variance-only is catastrophic on **every** seed (catastrophic rate 1.00): the
  lowest-OOF-variance candidate is systematically not the lowest-error one there.

So "variance-only is better overall" is true only *on average across regimes*; variance-only has a
100%-catastrophic regime of its own.

## 3. Finding 2 — `q_balance` is the only *uniformly* informative signal

Spearman ρ between each candidate selection score and its true ATE error, pooled within a scenario
(positive = informative: a low score picks a low-error candidate):

| scenario | ρ(variance) | ρ(moment-violation) | **ρ(q_balance)** | ρ(max_q) |
|---|---:|---:|---:|---:|
| S1 | 0.596 | 0.027 | 0.581 | 0.540 |
| S2 | 0.528 | −0.098 | 0.566 | 0.532 |
| exact | 0.259 | 0.400 | 0.271 | 0.283 |
| hamd | **0.083** | 0.544 | **0.635** | 0.054 |
| nonlinear | 0.654 | **−0.582** | **0.613** | 0.569 |

- **Variance** fails in hamd (0.083 — nearly useless).
- **Moment-violation** (the MSES screen's p-value signal) *flips sign*: strongly anti-informative in
  nonlinear (−0.582), null in S1/S2, informative only in exact/hamd. **Mechanism (post-review
  correction):** this is not an intrinsic property of identifying moments. The studentized moment
  test's *power* is confounded with residual variance — a wildly over-fit candidate has enormous
  residual variance and so its moment violation is statistically undetectable (high p, "compatible"),
  while a precise candidate's tiny remaining bias is detectable (low p). So in `nonlinear`, where the
  bad candidates are the high-variance over-fit sieves, "moment-compatible" ≈ "high-variance
  over-fit" ≈ "high-error", producing the negative ρ. The moment test itself is correctly calibrated
  (uniform p under the true bridge); the pathology is *using a studentized test as a hard screen*,
  where pass/fail tracks precision rather than bias. See
  [CODE_REVIEW_RECONCILIATION.md](CODE_REVIEW_RECONCILIATION.md) §disagreement.
- **`q_balance`** — the observable treatment-bridge self-normalization moment
  `|E_n[1{T=k} q_k] − 1|`, which needs no counterfactual truth — is **positively informative in all
  five regimes (0.27–0.64)**, including exactly the two regimes where the other signals fail.

This is the central positive result: the Hájek self-normalization moment is a more regime-robust
adequacy signal than either the OOF variance or the RFF identifying-moment p-value that MSES relied
on.

## 4. Finding 3 — soft multi-signal *weighting* beats any single signal *and* any hard gate

The robust rules are exactly the **library-standardized soft sums of variance with ≥1 adequacy
signal**; every single-signal rule and every hard-gate rule has a catastrophe regime. Ranking by the
honest robustness criterion — the *worst* per-regime median (a robust rule has no regime where it
blows up):

*(The `median` / `catastrophic` columns below are the LEGACY denominator-inflated oracle-ratio metric,
kept for continuity; the primary effect-size-normalized numbers are in `summary_pooled.csv` and the
banner. The qualitative ranking is unchanged, and `mses_varcap` is shown POST-FIX.)*

| selector | pooled median | pooled catastrophic | **worst-regime median** | worst-regime cat |
|---|---:|---:|---:|---:|
| soft_screen[λ=1.0]  (var + moment, soft) | 1.350 | 0.438 | **1.514** | 0.562 |
| biasvar_mult  (var + RFF-discrepancy) | 1.379 | 0.450 | **1.595** | 0.625 |
| varq_moment_combo  (var + q_balance + moment) | 1.395 | 0.463 | **1.689** | 0.688 |
| **mses_varcap[τ=3] (screen→cap→var, FIXED)** | 1.416 | 0.488 | **2.376** (S2) | 0.688 |
| rank_sum  (var + moment ranks) | 1.354 | 0.450 | 2.018 | 0.688 |
| varq_combo  (var + q_balance) | 1.768 | 0.562 | **5.408** (exact) | 0.875 |
| variance_only | 2.512 | 0.662 | **5.408** (exact/hamd) | 1.000 |
| min_q_balance | 1.947 | 0.588 | 5.408 (exact) | 0.812 |
| MSES  (moment, HARD gate) | 1.866 | 0.600 | **44.06** (nonlinear) | 0.938 |

Note: after the fallback-bug fix, `mses_varcap` no longer tracks MSES — it degrades to global
minimum-variance when the cap empties the survivor set, so it avoids the nonlinear catastrophe
(worst-regime 2.38 instead of 44.06) and joins the robust soft-rule tier. This is itself an instance
of the headline: converting the moment screen from a decisive gate into a *defeasible* one (fall back
to variance) removes its catastrophe.

Reading:

- **The single decisive lever is hard-gate vs soft-weight of the *same* moment signal.**
  `soft_screen[λ=1.0]` and MSES both use the RFF identifying-moment p-values and nothing else beyond
  variance. MSES *gates* on them (survive-or-die) and is 44× in nonlinear; `soft_screen` *weights*
  the identical signal against variance and is 1.33 there — robust. Softening the same signal removes
  the catastrophe.
- **No single-signal or hard-gate rule is robust.** variance-only, min_q_balance, and even
  `varq_combo` (variance + q_balance, no moment term) each have a 5.4× catastrophe regime — for
  `varq_combo` it is `exact`, where the moment signal is the informative one (§3, ρ 0.40) and the
  variance/q_balance pair is weak. The moment *term* is what fixes `exact`; the q_balance *term* is
  what fixes `hamd`; variance fixes `nonlinear`. A robust rule needs the soft sum, because the three
  signals' failure regimes do not coincide.
- **Three soft fusions are statistically indistinguishable at the top** (worst-regime 1.51–1.69):
  `soft_screen[λ=1.0]`, `biasvar_mult`, and `varq_moment_combo`. We do not claim one beats the others
  on 16 seeds/regime; the robust conclusion is the *class* (soft fusion), not a single winner.
- **`q_gated_variance` (hard q-screen → min variance) independently repairs the nonlinear
  catastrophe** (1.157 vs MSES 44.06): the pathological-weight survivors the RFF screen admits are
  exactly what an observable q-normalization gate rejects.

## 5. Finding 4 — non-primary metrics

- **`mses_varcap` — CORRECTED.** An earlier version reported it "failed (≡MSES)" and attributed this
  to the catastrophic survivors being *low*-variance. Both claims were wrong: the nonlinear
  catastrophic survivors are *high*-variance (sieve2/3 OOF variance ≈ 52k/70k), and the equivalence
  to MSES was a **fallback bug** (`pool = capped or survivors` fell back to the survivor set, not the
  global minimum-variance candidate its docstring promised). After the fix, when the variance cap
  empties the survivor set `mses_varcap` degrades to global minimum-variance, and it **repairs the
  nonlinear catastrophe**: nonlinear norm_regret 0.001 (oracle-ratio 44.06 → 1.157), other scenarios
  unchanged. See the review reconciliation for the regression test.
- **CATE/PEHE.** ATE-oracle and PEHE-oracle are usually *different candidates* (P(match) = 0.12–0.38
  across regimes), so ATE-level selection is not a reliable CATE selector — consistent with the
  Round-2 caution that the MSES criterion (ATE-influence variance) is not aligned with PEHE. The
  nonlinear ATE catastrophe also propagates to CATE (MSES PEHE-ratio 12.4).
- **CI coverage is not a trustworthy deployment signal.** MSES retains ~0.88 nominal ATE-CI coverage
  in nonlinear *despite* a 44× oracle-ratio, because its intervals are wide (variance-blind). High
  coverage bought with uninformative width is not evidence of a good point estimate.
- **q-safety diagnostics are observable red flags.** In nonlinear the catastrophic MSES pick has
  `max_q ≈ 47`, `q_balance ≈ 0.33`, `ess ≈ 399`, versus `max_q ≈ 8.9`, `q_balance ≈ 0.02`,
  `ess ≈ 838` for the healthy oracle pick — separable without any truth. This is the mechanism the
  q-gate exploits.
- **Abstention** never fired: with the full 6-candidate library every regime always had a moment-
  screen survivor, so MSES never abstained here.

## 6. Honest limits

- **Still not deployable.** The best soft-fusion rules reach pooled median oracle-ratio ≈ 1.35–1.40
  and worst-regime median ≈ 1.51–1.69 — better than every single-signal or hard-gate rule, but still
  far from Round-2's preregistered ≤ 1.10 / ≤ 0.20 targets. This is a *relative* improvement and a
  *direction*, not a solved selector.
- **No single winner is claimed.** Three soft fusions (`soft_screen[λ=1.0]`, `biasvar_mult`,
  `varq_moment_combo`) are indistinguishable at the top on 16 seeds/regime; the supported claim is
  the *class* (soft weighting of variance with an adequacy signal), not any one rule, and the equal
  weights / round-number thresholds are untuned.
- **Exploratory scope.** 16 seeds/regime, single `n = 2000`, `n_boot = 299`, and **global-OOF**
  single-shot selection (not the Round-2 nested-crossfit confirmatory estimator). No paired
  confidence intervals are reported on the selector differences. These numbers rank rules; they do
  not certify one.
- **No re-tuning of the verdict.** Per protocol, none of these rules was fit on the confirmatory
  final seeds and none is proposed as a replacement confirmatory claim. `q_balance` thresholds
  (0.10) and `max_q` cap (20) are round-number defaults, not tuned.

## 7. What it means for the paper

Round-3 turns the Round-2 negative result into a sharper, more useful message and a concrete redesign
direction:

1. **Symmetric failure.** The unreliability is not "the screen is bad, use variance" — *both* the
   identifying-moment screen and the efficiency criterion have catastrophe regimes. A robust selector
   cannot rely on either alone.
2. **Fusion beats gating, and a new signal helps.** Every robust rule is a library-standardized soft
   sum of variance with ≥1 adequacy signal; every single-signal or hard-gate rule has a catastrophe
   regime. The clean controlled contrast is `soft_screen[λ=1.0]` vs MSES: the *same* RFF moment
   signal is robust when soft-weighted (nonlinear 1.33) and catastrophic when hard-gated (44.06).
   Separately, the treatment-bridge self-normalization moment (`q_balance`) is the only signal
   positively informative in all five regimes and gives a truth-free q-gate that repairs the
   nonlinear catastrophe. The evidence-backed redesign lead the Round-2 report asked for is
   therefore: **replace the hard identifying-moment screen with a soft multi-signal score, and add
   the q-normalization moment as a first-class term** (it never flips sign, unlike the RFF p-value).
3. **Diagnostic honesty holds.** The bridge-error diagnostics remain the strongly-supported
   contribution; this study adds a mechanistic account (regime-dependent signal informativeness) and
   a promising, still-unproven redesign lead — not a new confirmatory success claim.
