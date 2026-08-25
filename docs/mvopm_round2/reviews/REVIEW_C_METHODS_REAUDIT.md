# Reviewer C independent methods/statistics/algorithm re-audit

**Audit date:** 2026-08-24  
**Audited commit:** `aaef262` (`mvopm-round2-confirmatory`)  
**Execution checkpoint recorded in final rows:** `aa983a2`  
**Scope:** independent read-only re-audit of source, tests, preregistration, reconstructed final
tables/raw records, and the requested manuscripts. Existing Review A/B conclusions were not used as
premises. No scientific code or result was modified.

## Verdict

**MAJOR_REVISION_REQUIRED.** I found no evidence that the 6,820-row final corpus was fabricated or
incompletely aggregated, and the headline negative conclusion—MSES is not a reliable near-oracle
selector in this experiment—survives the checks below. However, the code and design do not support
calling the screen a calibrated “formal test,” the kernel and sieve competitors are not
representation-fair, and the report materially overstates the evidence for structural oracle
switching. The current manuscripts should not be submitted as a coherent current account until the
P1 findings are corrected and all historical documents are explicitly versioned.

There is no P0 finding that reverses the negative MSES verdict. There are **seven P1/major findings**
that affect method interpretation or reproducibility, plus P2/P3 reporting and test-coverage issues.

## Reproducible checks performed

1. `python -m pytest -q tests/test_mvopm_invariants.py`: **27/27 passed**. These are useful structural
   invariants, but they do not test statistical calibration or several defects below.
2. Aggregate key check on `final/raw.csv`: 6,820 rows, no duplicate
   `(study, config_id, scientific_seed)` key, no duplicate `array_task_id`, IDs span 0–6819, all
   aggregate `git_commit` values are `aa983a2`, and retained result seed equals scientific seed.
3. Exact treatment-bridge support check from
   `finite_proxy_exact.population_bridges(ExactProxyParams())`:
   `q=[[0.149765,1.393577,5.450256],[5.714657,1.766135,0.098341]]`; 2/6 population bridge cells are
   below 1.
4. Regime-level winner reconstruction from `candidate_by_scenario.csv`: mean-ATE-error winners across
   the 20 R regimes are `kernel_sieve1` (12), `kernel_kernel` (6), `sieve1_kernel` (1), and
   `sieve2_sieve2` (1). `sieve1_sieve1` and `sieve3_sieve3` are never regime-level mean-risk winners.
   Mean-risk winners change in 2/8 sample-size comparisons and 3/8 proxy-noise comparisons, not the
   reported 70.4% and 62.9% per-replicate-winner transition rates.
5. D2 target sensitivity: recomputing against the realized sample CATE mean rather than population
   ATE leaves 43 evaluable runs, MSES mean error 2.062, median error 1.202, median oracle ratio 41.55,
   catastrophic fraction 0.953, 26 errors above 1, maximum 11.323. Thus the estimand inconsistency is
   real but does not rescue MSES.
6. “Catastrophic” magnitude audit: among 335 core/switching rows flagged by the relative 1.5× rule,
   153 have selected absolute ATE error below 0.05 and 204 below 0.10. The event is a preregistered
   relative-regret event, not uniformly an absolute catastrophe.
7. I2 abstention audit: nested any-outer-fold abstention is 70% in the overregularized-kernel cell,
   while the full-sample OOF deployment abstention rate is 58%; the other two cells are 0% under both
   definitions.

## P1 / major findings

### C1. Kernel treatment bridge excludes the repository's own exact true bridge

**Evidence.** `KernelBridges` parameterizes every treatment bridge as
`clip(1 + softplus(g), max=q_max)`, hence enforces `q >= 1`
([`opm/bridges/kernel_moment.py:169`](../../../opm/bridges/kernel_moment.py#L169),
[`opm/bridges/kernel_moment.py:241`](../../../opm/bridges/kernel_moment.py#L241)). The exact DGP only
requires a positive bridge and solves it without a lower-one restriction
([`opm/dgp/finite_proxy_exact.py:74`](../../../opm/dgp/finite_proxy_exact.py#L74),
[`opm/dgp/finite_proxy_exact.py:92`](../../../opm/dgp/finite_proxy_exact.py#L92)). The deterministic
population solution above has minimum 0.09834 and two of six cells below 1. The sieve competitor,
in contrast, is clipped to `[0,50]`
([`opm/bridges/sieve.py:104`](../../../opm/bridges/sieve.py#L104)). The historical manuscript even
advertises `q>=1` as a virtue
([`docs/两天历程_OPM到MVOPM.md:77`](../../两天历程_OPM到MVOPM.md#L77)), while the Round-2 exact
derivation establishes only positivity
([`docs/mvopm_round2/EXACT_DGP_BIAS_DERIVATION.md:43`](../EXACT_DGP_BIAS_DERIVATION.md#L43)).

**Why it matters.** Kernel-vs-sieve comparisons conflate optimizer/function-class behavior with an
asymmetric and incorrect support restriction. Mixed candidates containing kernel q inherit this
restriction. The convergence audit cannot repair representation misspecification. Claims about fair
solver competition, fixed-kernel performance, and which bridge side drives switching are therefore
not cleanly identified.

**Required fix.** Use a positive parameterization that covers `(0,q_max)`, e.g. scaled sigmoid or
`softplus(g)+epsilon`, and separately study signed/unrestricted q if theory allows it. Add a unit test
that the exact q table lies in every solver's representable support. This is a method change and must
be evaluated on new seeds, not patched into the frozen final run.

### C2. The “formal” OOF moment-test p-values lack a valid nuisance-estimation/bootstrap argument

**Evidence.** Candidate h/q predictions are produced by overlapping K-fold training samples
([`opm/validation/oof.py:39`](../../../opm/validation/oof.py#L39),
[`opm/validation/oof.py:48`](../../../opm/validation/oof.py#L48)). The test then treats the pooled
observation-level moment rows as the bootstrap units, drawing independent Gaussian multipliers
([`opm/validation/moment_tests.py:86`](../../../opm/validation/moment_tests.py#L86),
[`opm/validation/moment_tests.py:93`](../../../opm/validation/moment_tests.py#L93)). This conditions on
estimated predictions and ignores the shared, overlapping training-sample variation. Moreover, each
one-sided bridge moment is not Neyman-orthogonal to its own nuisance: perturbing h changes
`E[I(T=k)(Y-h)g]` to first order, and similarly for q. The preregistration calls these formal tests
without stating rates or a refitted/bootstrap theorem
([`docs/mvopm_round2/ROUND2_PREREG.md:48`](../ROUND2_PREREG.md#L48),
[`docs/mvopm_round2/ROUND2_PREREG.md:61`](../ROUND2_PREREG.md#L61)). Existing tests check only
determinism of p-values, not null size
([`tests/test_mvopm_invariants.py:192`](../../../tests/test_mvopm_invariants.py#L192)).

**Why it matters.** Cross-fitting prevents own-observation training leakage, but it does not by
itself make the pooled fitted-residual rows i.i.d. or remove first-order bridge-estimation error from
the moment statistic. Therefore `p_h`, `p_q`, `p_DR`, oracle-survival, rejection, and abstention are
empirical scores unless a theorem or calibration study justifies the bootstrap. D2's 11.5% oracle
survival is consistent with severe practical miscalibration/incompatibility, although it alone does
not distinguish size from candidate misspecification.

**Required fix.** Either (a) downgrade all p-values/screens to heuristic compatibility diagnostics,
or (b) supply a nuisance-aware test: honest train/test splitting with candidate fits fixed relative
to an independent test sample, repeated splitting/cross-fit aggregation with a proved null law, or a
bootstrap that refits the nuisance procedure. Add exact-null size simulations across n, K, candidate
training methods, and fold counts before using “formal.”

### C3. Finite-sample kernel U-statistic training loss can be negative and is unbounded below in h

**Evidence.** The code minimizes the off-diagonal U-statistic
`vals' (K-diag(K)) vals / [B(B-1)]`
([`opm/bridges/kernel_moment.py:37`](../../../opm/bridges/kernel_moment.py#L37),
[`opm/bridges/kernel_moment.py:167`](../../../opm/bridges/kernel_moment.py#L167)). For two observations
with residuals `(a,-a)` and positive off-diagonal kernel value `c`, this loss is `-a^2 c`; h is not
output-clipped, so the empirical objective itself is unbounded below. AdamW weight decay and
held-out early stopping may mitigate this, but they do not make the training criterion an empirical
squared maximum moment. The manuscript describes the population nonnegative RKHS norm and then
passes directly to minimization of the unbiased U-statistic without acknowledging this optimization
pathology
([`docs/技术全志_1_数学与实现.md:79`](../../技术全志_1_数学与实现.md#L79),
[`docs/技术全志_1_数学与实现.md:89`](../../技术全志_1_数学与实现.md#L89)).

**Why it matters.** An overparameterized network can lower the training loss by exploiting negative
finite-sample directions rather than satisfying moments. This undermines solver stability and the
interpretation of the convergence audit as adequate training.

**Required fix.** Train with a nonnegative V-statistic/RFF squared-moment objective, a cross-pair
construction with controlled optimization, or an explicitly regularized objective with a proved
lower bound. Log training-objective minima and add adversarial tests showing the solver cannot win by
scaling alternating residuals.

### C4. “All six candidates switch” is mostly a per-realization winner-noise claim, not structural risk switching

**Evidence.** `oracle_best` is the minimum realized absolute ATE error among candidates on each
simulation seed
([`opm/experiments/mvopm_round2/core.py:117`](../../../opm/experiments/mvopm_round2/core.py#L117),
[`opm/experiments/mvopm_round2/core.py:119`](../../../opm/experiments/mvopm_round2/core.py#L119)). The
analysis counts those per-seed winners and their transitions
([`scripts/analyze_round2.py:347`](../../../scripts/analyze_round2.py#L347),
[`scripts/analyze_round2.py:356`](../../../scripts/analyze_round2.py#L356)). The final report converts
this into “all six candidates” and 70.4%/62.9% switching
([`results/mvopm_round2/FINAL_REPORT.md:136`](../../../results/mvopm_round2/FINAL_REPORT.md#L136),
[`results/mvopm_round2/FINAL_REPORT.md:139`](../../../results/mvopm_round2/FINAL_REPORT.md#L139)).

Independent reconstruction gives only four regime-level mean-error winners across 20 regimes:
K/S1=12, K/K=6, S1/K=1, S2/S2=1. S1/S1 and S3/S3 never minimize regime mean risk. Mean-risk winner
changes occur in 2/8 sample-size and 3/8 proxy-noise contrasts. Individual modal winners also have
frequencies as low as 0.33–0.53 in most cells, which is substantial winner noise. Reusing the same
integer seed across n does not make datasets nested because array shapes change the RNG stream.

**Why it matters.** A candidate winning one noisy replicate is not evidence that it is risk-optimal
in a structural regime. The benchmark does contain real switching among at least four candidates,
but the report's six-candidate/genuine-transition wording overstates it and the preregistered
switching success criterion is not tested at the correct regime-risk level.

**Required fix.** Define oracle switching using expected risk per registered regime, report paired
seed-block uncertainty for risk differences, and distinguish per-replicate winner instability from
regime-level structural switching. Correct the report to “four regime-level mean-risk winners; all
six won at least one noisy replicate.”

### C5. The primary ATE estimand is inconsistent across DGPs

**Evidence.** S1/S2 use the realized sample mean of `tau(X)` as `ate_true`
([`opm/dgp/synthetic_main.py:55`](../../../opm/dgp/synthetic_main.py#L55),
[`opm/dgp/synthetic_main.py:107`](../../../opm/dgp/synthetic_main.py#L107)); bridge-complexity does the
same ([`opm/dgp/bridge_complexity.py:91`](../../../opm/dgp/bridge_complexity.py#L91)). The nonlinear D2
DGP instead fixes `ate_true=2.0`
([`opm/dgp/nonlinear_bridge.py:52`](../../../opm/dgp/nonlinear_bridge.py#L52),
[`opm/dgp/nonlinear_bridge.py:59`](../../../opm/dgp/nonlinear_bridge.py#L59)), even though realized
`mean(tau(X))` varies. Candidate and selector errors use `ds.ate_true` directly
([`opm/eval/oracle.py:23`](../../../opm/eval/oracle.py#L23)). The preregistration does not say whether
the target is population ATE or sample-conditional ATE
([`docs/mvopm_round2/ROUND2_PREREG.md:130`](../ROUND2_PREREG.md#L130)).

**Why it matters.** The pooled selector metric mixes two estimands. The discrepancy is modest at
n=4,000 but non-negligible relative to small oracle denominators. My sample-ATE D2 reconstruction
still shows catastrophic failure, so this does not change the negative verdict; it does change exact
ratios and violates a confirmatory design invariant.

**Required fix.** Freeze one target for every DGP. Prefer sample-conditional ATE if the estimator
averages over the observed X sample, or analytically compute population ATE everywhere. Add a test
that every DGP's `ate_true` obeys the chosen convention.

### C6. Primary pooling, weighting, and decision rules were not fully preregistered

**Evidence.** The preregistration says “pooled MSES” and “most of” several criteria, with “about,”
“material,” and “high” left undefined
([`docs/mvopm_round2/ROUND2_PREREG.md:146`](../ROUND2_PREREG.md#L146)). It never specifies whether the
primary pool is B2–E2, B2–E2+R, or all simulation studies, nor whether studies/regimes are equally
weighted. The report chooses 200 flagship plus 600 R rows
([`results/mvopm_round2/FINAL_REPORT.md:99`](../../../results/mvopm_round2/FINAL_REPORT.md#L99)), so R
gets 75% of the primary descriptive distribution. Primary medians and catastrophic rates are
compared to thresholds without seed-block confidence intervals
([`scripts/analyze_round2.py:152`](../../../scripts/analyze_round2.py#L152)); only paired mean
differences receive a stratified block bootstrap
([`scripts/analyze_round2.py:198`](../../../scripts/analyze_round2.py#L198)).

**Why it matters.** The negative verdict is not marginal, but a positive/mixed outcome would have
left substantial analytic degrees of freedom. Repeated R regimes share 30 scientific seed blocks,
and descriptive 600-row percentages should not be presented as if 600 independent replications.

**Required fix.** For any new confirmation, freeze the exact target population, study/regime weights,
independent resampling units, primary interval procedures, and a deterministic verdict rule. For the
current report, add seed-block intervals for the median-ratio and relative-failure-rate gates and
show study-equal and regime-equal sensitivities.

### C7. RHC confidence intervals are naive post-selection intervals

**Evidence.** A candidate is selected using the same full-sample OOF outcomes through moment tests
and pseudo-outcome variance
([`opm/experiments/mvopm_round2/studies.py:170`](../../../opm/experiments/mvopm_round2/studies.py#L170),
[`opm/experiments/mvopm_round2/studies.py:174`](../../../opm/experiments/mvopm_round2/studies.py#L174)).
Its CI is the ordinary `mean +/- 1.96*sd/sqrt(n)` interval
([`opm/estimator/tau_head.py:115`](../../../opm/estimator/tau_head.py#L115)), with no correction for
candidate selection, overlapping cross-fit nuisance fits, or the 30 split searches. The report
emphasizes that all 30 CIs exclude zero
([`results/mvopm_round2/FINAL_REPORT.md:218`](../../../results/mvopm_round2/FINAL_REPORT.md#L218)) but
does not label them naive/post-selection.

**Why it matters.** Repeated negative point estimates do establish algorithmic directional
stability on this one dataset, but the intervals cannot support post-selection causal significance.
Proxy validity/completeness also remain untestable in RHC.

**Required fix.** Relabel these as unadjusted descriptive Wald intervals. For confirmatory inference,
use an independent selection/estimation split, selective-inference method, or nested repeated
sample-splitting procedure with a justified variance estimator. Do not use “all CIs exclude zero” as
causal evidence.

## P2 / minor-but-required findings

### C8. “Catastrophic” is a misleading name for the preregistered relative event

The code defines catastrophe solely as `selected_error > 1.5*oracle_best_error`
([`opm/experiments/mvopm_round2/core.py:153`](../../../opm/experiments/mvopm_round2/core.py#L153)).
Because the minimum of six realized errors is often near zero, many flagged rows have tiny absolute
errors. In the core/switching pool, oracle-error quantiles are 0.00113 (5%), 0.00984 (25%), and
0.03092 (median); 153/335 flagged MSES rows have absolute error below 0.05. Keep the metric, because
it was preregistered, but rename it “relative >1.5× oracle failure” and always pair it with absolute
error/tail thresholds. The report's D2 failures are genuinely large; the pooled label is not.

### C9. Nested abstention and deployment abstention are conflated outside J2

`nested_mses` declares the entire task abstained if **any** outer fold has no survivor
([`opm/estimator/nested_mses.py:121`](../../../opm/estimator/nested_mses.py#L121),
[`opm/estimator/nested_mses.py:160`](../../../opm/estimator/nested_mses.py#L160)). This compounds
fold-level rejection and differs from the full-sample OOF deployment decision. I2 reports 70% under
the former versus 58% under the latter, yet the final report simply says MSES abstains 70%
([`results/mvopm_round2/FINAL_REPORT.md:197`](../../../results/mvopm_round2/FINAL_REPORT.md#L197)). The
report carefully distinguishes these estimands for J2
([`results/mvopm_round2/FINAL_REPORT.md:207`](../../../results/mvopm_round2/FINAL_REPORT.md#L207)) but
not I2. Report both and prespecify which defines deployment sensitivity/specificity.

### C10. Two preregistered descriptive sensitivity analyses are absent

Holm arm adjustment was promised as a sensitivity
([`docs/mvopm_round2/ROUND2_PREREG.md:68`](../ROUND2_PREREG.md#L68)), and maximum contrast variance
was promised as an efficiency sensitivity
([`docs/mvopm_round2/ROUND2_PREREG.md:78`](../ROUND2_PREREG.md#L78)). Repository search finds no Holm
implementation or report, and `po_var_max` is stored but not used in a sensitivity selector. These do
not alter the primary rule, but their omission must be disclosed as protocol incompleteness or the
analyses must be produced without changing the frozen method.

### C11. Aggregation does not verify that a result file belongs to its manifest row

The aggregator trusts `expected["out_path"]`, then copies expected study/seeds into the output while
accepting `prov["result"]`; it does not compare provenance config, array ID, study, seed, or commit to
the manifest
([`scripts/cluster/aggregate_round2.py:37`](../../../scripts/cluster/aggregate_round2.py#L37),
[`scripts/cluster/aggregate_round2.py:53`](../../../scripts/cluster/aggregate_round2.py#L53)). A
misplaced/cross-copied successful JSON could therefore be silently misattributed. Compact aggregate
checks found no key/commit anomaly, but the audit code should fail closed on exact config/provenance
mismatch and record hashes/package versions/dirty state.

### C12. PEHE diagnostics fit and evaluate the CATE head on the same X rows

Candidate OOF pseudo-outcomes are validly assembled, but `fit_cate_and_ate(ds.X, phi, ...)` fits the
head on all X and PEHE is immediately evaluated on the same X
([`opm/experiments/mvopm_round2/core.py:60`](../../../opm/experiments/mvopm_round2/core.py#L60),
[`opm/experiments/mvopm_round2/core.py:64`](../../../opm/experiments/mvopm_round2/core.py#L64)). The
nested MSES PEHE does the same
([`opm/estimator/nested_mses.py:174`](../../../opm/estimator/nested_mses.py#L174),
[`opm/experiments/mvopm_round2/core.py:205`](../../../opm/experiments/mvopm_round2/core.py#L205)).
These PEHE values are optimistic in-sample secondary diagnostics, not out-of-sample CATE risk. The
final report correctly avoids a CATE headline; it should explicitly label raw PEHE accordingly.

### C13. I2 “library truth” is asserted, not identified

The manifest assigns `library_truth` labels by construction
([`scripts/cluster/manifest_round2.py:115`](../../../scripts/cluster/manifest_round2.py#L115)), and the
analysis calls abstention in either declared inadequate cell a “true rejection”
([`scripts/analyze_round2.py:419`](../../../scripts/analyze_round2.py#L419)). The nonlinear and
bridge-complexity DGPs do not provide exact candidate bridges or an approximation-error threshold;
therefore “sieve1-only is inadequate” is plausible but not population-verified. Call these engineered
stress labels, not ground-truth sensitivity/specificity, unless population moments or exact bridge
approximation bounds are supplied.

## Manuscript claim-to-evidence audit

### `FINAL_REPORT.md`

The numerical MSES failure summary is reproducible and the `NOT_SUPPORTED` headline is appropriate.
Required changes are:

1. Narrow “held-out moments measure bridge error” to **the aligned exact-DGP perturbation family**;
   A2 uses one DGP and one prespecified corruption direction. The current broad formulation at
   [`FINAL_REPORT.md:243`](../../../results/mvopm_round2/FINAL_REPORT.md#L243) is stronger than the
   evidence.
2. Replace six-candidate structural switching and per-seed transition claims as described in C4.
3. Relabel “catastrophic” as the preregistered relative failure event and add absolute-tail rates.
4. Label RHC CIs naive/post-selection and distinguish full-deployment versus nested abstention in I2.
5. Disclose missing Holm/max-variance sensitivities and the inconsistent D2 target convention.
6. Retain the negative verdict: the sample-ATE D2 sensitivity and variance-only comparison still
   reject near-oracle MSES performance.

### `ROUND2_PREREG.md`

The preregistration did freeze seeds, methods, and broad goals before final execution. It was not a
fully determinate confirmatory statistical analysis plan because pooling/weights, uncertainty for
the principal gates, and the “most of” verdict rule were not fixed (C6). It also promised but did not
deliver two sensitivities (C10). A future protocol needs an explicit estimand table by DGP and a
calibrated-test theorem/simulation plan.

### `EXACT_DGP_BIAS_DERIVATION.md`

The aligned bias expansion and sign are correct. The limitation is external to the derivation: it
proves q remains **positive**, not that the kernel solver's `q>=1` parameterization is valid. Add this
distinction and avoid using A2 as evidence that the learned kernel q class is well specified.

### `CATE_SELECTION_ANALYSIS.md`

This is the most appropriately bounded current manuscript. Its independence/rate caveats at lines
29–38 are correct. Two additions are needed: the implemented PEHE is in-sample at the Stage-2-head
level (C12), and `relative_cate_risk`'s simple row-wise SE
([`opm/validation/cate_relative.py:24`](../../../opm/validation/cate_relative.py#L24)) likewise does
not account for repeated/overlapping nuisance fits. No CATE inference claim should be made.

### `LITERATURE_BOUNDARY.md`

The checked descriptions of P-learner, Causal Q-Aggregation/relative-error HTE evaluation, and the
Deconditional GP work are directionally consistent with the linked primary sources, and the ban on
“first” claims is appropriate. However, the document is dated before the final result and still gives
a conditional success narrative at lines 43–47. Add a post-result note that the condition failed and
that no positive MSES contribution claim is authorized. It is a boundary memo, not an exhaustive
2026 priority review.

### `技术全志_1_数学与实现.md`

This document predates Round 2 and must be bannered as a frozen Round-1/Pilot record. It conflicts
with the current CATE boundary by asserting that orthogonality alone makes ATE/CATE CIs tend to
nominal coverage
([`技术全志_1_数学与实现.md:71`](../../技术全志_1_数学与实现.md#L71),
[`技术全志_1_数学与实现.md:140`](../../技术全志_1_数学与实现.md#L140)); nuisance rates, smoothing bias,
fold dependence, and variance estimation are not proved. It also documents obsolete seed logic and
old nested code rather than MSES Round 2. The historical record may remain, but readers must not
mistake it for the current method specification.

### `技术全志_2_实验与结果.md`

This is explicitly stale after the final run: it says Round-2 final seeds are “100000+ (not run)”
([`技术全志_2_实验与结果.md:188`](../../技术全志_2_实验与结果.md#L188)) and ends with the exploratory
bias-variance result rather than the failed confirmation
([`技术全志_2_实验与结果.md:260`](../../技术全志_2_实验与结果.md#L260)). Its title and opening claim to
contain every experiment/full result are now false
([`技术全志_2_实验与结果.md:1`](../../技术全志_2_实验与结果.md#L1),
[`技术全志_2_实验与结果.md:3`](../../技术全志_2_实验与结果.md#L3)). Either append a Round-2 final volume
with the negative outcome or rename this “through Pilot 1, frozen 2026-08-11.”

### `两天历程_OPM到MVOPM.md`

This is a useful historical narrative, but its “next step” is now completed and failed. It still says
the bias+variance direction may achieve near-oracle selection and recommends a future confirmation
([`两天历程_OPM到MVOPM.md:238`](../../两天历程_OPM到MVOPM.md#L238),
[`两天历程_OPM到MVOPM.md:252`](../../两天历程_OPM到MVOPM.md#L252)). Add an epilogue pointing to
Round 2, recording `NOT_SUPPORTED`, the D2 failure, and the narrower aligned-mechanism result. Also
correct the obsolete global-RNG seed code excerpt at lines 134–139; current code uses solver-name
offsets and `torch.random.fork_rng`.

## Test-suite gaps

The 27 invariant tests pass, but the suite should additionally cover:

- null size/power of the complete fitted-candidate moment-testing pipeline, not only deterministic
  p-values;
- exact-q representability by every candidate solver;
- nonnegativity/lower-boundedness of the optimized kernel loss or an expected-failure test documenting
  the current U-statistic pathology;
- a single, declared ATE target convention across every DGP;
- exact manifest/provenance/config equality during aggregation;
- both h **and q** in candidate-order invariance (the current test checks h only at
  [`tests/test_mvopm_invariants.py:109`](../../../tests/test_mvopm_invariants.py#L109));
- out-of-sample Stage-2 CATE-head evaluation;
- Holm and maximum-variance sensitivity reconstruction;
- deployment versus nested-any-fold abstention semantics.

## Recommended disposition

1. **Preserve** the current raw corpus and negative result; do not overwrite or retune it.
2. **Correct the current report** for switching, relative-failure terminology, RHC intervals, and
   protocol omissions. These are claim corrections, not scientific reruns.
3. **Version the three Chinese manuscripts** as pre-Round-2 historical records or add a clear Round-2
   epilogue.
4. Treat q parameterization and moment-test calibration as new-method work requiring fresh development
   and, if pursued, a new preregistration/new seeds.
5. The strongest defensible present result is: **within the exact aligned perturbation study, the
   prespecified moment discrepancies track perturbation magnitude and predicted second-order bias;
   the tested MSES implementation fails as a general ATE selector.**
