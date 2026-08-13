# Reviewer A2 — methods/statistics post-run audit

**Independent final verdict for the proposed MSES thesis: NOT_SUPPORTED.** The corrected A2
experiment establishes the intended aligned product-bias mechanism, and the switching benchmark
shows a real but modest average advantage over the best fixed solver. Those valuable results do
not establish the confirmatory MSES reliability thesis: the two headline gates fail, pooled mean
error is harmed by a catastrophic nonlinear failure mode, and screening removes the useful
low-error candidates in precisely that setting. The broader validation research program remains
promising, but this result is not ICLR-ready as a near-oracle or generally reliable selector.

I inspected the frozen preregistration and amendment, exact-DGP derivation, observed-only
convergence audit, final aggregation audit, reconstructed task/candidate data, all final analysis
tables and evidence summary, representative nested raw artifacts, and the screening/nested
evaluation code. The final aggregation is complete (6,820/6,820 tasks, zero missing and zero
failed); engineering-level provenance is reviewed separately by Reviewer B2.

## Preregistered criteria

The primary criteria are mostly missed.

| Criterion | Confirmatory result | Assessment |
|---|---:|---|
| pooled MSES median ATE oracle ratio about <=1.10 | 1.325 in B2–E2+R; 1.392 in B2–E2 | **FAIL** |
| catastrophic rate <=.20 | 0.422 among 793 evaluable B2–E2+R tasks; 0.446 among 193 evaluable B2–E2 tasks | **FAIL** |
| material improvement over fixed kernel | R: mean ATE error 0.0785 vs 0.0912 and median ratio 1.304 vs 1.412; pooled B2–E2+R: MSES mean error 0.1874 vs 0.1024 | mixed; **not a pooled success** |
| improvement over fixed sieve1 where sieve1 is not oracle | median row gain 0.0104 and MSES wins 67.9% of rows, but mean gain -0.0438 with scientific-seed-block 95% CI [-0.0883,-0.0071] | **FAIL** |
| high oracle survival | 0.861 pooled and 0.924 in R, but only 0.115 in nonlinear D2 | mixed; one load-bearing collapse |
| genuine oracle switching | all six candidates win; modal oracle is not sieve1 in all 20 R regimes; registered sample/noise transitions change 66.7% | **PASS** |
| aligned product-bias mechanism | signed-bias slope 1.023, intercept -0.00034, R2 0.811; single-side cell means near zero | **PASS** |
| persistence in nonlinear/semi-synthetic settings | E2 is favorable; D2 is catastrophic | **FAIL** |

The all-simulation summary does not rescue the thesis: MSES median ratio is 1.356 and catastrophic
rate is 0.428 among 1,748 evaluable tasks, with 42 additional abstentions/missing selector errors.

## Answers to the eight required questions

### 1. Did corrected A2 establish the intended product-bias mechanism?

**Yes, strongly, at the ATE mechanism level.** The exact prediction is
`-r_h r_q E[tanh(X1)^2]`, with the independently computed multiplier 0.39429449. Across 5,000
registered A2 rows, predicted versus incremental signed bias has Pearson 0.900, Spearman 0.756,
slope 1.023, intercept -0.000342, and R2 0.811. The largest absolute single-side cell mean is
0.000812, whereas the largest both-sides cell mean is 0.06464. At `(r_h,r_q)=(.4,.4)`, the mean
incremental bias is -0.06464 versus -0.06309 predicted. The single-side means are no more than
about 1.1 cell standard errors from zero. Positivity was preserved without clipping (minimum
multiplier approximately 0.600).

The observed diagnostics also move in the intended directions: R2 is 0.519 for `D_h` versus true
h error, 0.515 for `D_q` versus true q error, and 0.457 for moment product versus true error
product. Because the same 200 seed labels recur over the 25 corruption cells, the row-level
correlations should be treated as dose-response descriptions, not 5,000 independent units. The
cell means and paired corruption construction nevertheless make the mechanism conclusion robust.

### 2. Did kernel undertraining explain Pilot-1 failures?

**It explained part of the old kernel weakness, but not the Pilot's central selection failure.**
The observed-only audit found no plateau by 180 epochs in any of S1, S2, nonlinear, or HAMD and
therefore correctly froze 300 epochs. This establishes that the Pilot's 45-epoch kernel was
undertrained. Round 2 kernel errors are dramatically smaller than Pilot values in S1/S2/nonlinear,
but those comparisons are not an isolated epoch ablation: B2–D2 also use n=4,000 rather than the
Pilot's n=1,500 and other evaluation details changed. HAMD kernel performance did not materially
improve.

Most importantly, the Pilot diagnosis that moments alone are variance-blind survives. In D2,
useful low-error candidates are screened out while high-variance sieve2/3 candidates survive and
are selected. Thus undertraining cannot explain the Round-2 reliability failure and was not the
sole explanation of the Pilot result.

### 3. Does MSES outperform moment-only selection?

**Not generally.** Relative to the registered product selector, MSES has better medians and wins
55.4% of 793 evaluable B2–E2+R task pairs, but its mean paired error reduction is **-0.0688**
(scientific-seed-block 95% CI [-0.1085,-0.0344]; negative favors product) because MSES
catastrophically fails in D2. Within R alone, MSES clearly
improves product (mean error 0.0785 vs 0.1251; median oracle ratio 1.304 vs 1.960; catastrophic
rate 0.415 vs 0.582). Within flagship B2–E2, product has much lower mean error (0.0987 vs 0.5259),
again because of D2. The correct conclusion is that the uncertainty term helps in the switching
matrix and E2, but the hard screen can turn it into a much worse selector when it excludes the
stable candidates.

MSES also does not generally beat variance-only: pooled mean error is 0.1874 versus 0.0818,
catastrophic rate 0.422 versus 0.394, and only 23.3% of evaluable pairs favor MSES (56.1% are exact
ties). The screening stage has not demonstrated incremental reliability over the efficiency
criterion by itself.

### 4. Does MSES improve over the best fixed solver when the oracle genuinely changes?

**Yes on average in the dedicated R benchmark, but modestly, heterogeneously, and far from
near-oracle.** The best fixed comparator in R is fixed kernel. MSES lowers mean ATE error from
0.09125 to 0.07850 (about 14%), lowers median oracle ratio from 1.412 to 1.304, and lowers the
catastrophic rate from 0.472 to 0.415. It has lower error in 54.7% of rows and ties in 17.8%.
The mean paired gain is 0.01274 with a scientific-seed-block bootstrap 95% CI
[0.00933,0.01546]; each resampled seed block retains all 20 registered R regimes.

The improvement is concentrated in Q regimes and is not uniform: several L/M/N cells are tied or
slightly worse, and the MSES exact-oracle match rate is only 0.306. This supports an adaptive
advantage over one fixed solver in the designed switching benchmark, not a general near-oracle
claim. Once D2 is pooled in, the extreme nonlinear losses reverse the mean comparison with fixed
kernel.

### 5. What is the catastrophic failure frequency?

**Unacceptably high.** Among evaluable tasks it is 42.2% (335/793) for B2–E2+R, 44.6% (86/193)
for B2–E2, and 42.8% (749/1,748) across all simulation-selection studies. Abstentions are excluded
from those denominators. D2 is decisive: 40/43 evaluable runs are catastrophic (93.0%), with seven
additional abstentions; treating either outcome as failure gives 47/50 (94%). D2 MSES mean ATE
error is 2.061, median 1.203, maximum 11.317, and median oracle ratio 52.7.

### 6. Is screening statistically meaningful?

**It is a formal finite-moment compatibility test, but it is not yet a reliable adequacy screen
for estimator selection.** Its positive evidence is limited but real: easy adequate L has zero
false abstentions, while the deliberately overregularized N library is rejected/abstained from in
70% of runs. Its limitations are equally clear: the constrained inadequate sieve1-only N library
is never rejected, and in nonlinear D2 the global causal-error oracle survives only 11.5% of outer
folds. There, average survival/selection frequencies are 0.855/0.580 for sieve2 and 0.745/0.090 for
sieve3 despite enormous pseudo-outcome variances, while useful kernel/mixed/sieve1 candidates are
mostly screened out. A finite RFF union-null pass establishes compatibility with the tested
moments, not proxy validity, bridge correctness, or low causal estimation error.

### 7. What can be claimed about abstention?

Only the narrow preregistered claim is supported: **MSES can abstain for some detectably inadequate
candidate libraries.** It abstains in 70% of the severe-overregularization cell and 0% of the easy
adequate cell. It has no sensitivity to the constrained sieve1-only misspecification cell (0/50),
so it is not a general detector of library inadequacy. Moreover, D2 shows that abstention does not
reliably prevent catastrophic deployment: it abstains only 7/50 times while 40/43 remaining
evaluations are catastrophic. No safety-guarantee or proxy-validity claim is warranted.

### 8. Is the evidence CATE-specific or only ATE?

**Only ATE-specific.** The headline selector target, screening-performance comparison, switching
benchmark, and success gates use ATE error. The implemented pairwise CATE relative-risk functional
is secondary/experimental, and this final analysis does not validate its conditional-unbiasedness
and nuisance-rate requirements or report a confirmatory CATE-selection experiment. PEHE fields
from fitted CATE heads do not turn an ATE-variance selector into a CATE-valid selector. The work
must not be called proximal CATE model selection on this evidence.

## Findings requiring action

### MAJOR-1 — the reliability/near-oracle claim is contradicted by the nonlinear result

D2 is not a mild miss: screening has selected the high-variance candidates it was designed to
avoid after removing the useful candidates. The final report must headline this failure, not hide
it behind pooled medians, R-regime counts, or abstention-excluded denominators. Any ICLR-facing
near-oracle, reliable, or safety framing should be removed.

### MAJOR-2 — RESOLVED: paired inference now uses scientific-seed blocks

The first post-run table bootstrapped and applied Wilcoxon tests to all 793 task rows as if
independent, even though R contributes 600 rows from 30 scientific seeds reused over 20 regimes.
The reporting analysis has now been corrected without changing any model, seed, selector
decision, or raw result. It forms paired-error blocks by `(study, scientific_seed)`, resamples the
same number of whole blocks separately within each study, and retains every regime row belonging
to a sampled seed. Row-level Wilcoxon p-values were removed; row medians and win fractions are
explicitly descriptive.

I verified the implementation and rebuilt the three load-bearing contrasts independently with
100,000 bootstrap replicates under a separate RNG seed. The stored 5,000-replicate intervals and
my independent intervals agree to ordinary Monte Carlo precision:

| Scope/comparator | Rows / seed blocks | Stored mean gain and 95% block CI | Independent 100k CI |
|---|---:|---:|---:|
| B2–E2+R vs product | 793 / 223 | -0.06882 [-0.10853,-0.03444] | [-0.10831,-0.03467] |
| R vs fixed kernel | 600 / 30 | 0.01274 [0.00933,0.01546] | [0.00939,0.01550] |
| oracle-not-sieve1 vs fixed sieve1 | 670 / 185 | -0.04384 [-0.08829,-0.00712] | [-0.08790,-0.00676] |

The generated table and `EVIDENCE_SUMMARY.json` contain core, flagship, R-only, and
oracle-not-sieve1 scopes with the correct row and block counts. This resolves the reporting flaw
and strengthens, rather than weakens, the negative pooled conclusion. Future protocols should
still prespecify regime weights and the clustered estimand explicitly.

### MINOR-1 — distinguish evaluable catastrophic rates from total failure rates

All reported catastrophic proportions omit MSES abstentions. Always report both numerator and
evaluable denominator and separately report abstention. In D2, 93.0% catastrophic among
evaluable runs and 14% abstention jointly imply 94% catastrophic-or-abstain over all registered
runs.

### MINOR-2 — avoid interpreting oracle survival/top-1 as fold-specific oracle recovery

The nested fold choices are compared with a global-OOF causal-error oracle. This is a reasonable
fixed reference for descriptive evaluation, but it is not an oracle estimated independently
inside each outer-training population. Label the rate as survival/match to the global candidate
oracle.

### MINOR-3 — post-hoc bias-variance scores remain exploratory

`biasvar_mult` and `biasvar_add` were generated from the Round-1 finding and are correctly labeled
`EXPLORATORY_POSTHOC_FROM_ROUND1`. Their favorable summaries cannot be promoted to a Round-2
confirmatory method, used to replace MSES, or used to tune a third selector on these seeds.

## Claim boundary and recommendation

The strongest defensible contribution is narrower than the thesis: (i) a clean experimental
confirmation of the proximal product-bias mechanism; (ii) a reproducible formal moment-screening
and nested estimator-selection framework; (iii) evidence that combining screening with
uncertainty can outperform a fixed solver in a purposely heterogeneous switching benchmark; and
(iv) an unusually informative negative result showing that hard moment screening can discard
causally useful estimators and expose high-variance survivors.

That package is scientifically useful, but the preregistered proposed-method thesis receives the
final verdict **NOT_SUPPORTED**. The next confirmatory design needs a screen/selection rule that
prevents the D2 failure without using these final outcomes for tuning, followed by a fresh seed
matrix. An honest negative-results or validation/falsification paper may be possible sooner, but
it would be a different claim from reliable near-oracle MSES.
