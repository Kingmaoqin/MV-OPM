# Secondary CATE-selection analysis

## Why ATE variance is not PEHE risk

MSES's primary efficiency term is the variance of an OOF arm-contrast pseudo-outcome. Dividing it
by n estimates uncertainty of a marginal ATE mean. PEHE instead measures an L2 error of a
conditional function over X. A low-variance sample mean can coexist with poor subgroup fit,
oversmoothing, local bias, or cancellation of positive and negative conditional errors. Therefore
ATE influence-function variance is target-aligned for ATE selection but is not automatically a
valid proxy for CATE/PEHE risk.

## Relative-risk identity

For two fixed CATE candidates a and b,

`R(a)-R(b) = E[tau_a(X)^2-tau_b(X)^2-2(tau_a(X)-tau_b(X))*tau(X)]`.

Suppose an evaluation-fold proximal signal D satisfies `E[D|X, training data]=tau(X)` and both
candidate predictions on that fold are measurable functions of X and training data. Then replacing
tau by D leaves the conditional expectation unchanged. The sample average of

`tau_a^2-tau_b^2-2(tau_a-tau_b)D`

is consequently unbiased for the conditional risk difference. The implementation is
`opm.validation.cate_relative.relative_cate_risk`.

## Conditions that cannot be skipped

The identity alone is not enough. A defensible evaluator requires:

1. evaluation observations excluded from every candidate and proximal-signal nuisance fit;
2. a proximal signal whose conditional mean is the target CATE under the required identification,
   completeness, positivity, and bridge conditions;
3. sufficiently small nuisance remainder, with cross-products controlled under a doubly robust or
   orthogonal construction;
4. no reuse of evaluation outcomes to tune the candidates being compared;
5. finite second moments and a variance estimate that respects repeated cross-fitting;
6. for multiple treatments, correct arm ordering and a prespecified contrast aggregation.

If `E[D|X]=tau(X)+b_D(X)`, evaluator bias is exactly
`-2 E[(tau_a(X)-tau_b(X)) b_D(X)]`. Thus moment screening may reduce a relevant nuisance failure,
but passing a finite set of moments does not prove this conditional bias is zero.

## Relationship to existing CATE evaluation

[Causal Q-Aggregation](https://arxiv.org/abs/2310.16945) uses a doubly robust loss to obtain CATE
model-selection/aggregation regret guarantees under standard causal settings, while the newer
[relative-error evaluation framework](https://arxiv.org/abs/2510.16419) explicitly derives
nuisance conditions for robust pairwise HTE comparison. These make clear that a pseudo-label
substitution must carry nuisance-rate and independence arguments; algebra alone is insufficient.

For proximal CATE, the [P-learner](https://openreview.net/forum?id=Fq03w1f6hy) supplies the closest
target-specific identification/learning foundation. A Round-2 proximal relative-risk evaluator is
mathematically plausible under the conditions above, but this repository currently provides only
the pairwise evaluation functional and simulation checks—not a proved proximal nuisance-rate
theorem.

## Confirmatory boundary

The CATE evaluator is therefore **secondary and experimental**. Round 2 may report whether its
pairwise signs agree with oracle PEHE differences in simulations, but the confirmatory claim
remains ATE-level estimator selection plus bridge diagnostics unless the methods review and final
data validate the stronger conditions. No CATE-selector headline is authorized by this analysis.
