# Round-2 literature boundary

**Audit date:** 2026-08-12. This boundary is intentionally narrow and makes no priority claim.

## What already exists

* **Proximal CATE estimation.** The [P-learner](https://openreview.net/forum?id=Fq03w1f6hy)
  develops a two-stage CATE learner under proxy-based hidden-confounding assumptions and gives an
  oracle-style error result for kernel regression. MV-OPM is not the first proximal CATE method.

* **Generic CATE selection and aggregation.** [Causal Q-Aggregation](https://arxiv.org/abs/2310.16945)
  studies doubly robust CATE model selection/ensembling and optimal oracle regret, including an IV
  extension. The [relative-error HTE evaluator](https://arxiv.org/abs/2510.16419) derives conditions
  under which pairwise CATE risk differences can be estimated robustly. Round 2 must not claim the
  first oracle-free CATE evaluator or generic CATE selector.

* **Uncertainty-aware proximal learning.** The 2026
  [Deconditional Gaussian Process framework](https://arxiv.org/abs/2603.02159) supplies posterior
  uncertainty and marginal-likelihood model selection for IV/proximal learning. MSES uses frequentist
  OOF pseudo-outcome variance for target-aligned efficiency after a moment screen; that is a different
  construction, not the first uncertainty-aware proximal method.

* **Decision-aware proximal work.** [Optimal Treatment Regimes for Proximal Causal Learning](https://openreview.net/forum?id=zGdH4tKtOW)
  identifies and estimates individualized regimes from outcome and treatment bridges. This is a
  policy-value objective, whereas primary MSES compares estimators for ATE error. No first
  decision-aware-proximal claim is available.

* **Flexible/neural doubly robust bridge learning.** [NMMR](https://openreview.net/forum?id=fRWwcgfXXZ)
  learns proximal bridges with neural maximum-moment restrictions. More recent
  [neural doubly robust proximal estimation](https://openreview.net/pdf/6aebf4166606cb3fb6630ca5fe3cf9e7bab8053d.pdf)
  and [density-ratio-free doubly robust proxy learning](https://openreview.net/forum?id=a9HOg4f9Gh)
  expand the bridge-estimation toolbox. MV-OPM's candidate library is an evaluation test bed, not a
  claim to a novel bridge solver.

## Defensible intended contribution, conditional on results

The distinct question is whether a **model-agnostic two-stage validation framework** can use common,
held-out proximal identifying-moment tests to screen structural bridge inadequacy and then use OOF
pseudo-outcome uncertainty to choose efficiently among heterogeneous bridge solvers. The key
empirical comparison is against moment-only selection, variance-only selection, fixed solvers, and
oracle switching across prespecified structural regimes.

If the confirmatory evidence succeeds, the defensible story is:

> Identifying moments diagnose structural bridge adequacy but omit finite-sample uncertainty;
> separating moment screening from target-aligned efficiency comparison improves proximal
> estimator selection across heterogeneous bridge families.

This is narrower than “first proximal CATE method,” “first selector under hidden confounding,” or
“first uncertainty-aware proximal model selection.” The word “first” is prohibited without a
separate exhaustive priority review.
