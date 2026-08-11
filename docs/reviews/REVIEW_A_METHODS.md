# Reviewer A — Statistical / Methodological (MV-OPM Phase 2)

**Reviewer-agent note:** independent reviewer-agent spawning is unavailable in this session (an
earlier attempt hit the provider session limit). Per the prompt's fallback clause, this is a
strictly isolated review pass focused only on methodology; engineering is covered separately by
Reviewer B; the two are cross-reconciled in `CROSS_RECONCILIATION.md`.

Scope audited: `opm/dgp/finite_proxy_exact.py`, `opm/validation/moments.py`, `opm/validation/selector.py`,
`opm/estimator/nested_crossfit.py`, `opm/experiments/mvopm/{core,studies,scenarios}.py`,
`tests/test_mvopm_invariants.py`.

## Findings

- **A-1 [CRITICAL → RESOLVED, design].** *Which causal-reliability target should the moment
  diagnostics predict?* Theorem 2 gives the pseudo-outcome/ATE **bias** ≈ ‖δ_h‖·‖δ_q‖. The
  identifying moments measure **bias** (E[R|Z]=0), NOT variance. Therefore the product score is
  theoretically aligned with **ATE error**, not PEHE (which mixes bias and variance). A single
  pilot row confirmed the failure mode: an overfit `sieve3` that is moment-adequate out-of-sample
  but high-variance was ranked *best* by the product score yet had catastrophic PEHE.
  **Resolution (frozen BEFORE the pilot aggregate, based on theory, not on results):** the
  **primary causal-reliability target is ATE error**; PEHE is reported as a secondary target and
  as a stress test of the variance blind spot. The primary *score* remains the **product** (§6,
  unchanged). Both targets are computed for every row (`pehe::*` and `ate::*`). This is a
  prespecified analysis choice, not a post-hoc score change.
- **A-2 [MAJOR → RESOLVED].** Residual normalization. `moments.py` normalizes R, S by within-arm
  sd before forming discrepancies, so h- and q-sides are dimensionless and comparable and scale
  does not trivially decide the ranking (prompt §5). Documented in the module docstring.
  *Consequence made explicit:* normalization removes residual-variance magnitude, which is why the
  diagnostic tracks bias, not variance — consistent with A-1. Kept as prespecified.
- **A-3 [OK].** Exact-bridge DGP verified numerically: E[Y−h_t|V,T=t]≈0, E[1{T=t}q_t|W]≈1, STAR
  ATE=2.01 at n=2e5. This is a genuine `q_true` (unlike old E2), so Study A is a clean product
  mechanism test.
- **A-4 [OK].** Nested cross-fitting: inner folds strictly within OUTER_TRAIN; selection uses only
  inner-validation moments; selected candidate refit on full OUTER_TRAIN; OUTER_TEST only enters
  the outer-OOF pseudo-outcome. Index audit asserted by `test_nested_split_audit_clean`.
- **A-5 [OK].** Oracle isolation: the validator/selector/candidate packages do not import
  `opm.eval.oracle` (asserted by `test_selection_packages_do_not_import_oracle`); the
  `ObservedDatasetView` raises on every truth attribute, incl. smuggling through `meta`.
- **A-6 [OK].** Prespecified scores {product(primary), h, q, sum, max} + competitors
  {fixed_kernel, fixed_sieve1, random, ess, qbalance, oracle}. No post-hoc score invented.
- **A-7 [NOTE].** Abstention (Study I) uses `min_total_D` as a falsification signal, reported with
  falsification language (not "proves proxies valid"). Detection threshold to be calibrated from
  valid-proxy runs, not tuned to pass.
- **A-8 [NOTE].** Ranking metrics use Spearman/Kendall of score vs causal error across candidates;
  the experimental unit is the seed. Paired seeds across selectors.

## Verdict
No unresolved CRITICAL/MAJOR. A-1 and A-2 resolved by freezing **ATE error as the primary target**
(theory-aligned) with PEHE as a documented secondary/stress metric, product unchanged as primary
score. **PASS to development pilot.**
