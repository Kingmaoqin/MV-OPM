# MV-OPM Round 2 confirmatory preregistration

**Frozen protocol version:** 1.0  
**Frozen before confirmatory results:** yes  
**Protected starting commit:** `6122aeb9a145c5835939dc292161e0c5715434bf`  
**Round-2 branch:** `mvopm-round2-confirmatory`

This document fixes the confirmatory analysis before any Round-2 final seed is run. Existing
Pilot-1 evidence under `results/mvopm/` is immutable. Round-2 evidence is written only beneath
`results/mvopm_round2/`. Development and convergence-audit results are labeled as such and are
not pooled with confirmatory results.

## Confirmatory thesis and method

The thesis is that reliable proximal estimator selection needs two distinct signals:
identifying-moment adequacy for structural bridge bias and finite-sample uncertainty for
statistical efficiency.

The **primary method is Moment-Screened Efficiency Selection (MSES)**:

1. Screen candidate bridge pairs using formal studentized identifying-moment tests.
2. Among survivors, select the candidate with the smallest OOF proximal pseudo-outcome
   contrast variance.
3. If no candidate survives, return `ABSTAIN_LIBRARY_INADEQUATE`.

The Round-1 `biasvar_mult` and `biasvar_add` scores are retained only as
`EXPLORATORY_POSTHOC_FROM_ROUND1`; neither is MSES or a confirmatory proposed method.

## Candidate library and training

The fixed global library is unchanged:

`kernel_kernel`, `sieve1_sieve1`, `sieve2_sieve2`, `sieve3_sieve3`,
`kernel_sieve1`, and `sieve1_kernel`.

On high-dimensional RHC/HAMD or p=40 stress configurations where polynomial feature growth is infeasible, the
prespecified feasible subset is `kernel_kernel`, `sieve1_sieve1`, `kernel_sieve1`, and
`sieve1_kernel`; the exclusion and feature dimension are recorded in each raw row.

Before the convergence audit, the maximum allowed kernel budget is **300 epochs**, patience 45,
evaluation every 3 epochs, and batch size 1024. The development audit evaluates caps
45/90/180/300 and freezes the smallest budget satisfying the prespecified observed-data plateau
and prediction-stability rule; if no plateau occurs before 300, it freezes 300. Oracle PEHE/ATE
error cannot alter that choice. The chosen value is recorded in the audit and an explicit
pre-final amendment below before final execution. The secondary CATE head uses 300 maximum epochs
and patience 30, rather than the shortened Pilot-1 head.

## Cross-fitting and moment tests

Selection uses four outer folds and three inner folds. Within each outer-training set, every
candidate receives h and q predictions for every observation from an inner-fold model that did
not train on that observation. One candidate-independent instrument/RFF bank is then fit on the
entire outer-training validation population and applied to all candidate OOF residuals. Outer-test
observations cannot affect scalers, bandwidths, RFF features, tests, selection, or refitting.

For candidate m and arm k:

* outcome residual: `R_ik = 1{T_i=k}(Y_i-h_mk(W_i,X_i))`;
* treatment residual: `S_ik = 1{T_i=k}q_mk(V_i,X_i)-1`.

The primary formal test is the studentized RFF maximum statistic over 200 shared cosine RFF
moments. Moments with nonfinite or effectively zero standard error are excluded and counted.
The null distribution uses a centered Gaussian multiplier bootstrap. Development runs use 499
draws; final runs use 999. The finite-sample p-value is
`(1 + number(T_boot >= T_obs))/(B + 1)`. Bootstrap seeds are explicit manifest fields and are
separate from DGP, split, model-fit, and RFF-bank seeds. Numerical warnings are retained.

The per-arm double-robust union-null p-value is fixed as
`p_DR[m,k] = max(p_h[m,k], p_q[m,k])`. An arm fails only when both bridge-side tests reject. With
K required arms, the primary Bonferroni threshold is `0.05/K`; a global candidate survives only
if `min_k p_DR[m,k] >= 0.05/K`. Holm adjustment is descriptive sensitivity only.

Normalized RFF-L2, normalized RFF-max, and MMR discrepancies remain descriptive diagnostics and
do not replace the primary test.

## Efficiency criterion and selector outputs

For every surviving candidate and non-reference contrast k versus 0, form the inner-OOF signal
`psi_i,m,k = phi_i,m,k - phi_i,m,0`. The primary efficiency criterion is
`V_m = mean_{k=1,...,K-1} Var(psi_m,k, ddof=1)`. `max_k Var(psi_m,k)` is sensitivity only.
The selected candidate minimizes `V_m`; ties are resolved lexicographically to make the result
candidate-order invariant. Estimated ATE SE is `sqrt(V_m,k/n)` and CI width is `3.92*SE`.

## Confirmatory comparators

MSES is compared with oracle, fixed sieve1, fixed kernel, original product, h-only, q-only, sum,
max, variance-only, and random. `biasvar_mult` and `biasvar_add` are included only with the label
`EXPLORATORY_POSTHOC_FROM_ROUND1`.

## Structural regime matrix

The core matrix is frozen at 16 regimes: family `{L,Q,N,M}` × n `{1500,6000}` × proxy noise
`{low,high}`, with `c_U=1`. The stress matrix has family `{L,Q,N,M}`, n=3000, p=40, and low proxy
noise. Constants and equations are fixed in `opm/dgp/bridge_complexity.py` before final execution.
No regime is changed based on its oracle winner.

## Studies and seed manifests

Every seed is explicitly stored in `configs/seeds/mvopm_round2_final.json`; no scientific seed is
derived from an array index or process ID. The disjoint ranges are:

* A2 aligned mechanism: 200000–200199 per 25 corruption cells;
* B2 S1: 201000–201049;
* C2 S2: 202000–202049;
* D2 nonlinear E9: 203000–203049;
* E2 HAMD: 204000–204049;
* R regime matrix: 205000–205029 per regime;
* F2 confounding: 206000–206029 per level;
* G2 proxy corruption: 207000–207029 per kind/severity cell;
* H2 sample-size scaling: 208000–208029 per family/n cell;
* I2 library inadequacy: 209000–209049 per cell;
* J2 RHC split seeds: 210000–210029.

Development-only convergence seeds are 220000–220007 and are never pooled with final evidence.
Infrastructure reruns keep the same scientific seed. Algorithmic failures remain in the raw
record and are not replaced.

## Fixed study configurations

Study A2 uses `g(X)=tanh(X[:,0])`, keeps both reference-arm bridges correct, and perturbs only the
treated arm with `r_h,r_q in {0,.05,.10,.20,.40}`. q is not clipped after perturbation; positivity
is checked. There are 200 repetitions per cell at n=4000.

B2–E2 use 50 seeds. R uses 20 regimes × 30 seeds. F2 uses `c_U={0,.5,1,2}` × 30 seeds. G2 uses
`{additive,missing,shift,heavy_tail}` × `{0,.1,.2,.3}` × 30 seeds. H2 uses n
`{1000,2000,4000,8000}` for families L and N × 30 seeds. I2 uses valid/easy and prespecified
library-misspecification cells × 50 seeds. J2 uses 30 repeated split seeds; these are algorithmic
stability analyses, not independent datasets.

## Targets and metrics

The primary simulation target is absolute ATE error, averaged across non-reference contrasts for
multi-treatment data. PEHE is secondary. For selector s:

* oracle ratio: `error(s)/error(oracle_best)` (with a documented numerical floor only when both
  errors are essentially zero);
* regret: `error(s)-error(oracle_best)`;
* catastrophic selection: `error(s) > 1.5*error(oracle_best)`;
* top-1, top-2, Spearman, and Kendall tau where a scalar ranking exists;
* MSES survivor-set size, oracle survival, selected candidate, top-1, and top-2;
* false/true rejection and oracle-candidate rejection where bridge adequacy truth is known.

Every candidate table retains D_h, D_q, p_h, p_q, p_DR, screen status, OOF variance, CI width,
PEHE, ATE error, ESS, max q, and every selection/oracle/fixed-baseline indicator.

## Success criteria and claim boundary

Strong primary evidence requires most of: pooled MSES median ATE oracle ratio about ≤1.10;
catastrophic rate ≤.20; material improvement over fixed kernel; improvement over fixed sieve1 in
regimes where sieve1 is not oracle; high oracle survival; genuine oracle switching; confirmation
of the aligned product-bias mechanism; and persistence in nonlinear/semi-synthetic settings.
These criteria do not stop runs and do not authorize post-hoc changes.

MSES tests empirical compatibility of the available bridge models with tested identifying
moments. It does **not** establish causal proxy validity. The allowed abstention claim is only that
MSES can abstain when the available candidate library is detectably inadequate.

The confirmatory headline is ATE-level estimator selection. Any CATE relative-error evaluator is
secondary and can be called valid only after an independent mathematical audit of cross-fitting,
conditional unbiasedness, nuisance error, and independence requirements.

## Pre-final amendments

**2026-08-12, before any final task:** the observed-data audit in
`results/mvopm_round2/development/convergence_audit/CONVERGENCE_AUDIT.md` found no plateau before
300 epochs. Median held-out RFF-L2 discrepancy continued to improve by substantially more than 5%
from 180 to 300 in S1, S2, nonlinear, and HAMD, and successive predictions remained outside the
5% stability tolerance. Per the frozen rule, the final kernel cap is **300 epochs**, patience 45,
evaluation every 3 epochs, batch size 1024. No oracle metric was computed or loaded by the audit.
