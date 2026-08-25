# Response to Reviewer E — Round-2 remediation

**Date:** 2026-08-24  
**Branch:** `mvopm-round2-remediation`  
**Frozen checkpoint:** `aa983a2`

## Disposition

Reviewer E found no P0 and identified one P1 execution-namespace defect. The P1 is closed by
making the old final workflow non-executable: the frozen manifest cannot be rebuilt or used by the
worker, and its aggregate/analysis locations cannot be overwritten implicitly. This patch does not
invent a new confirmatory program, preregistration, or scientific seeds. A new final execution is
therefore intentionally blocked until those scientific inputs and disjoint namespaces exist.

## Finding-by-finding response

| Finding | Disposition |
|---|---|
| E-01 frozen final namespace collision | Fixed. `build("final")` hard-refuses; worker and launcher reject registered frozen bytes even after copy/rename. Modified or unregistered manifests whose `out_path` resolves into frozen final are rejected before mkdir/execution. Aggregate and analysis targets reject real-path ancestor/child intersection with frozen roots, regardless of input phase. Regression tests cover each bypass. |
| E-02 Stage-2 dependency boundary | Narrowed honestly. The scope is `head_only_cross_fitted_with_reused_stage1_oof`; no end-to-end honest CATE-risk claim is made. Full nested Stage-1/Stage-2 evaluation remains future work. |
| E-03 mutable/smuggleable observed view | Fixed for the in-process selector boundary. Arrays use immutable byte buffers, `setflags(write=True)` fails, and selector-facing metadata is empty. Tests cover both attacks. Python in-process privacy is still not claimed as a hostile-code security boundary. |
| E-04 provenance gaps | Fixed. Legacy manifest hashes map to exact execution checkpoints; executable manifests require one full 40-character source commit and explicit clean policy; malformed errors fail validation; source status and dirty-path content digest are rechecked before promotion; provenance failures are retryable rather than terminal. |
| E-05 self-asserted RHC sidecar | Fixed fail-closed. Schema completeness is distinct from scientific approval, and only a sidecar digest in the committed approval registry can authorize causal J2. The registry is empty. |
| E-06 post-hoc subgroup interval | Relabeled `descriptive_conditional_on_posthoc_fixed_regime_subgroup`, with `subgroup_reestimated_in_bootstrap=False`. No unconditional inferential claim remains. |
| E-07 minibatch V-stat target | Documented as `minibatch_vstat_with_diagonal_regularization`, including the batch cap and finite-batch diagonal term. Development sensitivity/convergence work remains required before new evidence. |
| E-08 semantic/compatibility debt | Fixed. Abstention, RHC deployment diagnostics, RHC ATE source, F2 overlap availability, frozen I2 availability, sensitivity availability, and G2 shared-baseline handling now have explicit source/status fields. Frozen G2 pooled selector summaries use 390 deduplicated task rows rather than weighting 480 raw rows. |

## Verification after response

- Python byte-compilation and patch whitespace checks passed.
- Full repository suite: **61/61 passed**.
- Strict read-only frozen-corpus scan: **6,820/6,820 passed**, 0 errors, checkpoint uniformly
  `aa983a2`.
- Frozen aggregate compatibility: all **17** expected table functions completed in a temporary
  directory. Missing post-remediation fields are explicitly marked
  `not_available_in_frozen_corpus`; all 6 figures completed, pooled G2 used 390 task rows, and the
  all-simulation selector scope used 1,700.
- Frozen preregistration, final seed registry, manifest, 6,820 task records, committed aggregates,
  and report were not modified.

## Remaining release gate

This branch is suitable for code review and development diagnostics, but it is deliberately not
ready for a new confirmatory final run. That run requires a new immutable program ID, a new
preregistration, new unexposed scientific seeds, disjoint result/aggregate/analysis roots, and the
prespecified convergence evidence. Confirmatory RHC J2 additionally requires an externally approved
sidecar digest.
