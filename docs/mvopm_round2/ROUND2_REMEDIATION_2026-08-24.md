# MV-OPM Round 2 remediation handoff

**Date:** 2026-08-24  
**Remediation branch:** `mvopm-round2-remediation`  
**Frozen corpus checkpoint:** `aa983a2`  
**Pre-remediation audit commit:** `aaef262`

## Evidence-preservation rule

This patch does not modify the frozen preregistration, final seed manifest, any of the 6,820
per-task final results, or the committed final aggregate/report. Solver and estimand changes below
define a post-Round-2 implementation and require a new preregistration and new scientific seeds
before they can produce new scientific evidence. They must not be substituted into the frozen
confirmatory corpus.

## Implemented corrections

| Audit issue | Remediation |
|---|---|
| Kernel q was constrained to `q >= 1` | Replaced `1 + softplus` by `q_min + softplus`, with explicit positive bounds and a regression test reconstructing the exact-DGP q values below one. |
| Kernel U-statistic training objective was indefinite | Training now uses the nonnegative V-statistic RKHS norm. The unbiased U-statistic remains evaluation-only. Minimum/final objectives, update count, parameter norm, and prediction scales are exposed to the convergence audit. |
| The minibatch V-statistic changes the finite-batch target | The training scope is now labeled `minibatch_vstat_with_diagonal_regularization`; the batch cap is recorded. This is a new estimator requiring batch-size sensitivity and convergence studies, not an unchanged substitute for the population cross-moment. |
| Fitted-nuisance multiplier p-values were called formal/calibrated | The API and output now label them fixed-prediction empirical compatibility diagnostics and explicitly deny fitted-model-class calibration. No formal size claim is made. |
| Bootstrap streams changed with library composition | Streams now use a stable SHA-256 candidate-name mapping; expansion invariance is tested for p-values, h, and q. |
| D2 used a different ATE estimand | D2 now uses the realized empirical-X mean of `tau(X)` as `ate_true` and retains the population ATE separately. |
| PEHE was evaluated on the Stage-2 training X | Candidate and nested-MSES PEHE use head-only cross-fitted Stage-2 predictions. Output scope is explicitly `head_only_cross_fitted_with_reused_stage1_oof`; it is not described as end-to-end nested CATE risk. Unused full-sample candidate heads were removed. RHC skips CATE fitting because it has no oracle CATE target. |
| RHC reported the global adaptive candidate's naive interval as primary | Primary RHC output now uses the nested outer-OOF adaptive estimate. Its Wald interval is explicitly descriptive and carries no post-selection coverage guarantee; the deployment estimate and naive interval are separate fields. |
| RHC proxy mapping lacked verifiable provenance | Confirmatory J2 now fails closed unless a sidecar verifies the exact data checksum, V/W mapping, DOI/citation/page/variable definition, preparation-script checksum, reviewer, and date, and its exact digest appears in a committed external-approval registry. The registry is empty; no sidecar can self-authorize. |
| I2 labels asserted unproved library truth | Outputs use engineered scenario labels and separate nested-any-fold, mean-fold, and deployment abstention rates. |
| Task-level oracle conditioning overstated switching | Analysis uses regime mean ATE risk for primary switching and for the non-sieve1 subgroup. Sample-wise argmin instability remains a separately named descriptive artifact. |
| A2 correlations treated 5,000 rows as independent | Diagnostic correlations now use the 25 corruption-cell means. |
| F2 conflated confounding and overlap | S1/S2 retain true arm probabilities; tasks report arm counts and overlap summaries so the joint change is visible. |
| G2 repeated the zero-corruption baseline four times | Future manifests contain one shared zero-corruption baseline. Frozen raw rows are unchanged; pooled frozen selector summaries deduplicate them to one shared baseline (390 G2 task rows rather than 480). |
| Holm/max-variance sensitivities were absent | The frozen Bonferroni/mean-variance rule remains primary; Holm arm rejections and maximum-contrast-variance selection now run through the same nested outer evaluation as named sensitivities. |
| q clipping/pathology was hidden | Candidate rows retain min/max q, lower/upper boundary fractions, balance error, and an explicit numerical-pathology flag. |
| Relative `>1.5x oracle` failures were called catastrophic without absolute context | New tables use an explicit relative-failure label and add absolute ATE-error tail rates at 0.05 and 0.10. The legacy raw field remains readable. |
| ObservedDatasetView was mutable and metadata was blacklist-based | Arrays are backed by immutable byte buffers, so `setflags(write=True)` fails; attributes are sealed and selector-facing metadata is empty. Alias-smuggling, metadata-smuggling, and mutation tests were added. |
| Resume/aggregation accepted stale or swapped results | Manifest rows, config, array index, study/dispatch, all four seeds, full source commit, clean policy, dirty state and dirty-path content digest, manifest hash, result status/error schema, and result checksum are validated fail-closed. The two legacy manifests are registered to exact historical checkpoints, and worker source state is compared again before result promotion. |
| A repaired final run could overwrite the frozen Round-2 namespace | `build("final")` now hard-refuses. Worker/launcher execution through registered frozen-manifest bytes refuses unconditionally. Modified or unregistered manifests are also rejected before directory creation or execution whenever any `out_path` resolves inside the frozen final namespace. Aggregate/analysis targets reject real-path intersection with frozen roots. A new final remains blocked until a new program ID, preregistration, unexposed seed registry, manifest, and result/analysis roots are supplied. |
| Result writes could race or tear | Workers use a nonblocking per-result OS lock, same-directory fsync plus atomic replace, unique attempt archives, and a SHA-256 companion file. |
| Runtime environment was not reproducible/auditable | Direct versions are pinned in `requirements-lock.txt`; workers record Python/package/platform/CPU, BLAS/LAPACK, CUDA/cuDNN, determinism flags, thread caps, full git state, and manifest hash. |

## Statistical boundaries that remain

1. The fixed-prediction multiplier diagnostic is not a nuisance-aware test of fitted candidate-class
   adequacy. A formal claim still requires an honest independent test sample, refit bootstrap, or a
   locally robust moment plus supporting theory and calibration experiments.
2. Nested RHC intervals are descriptive. They do not establish nominal selective-inference or
   repeated-cross-fit coverage.
3. RHC remains a provisional engineering example until an externally reviewed proxy/data sidecar
   is supplied and its digest is added to the committed approval registry.
4. The frozen preregistration's incomplete pooling/weighting/verdict specification cannot be repaired
   retrospectively. Any new confirmatory program must freeze those details before new seeds.
5. Raising the kernel cap to 300 was a frozen cap decision, not proof of convergence. The repaired
   solver needs a new observed-data convergence audit before new final runs.

## Validation

- Patch hygiene and Python byte-compilation: passed.
- Focused kernel, provenance, RHC, Stage-2, sensitivity, and isolation regressions: passed.
- Full repository test suite: 61/61 passed after the independent-review response.
- Strict read-only manifest/result validation: 6,820/6,820 final JSON records passed; execution
  checkpoint was uniformly `aa983a2`; manifest SHA-256 was
  `53acb6c62b0a24f1ff7dc236244f9ca6c287e767c8ed52c1ccc596cf5162820d`.
- Revised read-only analysis on the frozen aggregates: 6,820 tasks, 10,020 candidates, 25 A2 cells,
  20 R regimes, and 16 regime-level transitions completed in a temporary directory. The four
  regime mean-risk winners were `kernel_kernel`, `kernel_sieve1`, `sieve1_kernel`, and
  `sieve2_sieve2`. All 17 tables and 6 figures were generated; pooled G2 used 390 deduplicated task
  rows and `all_simulation_selection` used 1,700.
- Independent remediation review completed with no P0 and one P1 namespace finding. The P1 and
  all concrete P2/P3 fail-closed or labeling requests were addressed; final independent re-review
  is recorded in the review addendum.
