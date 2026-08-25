# Reviewer E — Round-2 remediation independent audit

**Audit date:** 2026-08-24  
**Branch:** `mvopm-round2-remediation`  
**Frozen execution checkpoint:** `aa983a2`  
**Pre-remediation audit commit:** `aaef262`  
**Independence:** this reviewer did not implement the remediation. The audit was read-only except
for this report. No scientific source, test, manifest, raw result, aggregate, or prior manuscript
was edited.

## Verdict

**CHANGES_REQUIRED_BEFORE_NEW_EXECUTION.** There is no P0 finding and no evidence that the frozen
6,820-result corpus has already been changed. The core mathematical repairs are mostly implemented
as described, and all 50 repository tests passed independently. However, one P1 execution-namespace
defect must be fixed before the remediation branch is used: the new `final` manifest still reuses
the old final seeds, manifest filename, result paths, aggregate paths, and analysis namespace. A
normal post-commit final run would therefore replace the canonical frozen corpus instead of creating
a new scientific program.

I also found several P2 boundaries that the current tests do not exercise: Stage-2 PEHE is
head-cross-fitted but not end-to-end nested across the first-stage bridge fits; the observed-data
view can be made writable and can carry mutable truth through an allowlisted key; legacy result
validation does not enforce the recorded execution checkpoint; error records are under-validated;
the RHC sidecar gate verifies a self-assertion rather than the cited scientific mapping; and a
post-hoc mean-risk subgroup interval does not repeat subgroup construction inside its bootstrap.

## Independent verification performed

1. Read the synthesis and both prior independent reviews, then audited every current source/test
   diff and the remediation handoff.
2. Ran the kernel/selector/provenance/RHC/Stage-2 remediation tests plus Round-2 invariants:
   **42/42 passed**.
3. Ran the remaining DGP, closed-form bridge, DR, and cross-fit tests: **8/8 passed**. Total:
   **50/50 passed** under Python 3.12.12 and the exact direct versions in
   `requirements-lock.txt`.
4. Streamed all final manifest rows and JSON records read-only: **6,820/6,820** passed the current
   config/index/seed validator; every record reported short checkpoint `aa983a2`.
5. Re-ran the revised table functions against the frozen `raw.csv` and `raw_candidates.csv` into a
   temporary directory. All 17 expected tables were produced: 6,820 tasks, 10,020 candidate rows,
   25 A2 cells, 20 R regimes, 16 regime-level transitions, and four regime mean-risk winners.
6. Confirmed there is no RHC provenance sidecar and that
   `load_rhc(require_verified_proxy_mapping=True)` fails before model fitting.
7. Confirmed the current lock rejects a second owner and the checksum rejects a changed JSON.

## P0 — none

No current frozen-result corruption, truth use by the existing selector, fabricated RHC sidecar, or
defect capable of reversing the stored `NOT_SUPPORTED` result was found.

## P1 — must fix before any new final execution

### E-01 — The remediation final workflow targets the frozen seeds and namespace

**Code:**

- `scripts/cluster/manifest_round2.py:11-12, 50-53, 73-78, 145-159`
- `scripts/cluster/worker.py:202-212`
- `scripts/cluster/aggregate_round2.py:22, 134-167`
- `scripts/analyze_round2.py:24-25, 874-910`

**Evidence:** `SEED_PATH` still points to `configs/seeds/mvopm_round2_final.json`; `OUT` still points
to `results/mvopm_round2`; phase `final` writes `manifest_final.jsonl`; every output row still lands
under `results/mvopm_round2/final/<study>/<config>/seed.../result.json`. The worker deliberately
archives any existing canonical record and atomically promotes the new record in its place.
Aggregation then writes the same `final/raw.csv`, `raw_candidates.csv`, `AGGREGATION_AUDIT.json`,
and analysis directory.

After the remediation is committed, `build("final")` is allowed. Its rows name the new source
commit, so `--resume` correctly rejects all old `aa983a2` results as stale and schedules them for
replacement. It also overwrites the tracked frozen manifest before execution. The three removed G2
zero-severity cells mean the new manifest would contain 6,730 rather than 6,820 rows, making the
namespace collision especially visible.

**Impact:** a routine final run would contaminate the frozen confirmatory evidence package and reuse
the exposed final scientific seeds for a changed solver/estimand. The old JSON is copied into an
`attempts/` directory, so loss is partly recoverable, but the canonical corpus, aggregate, and report
inputs would no longer be the frozen experiment.

**Required fix:** make the repaired method a new program with a new immutable program ID, new
preregistration, new seed registry, new manifest filename, and disjoint result/aggregate/analysis
root. The old Round-2 manifest generator should either remain frozen or hard-refuse a new `final`
build. Add a regression that no path or scientific seed in a remediation/future final manifest
intersects the frozen Round-2 manifest. Do not rely on `attempts/` as evidence preservation.

## P2 — substantive but not evidence-destroying in the current state

### E-02 — Stage-2 PEHE is not end-to-end cross-fitted across Stage 1

**Code:** `opm/estimator/tau_head.py:115-149`,
`opm/experiments/mvopm_round2/core.py:46-80`, `opm/estimator/nested_mses.py:204-216`  
**Test:** `tests/test_round2_remediation.py:156-169`

`crossfit_cate_predictions` excludes evaluation row `j` from the Stage-2 head fit. But its training
labels are OOF pseudo-outcomes `phi_i` created earlier by a different first-stage fold partition.
For most `i` outside `j`'s Stage-2 validation fold, the bridge model used to construct `phi_i` was
trained using row `j`. Thus `j`'s observed `(T,Y,W,V)` can influence the labels used to train the
head that predicts at `X_j`. The new test checks only finite shape and coverage; it does not trace
this two-stage dependency.

**Impact:** `pehe_evaluation_scope="stage2_cross_fitted"` is literally correct at the head level,
but the result is not an end-to-end honest out-of-sample CATE risk. It may remain optimistic or have
nonstandard dependence. ATE evaluation is unaffected because it uses the already outer-OOF
pseudo-outcome mean.

**Fix:** either label this more narrowly as `head_only_cross_fitted_with_reused_stage1_oof` and keep
it secondary, or construct an outer Stage-2 fold in which every nuisance and pseudo-outcome used to
train the head is generated without the Stage-2 evaluation fold (with inner cross-fitting inside the
outer training sample). Add a row-dependency/leakage test, not just a coverage test.

### E-03 — `ObservedDatasetView` blocks accidental mutation, not adversarial mutation/smuggling

**Code:** `opm/data/views.py:17-27, 35-49, 56-82, 96-108`  
**Test:** `tests/test_mvopm_invariants.py:161-174`

The arrays are owning NumPy copies with `writeable=False`. A consumer can call
`view.X.setflags(write=True)` and then mutate them. `MappingProxyType` prevents replacing a metadata
entry but does not freeze its value. The allowlist also has no scalar/type validation: placing a
truth array in an allowed key such as `ds.meta["dgp"]` carries that array into the view, where it
remains mutable. Both bypasses were reproduced. Private `_X` and `_meta` are also directly reachable
in Python.

**Impact:** current generated metadata uses benign scalar values and no current selector was found
reading truth, so this is not evidence of leakage in the frozen experiment. It does invalidate the
strong “immutable/physically hidden” claim and leaves the security boundary convention-based.

**Fix:** validate allowlisted values against an explicit scalar/enum schema and deep-copy/freeze
them. For arrays, return a copy per access or use storage backed by an immutable buffer so
`setflags(write=True)` fails. If adversarial isolation is truly required, place selection in a
separate process with a serialized observed-only schema; Python object privacy is not a security
boundary. Add tests for `setflags`, allowed-key smuggling, nested mutable metadata, and `_X` access.

### E-04 — Provenance is strong for new successful records, but three fail-closed gaps remain

**Code:** `opm/provenance.py:62-100, 171-184, 192-236`,
`scripts/cluster/worker.py:145-200`, `scripts/cluster/launch.py:32-61`

1. Source cleanliness is captured only before a potentially long task. Code can change in the
   shared worktree after line 154, yet the result will retain `git_dirty=false`. There is no
   end-of-run source-state comparison or isolated checkout.
2. For a legacy manifest without `source_commit`, `validate_result_provenance` does not compare the
   result's `git_commit`. The independent scan found all 6,820 rows at `aa983a2`, but the current
   aggregator would accept an otherwise matching legacy result from another checkpoint.
3. An error record with exact config/seeds but no nonzero `exit_code`, no `error`, and no valid
   `failure_kind` passes validation. `_terminal` then treats it as a terminal algorithmic record.
   This malformed-record acceptance was reproduced. The OS lock itself correctly rejected a second
   owner.

**Impact:** none of these defects was observed in the current corpus, and new successful records do
receive exact config/source/manifest/checksum validation. They nevertheless make the blanket
“fail-closed” claim too broad and can misclassify a future or legacy run.

**Fix:** execute from an immutable detached worktree/container or compare source status/diff hash
again before promotion; register and enforce `aa983a2` explicitly for legacy Round 2; require all new
manifest rows to have the same nonempty source commit and cleanliness policy; validate error records
with nonzero exit, nonempty exception, valid failure class, and an explicit retry policy. Add tests
for source changes during a task, mixed/missing source fields, wrong legacy checkpoint, and malformed
error status.

### E-05 — The RHC sidecar gate verifies self-asserted fields, not the scientific citation

**Code:** `opm/data/rhc.py:31-49, 63-72`  
**Test:** `tests/test_round2_remediation.py:139-153`

The checksum and exact V/W string lists are checked correctly, and current confirmatory J2 fails
closed because no sidecar exists. But citation verification is only
`bool(paper_citation and paper_page_or_table)`, while `proxy_mapping_verified` is trusted directly.
The regression test passes a fabricated `"p. 123, Table 4"`; any nonempty placeholder would pass.

**Impact:** adding a syntactically complete but scientifically unreviewed sidecar upgrades J2 to
`causal_inference_authorized=True`. The gate prevents accidental omission but does not establish that
the paper actually defines these variables as the claimed proxies.

**Fix:** make the sidecar an externally reviewed evidence record with fixed DOI/bibliographic ID,
exact quoted variable definition within copyright limits, real page/table, preparation-script hash,
reviewer/date, and preferably a signed/committed approval digest. Reject placeholders and distinguish
`schema_complete` from `scientifically_verified`. The current absence of a sidecar should remain
fail-closed.

### E-06 — The non-sieve1 mean-risk subgroup CI conditions on a data-selected subgroup

**Code:** `scripts/analyze_round2.py:246-291, 294-335, 824-850`

Replacing per-task realized-oracle conditioning with regime mean risk is a major improvement. But
the same candidate rows first determine which regimes have a non-sieve1 mean-risk oracle and then
estimate MSES-minus-fixed-sieve1 performance within that subgroup. The seed-block bootstrap keeps
the eligible regime set fixed; it does not re-estimate mean-risk winners and subgroup membership in
each resample.

**Impact:** the subgroup mean and its displayed 95% interval do not include uncertainty from the
post-hoc regime classification. Near-tied regimes can be selected into the subgroup by sampling
noise. This is a secondary-claim problem and does not affect the full-scope MSES failure.

**Fix:** label the subgroup contrast descriptive/post-hoc, or repeat winner estimation and subgroup
construction inside each block bootstrap. Prefer a separately frozen structural-regime definition,
leave-one-seed-block-out classification, or report all regime-specific paired contrasts with
mean-risk gaps rather than a selected-subgroup confidence interval.

### E-07 — The minibatch V-statistic is bounded but changes the finite-batch target

**Code:** `opm/bridges/kernel_moment.py:49-60, 174-207`  
**Test:** `tests/test_kernel_ustat.py:32-50`

The repair correctly removes the unbounded-below U-statistic objective, and the implemented Gaussian
kernel quadratic is nonnegative up to numerical error. However, for iid batch size `B`,

`E[V_B] = ((B-1)/B) * population_cross_moment + (1/B) * E[r^2 K(z,z)]`.

With `K(z,z)=1`, the diagonal adds outcome-residual MSE on the h side and a shrinkage-like
`(q-1)^2` term on treated observations on the q side. Because training uses a fixed minibatch cap,
this term does not disappear merely by increasing dataset size. The nonnegativity test is valid but
does not test bridge recovery, target shift, or repaired-solver convergence.

**Impact:** the new objective is safe to optimize but is a new estimator, not a drop-in unbiased
version of the old population norm. The handoff correctly says new evidence and a new convergence
audit are required; no scientific performance claim should precede that audit.

**Fix:** document the diagonal regularization explicitly, quantify its sensitivity to batch size,
and run exact-bridge recovery and observed-data convergence studies on new development seeds.
Consider full-fold V-stat/RFF mean-squared objectives or a prespecified diagonal-weight sensitivity.
Do not describe the objective as unchanged solely because it is nonnegative.

## P3 — clarity and coverage debt

### E-08 — Some output names and frozen-data compatibility paths remain semantically incomplete

- `scripts/analyze_round2.py:341-355` still emits generic `abstention_rate` for the
  any-outer-fold status, while the I2-specific table uses explicit nested/deployment names.
- `scripts/analyze_round2.py:924-926` always says “reported deployment ATEs”; future repaired J2
  rows instead use nested outer-OOF ATE as primary. The generated README should branch on source.
- `opm/experiments/mvopm_round2/studies.py:238-249` attaches plain `ess`, `max_q`, and `q_balance`
  from the deployment-selected candidate to the primary nested ATE. Rename them
  `deployment_*` or compute nested diagnostics so sources cannot be conflated.
- Running `sensitivity_tables` on the frozen CSV writes an empty, headerless CSV because the new
  sensitivity selectors did not exist in the frozen results. Write an explicit schema/status row.
- The revised analysis can read all frozen data, but frozen I2 has no mean-fold abstention field,
  frozen F2 has no true-overlap columns, and frozen G2 still contains four zero-severity baselines.
  These should be marked `not_available_in_frozen_corpus`; pooled frozen summaries should deduplicate
  G2 zero severity instead of implying the future manifest change retroactively repaired it.

These issues affect interpretation and automation, not the core stored verdict.

## Repairs verified as correct in scope

The following changes behaved as intended under code inspection and directed tests:

- `q_min + softplus` represents the exact finite-proxy q values below one, while preserving a
  positive lower bound and upper cap.
- Kernel training uses a PSD V-statistic quadratic; the adversarial two-row U-statistic remains
  negative while the V-statistic is nonnegative.
- Candidate bootstrap seeds are stable under library expansion for a fixed base seed and candidate
  name.
- Moment outputs consistently carry the fixed-prediction empirical-compatibility calibration label;
  they no longer claim fitted-model-class adequacy.
- D2 now uses empirical-X mean `tau(X_i)` as `ate_true` and retains population ATE separately.
- Holm arm adjustment and maximum-contrast-variance selection are explicitly separate nested
  sensitivities; the tested Holm step-down example is correct.
- Primary RHC ATE payload comes from nested outer-OOF adaptive pseudo-outcomes, while the deployment
  ATE/naive interval has separate fields and the primary interval denies a coverage guarantee.
- I2 uses engineered scenario labels and separates nested-any-fold from deployment abstention where
  the new raw fields exist.
- A2 correlations use 25 cell means; switching uses regime mean ATE risk and keeps sample-wise argmin
  instability separate. The frozen data yield four regime-level winners, not six.
- DGP arm probabilities/arm counts, q boundary/pathology fields, direct environment versions,
  atomic same-directory promotion, checksum verification, and nonblocking result locking are present.
- The current RHC J2 causal path is fail-closed due to the absent sidecar.

## Statistical boundaries that remain honest and must stay in every new manuscript

1. The multiplier diagnostic is not a nuisance-aware fitted-candidate-class adequacy test.
2. Nested RHC Wald intervals are descriptive and have no demonstrated selective/repeated-cross-fit
   coverage.
3. RHC remains provisional until the actual data/proxy mapping receives external source review.
4. Kernel V-stat/q-link changes define a new estimator and require new development diagnostics,
   preregistration, and unexposed scientific seeds.
5. Stage-2 head cross-fitting alone is not end-to-end nested CATE-risk evaluation.
6. The frozen Round-2 pooling/weighting/verdict ambiguity cannot be repaired retrospectively.
7. Regime mean-risk winners are empirical finite-replication summaries; near-tie uncertainty and
   post-hoc subgroup selection must be shown.

## Release recommendation

Do not run or merge this branch as a new `final` experiment until E-01 is fixed and regression-tested.
After that, E-02 through E-07 should either be repaired or explicitly narrowed in the new
preregistration and manuscripts. The current frozen corpus and `NOT_SUPPORTED` report should remain
under the original namespace and checkpoint without regeneration.
