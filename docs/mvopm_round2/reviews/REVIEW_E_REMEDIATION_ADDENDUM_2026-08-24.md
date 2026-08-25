# Reviewer E remediation addendum — independent re-review

**Re-review date:** 2026-08-24  
**Branch:** `mvopm-round2-remediation`  
**Audited worktree base:** `aaef2625a10f3c20827ebbb81ff5c734bc4a4254` plus the current
remediation diff  
**Frozen execution checkpoint:** `aa983a2`  
**Independence:** this reviewer did not implement the remediation. The re-review was read-only
except for this new addendum. No scientific source, test, preregistration, seed registry, manifest,
raw result, aggregate, or prior manuscript was edited.

## Final verdict

**PASS_FOR_CODE_REVIEW_AND_DEVELOPMENT_ONLY.** Within the E-01--E-08 remediation scope, no P0, P1,
or P2 finding remains. In particular, E-01 is now fail-closed at the frozen `final` namespace,
not merely at one manifest pathname or one source checkpoint.

**BLOCKED_FOR_NEW_CONFIRMATORY_FINAL_EXECUTION.** This verdict does not authorize a new final run.
The branch deliberately has no new immutable program ID, preregistration, unexposed seed registry,
manifest, disjoint evidence root, or prespecified repaired-solver convergence evidence. RHC J2 also
has no externally approved sidecar digest. Those are scientific release inputs, not items that this
code-review addendum can infer or waive.

## Severity disposition

- **P0: none.** No corruption, fabrication, truth leak in the current selector path, or defect that
  changes the frozen `NOT_SUPPORTED` verdict was found.
- **P1: none remaining.** The earlier E-01 namespace collision and both bypasses found during this
  re-review were repaired and regression-tested.
- **P2: none remaining within E-01--E-08.** The earlier provenance execution gap and G2 pooled
  baseline inconsistency were also repaired before this final verdict.

## E-01 — frozen final namespace is now fail-closed

The first response was insufficient: it rejected a registered frozen manifest only when its source
checkpoint differed, and aggregation initially protected a pathname rather than the manifest bytes
and destination namespace. During this re-review I additionally reproduced two design bypasses:

1. copying the frozen manifest or supplying an explicit output root could evade pathname-only
   aggregation protection; and
2. changing any manifest byte and supplying a current `source_commit` left the manifest
   unregistered while retaining `out_path` entries inside `results/mvopm_round2/final`. A
   development analysis could likewise target `final/analysis`.

The final patch closes all of these paths:

- `scripts/cluster/manifest_round2.py:41-43` unconditionally refuses `build("final")`;
- `scripts/cluster/worker.py:149-163` and `scripts/cluster/launch.py:81-92` reject the exact bytes of
  either registered legacy manifest, including copies/renames, and separately validate every row's
  canonical output path before directory creation, locking, or subprocess launch;
- `opm/provenance.py:255-286` requires executable manifests to pin one full 40-character lowercase
  hexadecimal commit, an explicit Boolean cleanliness policy, absolute output paths, and no
  real-path intersection with a protected root;
- `scripts/cluster/aggregate_round2.py:52-72` identifies registered manifests by SHA-256 and rejects
  implicit writes or any aggregate destination whose real path equals, contains, or is contained by
  a frozen root; and
- `scripts/analyze_round2.py:959-977` rejects any analysis destination intersecting frozen final,
  regardless of whether the input phase is `final` or `development`.

The regression suite covers the original manifest, a copied/renamed manifest, a modified
unregistered manifest targeting frozen final, aggregate ancestor/child overlap, and both final- and
development-input analysis overlap (`tests/test_round2_remediation.py:135-260`). The modified
manifest test also asserts rejection occurs before `mkdir` and before scientific `run_row`.

This is sufficient to freeze the canonical Round-2 final namespace through the repository's
manifest/launcher/worker/aggregate/analysis workflows. It is not a claim that hostile code with
arbitrary filesystem write access cannot edit files directly.

## E-02--E-08 disposition

| Finding | Independent re-review conclusion |
|---|---|
| **E-02 Stage-2 dependence** | **Honest downgrade accepted.** `crossfit_cate_predictions` excludes each evaluation row from the head fit but explicitly says Stage-1 OOF pseudo-outcomes are reused (`opm/estimator/tau_head.py:115-153`). Emitted scope is `head_only_cross_fitted_with_reused_stage1_oof` (`opm/experiments/mvopm_round2/core.py:131-133`). No end-to-end honest CATE-risk claim remains. |
| **E-03 observed-only boundary** | **Fixed in the claimed in-process scope.** Arrays are copied into immutable `bytes`-backed buffers, object dtype is rejected, and selector-facing metadata is empty (`opm/data/views.py:17-46`). Tests cover `setflags(write=True)`, alias mutation, and allowlisted-key smuggling. Python object privacy is correctly not presented as a hostile-code security boundary. |
| **E-04 provenance** | **Fixed for the execution pipeline.** Registered legacy hashes require an exact recorded short checkpoint; new executable manifests require a full commit and explicit clean policy; result config/index/study/dispatch/four seeds/source/manifest/status are checked; malformed errors fail; checksums, locks, atomic promotion, environment capture, and start/end Git status plus dirty-path content digests are present. The endpoint comparison is an operational integrity gate, not a substitute for the new program's still-required immutable execution environment. |
| **E-05 RHC sidecar** | **Fail-closed.** Schema completeness cannot self-authorize causal use. Authorization additionally requires the exact sidecar SHA-256 in the committed external-approval registry, which is empty (`opm/data/rhc.py:25-66`). Current confirmatory J2 therefore remains unavailable. |
| **E-06 post-hoc subgroup interval** | **Honest downgrade accepted.** The selected mean-risk subgroup interval is explicitly `descriptive_conditional_on_posthoc_fixed_regime_subgroup` and records `subgroup_reestimated_in_bootstrap=False` (`scripts/analyze_round2.py:325-375`). It is not represented as unconditional inference. |
| **E-07 V-statistic target** | **Honest new-estimator boundary accepted.** Training uses the nonnegative V-statistic quadratic (`opm/bridges/kernel_moment.py:52-63`) and convergence outputs label it `minibatch_vstat_with_diagonal_regularization` with the batch cap (`opm/experiments/mvopm_round2/convergence_audit.py:65-70`). The manuscripts correctly require new batch-size sensitivity and convergence evidence. |
| **E-08 semantics/frozen compatibility** | **Closed.** Sensitivity absence, F2 overlap absence, I2 outer-fold availability, nested versus deployment abstention, RHC ATE/diagnostic sources, and relative versus absolute failure events are explicitly named. Frozen G2 pooled selector inputs now retain one shared zero-severity baseline (`scripts/analyze_round2.py:169-220`), yielding 390 G2 rows rather than weighting all 480 raw rows. |

## Independent verification

1. Inspected the complete current remediation diff, the original Reviewer E report, the response,
   the remediation handoff, and the E-01 follow-up fixes. `git diff --check` passed.
2. Ran the focused remediation, Round-2 invariant, and kernel tests: **53/53 passed**.
3. Ran the complete repository suite: **61/61 passed** across eight test files.
4. Streamed the frozen final manifest and all raw result records through the current exact
   provenance validator: **6,820/6,820 passed**, with checkpoint exactly `aa983a2` for every row.
5. Confirmed the frozen manifest SHA-256 remains
   `53acb6c62b0a24f1ff7dc236244f9ca6c287e767c8ed52c1ccc596cf5162820d`.
   Frozen aggregate hashes also remain:
   `AGGREGATION_AUDIT.json=0b96b4d93bb38df788f069d44f22a56dd3e7de95948ccd790f8a3d3cecdc42a3`,
   `raw.csv=308a228d7afe63a28e37e211b291825c3095a5c2f4d3917ecca3627edb5f0130`, and
   `raw_candidates.csv=45b76de690cfb020bdb73938136756f9fbad080410e7404493bc3f21a86c0746`.
6. Reconstructed the revised analysis from frozen aggregates into an automatically removed
   temporary directory: **17 tables and 6 figures** completed. The G2 selector scope contained
   **390** deduplicated rows, `all_simulation_selection` contained **1,700**, and the emitted handling
   field was `single_shared_zero_severity_baseline`.

## Manuscript consistency

`REVIEW_E_RESPONSE_2026-08-24.md` and `ROUND2_REMEDIATION_2026-08-24.md` now match the inspected code
and tests: they report 61/61 tests, namespace-level E-01 protection, exact executable-source policy,
dirty-path content digests, frozen G2 pooled deduplication, and the explicit statistical limitations.

The original `REVIEW_E_REMEDIATION_2026-08-24.md` remains the historical pre-fix finding record.
This addendum supersedes its `CHANGES_REQUIRED_BEFORE_NEW_EXECUTION` status only for code review and
development diagnostics. Its scientific cautions remain operative, and no repaired output may be
presented as new confirmatory evidence without the release inputs listed above.
