# MV-OPM repaired confirmatory v1 preregistration

**Program ID:** `mvopm_repaired_v1`  
**Protocol version:** 0.2 (pre-final convergence amendment)
**Date frozen:** 2026-08-24  
**Protected remediation commit:** `d08fc5fa4ea7a715c7f73d6b0d7d5dbcd0101004`  
**Final execution:** forbidden until the release-only authorization amendment is committed
**Final seed generation:** authorized

## Evidence separation

This is a new scientific program, not a regeneration or continuation of the frozen Round-2 final
corpus. No Round-2 final seed or output path may be reused. All new evidence is written below
`results/mvopm_repaired_v1/`; the old `results/mvopm_round2/` namespace, seed registry,
preregistration, raw records, aggregates, and `NOT_SUPPORTED` verdict remain immutable.

The development convergence audit is written below
`results/mvopm_repaired_v1/development/convergence_audit/` and is never pooled with final evidence.
Final scientific seeds will be generated only after the observed-data convergence decision is
frozen. Their allocation file and its SHA-256 will be committed before the final manifest is built.

## Thesis and primary algorithm

The thesis remains that proximal estimator selection requires two distinct observed-data signals:
identifying-moment compatibility for structural bias risk and out-of-fold uncertainty for finite-
sample efficiency.

The primary algorithm is Moment-Screened Efficiency Selection (MSES):

1. compute fixed-prediction empirical identifying-moment compatibility diagnostics for each
   candidate and arm;
2. retain candidates satisfying the prespecified empirical compatibility screen;
3. among survivors, select the candidate with the smallest mean out-of-fold pseudo-outcome
   contrast variance; and
4. return `ABSTAIN_LIBRARY_INADEQUATE` when no candidate survives.

The multiplier p-values are explicitly scoped as
`fixed_prediction_empirical_compatibility_not_fitted_model_class_adequacy`. They are algorithmic
screening scores, not a formal nuisance-aware test of fitted candidate-class adequacy. No type-I
error, proxy-validity, or model-class sufficiency claim is permitted. Bonferroni at `0.05/K` is the
primary screen; Holm is a named descriptive sensitivity. Maximum contrast variance is a named
efficiency sensitivity; mean contrast variance is primary.

The Round-1 `biasvar_mult` and `biasvar_add` scores remain exploratory baselines and cannot be
promoted to the confirmatory method.

## Candidate library and repaired estimator

The global library is fixed as `kernel_kernel`, `sieve1_sieve1`, `sieve2_sieve2`,
`sieve3_sieve3`, `kernel_sieve1`, and `sieve1_kernel`. The feasible high-dimensional HAMD subset is
`kernel_kernel`, `sieve1_sieve1`, `kernel_sieve1`, and `sieve1_kernel`.

The kernel q bridge is `q_min + softplus(raw)` capped at `q_max`, with `q_min=1e-4` and `q_max=50`.
Kernel bridge training minimizes the nonnegative minibatch V-statistic RKHS quadratic. Its diagonal
term changes the finite-batch target and is labeled
`minibatch_vstat_with_diagonal_regularization`; the evaluation-only unbiased U-statistic is not an
optimization objective. Boundary fractions, balance error, prediction scales, objective traces,
parameter norm, and numerical-pathology flags are retained.

## Development-only convergence decision

Before final seeds are generated, 8 new development seeds are run on each of S1, S2, nonlinear E9,
and HAMD semi-synthetic data at n=1500. Each task evaluates kernel caps 45, 90, 180, and 300 with
evaluation every 3 epochs, batch-size cap 1024, and patience `max(15, round(0.15*budget))`.

The HAMD audit input is fixed at
`/home/xqin5/DpressionTreatmenteffect/data-merged-def-1.csv`, SHA-256
`b40d7c17e687170c2c17c3eb892fd90a73a51e3b1e96864430ffdc2ce7264609`. The runner validates this
digest before task dispatch and again before issuing a decision; every shard records it. A mismatch
is an infrastructure/provenance failure and cannot fall back to another file.

Only observed-data quantities may determine the cap: held-out RFF-L2 and MMR discrepancies,
successive h/q prediction changes, optimizer objective/updates, parameter norm, q boundaries, and
numerical stability. Oracle ATE error, PEHE, bridge truth, and candidate winner are absent from the
audit runner.

For adjacent caps B and B', B is the first plateau only when the median held-out RFF-L2 relative
improvement at B' is below 5% and both median relative h and q prediction changes at B' are below
5%. If no earlier cap qualifies, 300 is selected. The decision, audit hashes, device, source commit,
and summary are frozen in `CONVERGENCE_DECISION.json` and a pre-final amendment below. The selected
cap cannot be reduced or increased using final or oracle outcomes. Batch-size sensitivity remains a
required development diagnostic and must be described as estimator sensitivity rather than proof
of population-objective equivalence.

The selected kernel cap is 300 epochs. Before final execution, a descriptive batch-size sensitivity
diagnostic will evaluate caps 256, 512, and 1024 at this fixed epoch cap, using the same 8
development seeds, four development scenarios, n=1500, held-out observed-data moment diagnostics,
and no oracle quantities. It cannot alter the selected epoch cap, final seeds, candidate library, or
task allocation. The batch-1024 records from the frozen convergence audit are the reference; new
batch-256 and batch-512 records must reproduce its split/bank construction and pass the same finite-
value, HAMD-input, source, checksum, and completeness gates. Its interpretation is limited to
finite-minibatch estimator sensitivity and is not evidence of population-objective equivalence.

The secondary CATE head cap is 300 epochs with patience 30. Its evaluation scope is
`head_only_cross_fitted_with_reused_stage1_oof`; it is not an end-to-end nested CATE-risk estimate.

## Cross-fitting, seeds, and provenance

Final selection uses four outer folds and three inner folds; outer-test observations cannot affect
scalers, banks, diagnostics, selection, or refitting. The primary uncertainty criterion is the mean
across non-reference contrasts of the ddof=1 variance of complete inner-OOF pseudo-outcome
contrasts. Ties are lexicographic.

Scientific seeds are sampled without replacement from `[1_000_000_000, 2_000_000_000)` using the
operating system cryptographic RNG and explicitly stored; they are not derived from array indices.
Development and final seeds are disjoint from one another and from all registered Round-2 seeds.
Bootstrap and bank seeds are deterministic SHA-256 streams over program ID, study, config, and
scientific seed with collision resolution in canonical manifest order. Split seed equals the
explicit scientific seed. Infrastructure retries retain the same row and seed.

Every executable manifest row pins one full lowercase-hex source commit and explicit clean-source
policy. Workers validate manifest/config/index/study/dispatch/all seed streams/source state and
content digest, use per-result locks and atomic promotion, and record environment provenance.

## Fixed studies and replication counts

The authorized final program contains 6,700 tasks:

- A2: 25 aligned corruption cells × 200 seeds at n=4000 = 5,000;
- B2 S1, C2 S2, D2 nonlinear, E2 HAMD: 50 seeds each = 200;
- R: 20 frozen structural regimes × 30 seeds = 600;
- F2: four confounding levels × 30 seeds = 120;
- G2: one shared zero-corruption baseline plus 12 positive corruption cells × 30 seeds = 390;
- H2: two families × four sample sizes × 30 seeds = 240; and
- I2: three engineered library scenarios × 50 seeds = 150.

J2 RHC is not authorized in this program because the external scientific proxy-mapping approval
registry is empty. It may not be silently replaced by a schema-complete self-asserted sidecar. A
future J2 amendment requires an externally reviewed sidecar digest and separate authorization.

Study definitions, regime matrix, aligned A2 perturbations, empirical-X D2 ATE target, feasible HAMD
library, F2 overlap reporting, I2 engineered-scenario labels, and G2 shared baseline follow the
repaired source at the final execution commit. No task-level oracle outcome can change a candidate,
study, budget, or seed allocation.

## Targets, comparisons, and reporting boundaries

Primary simulation performance is absolute ATE error averaged across non-reference contrasts.
Secondary metrics are oracle ratio, regret, relative event `error > 1.5 * realized oracle error`,
absolute error tails at 0.05 and 0.10, top-1/top-2, oracle survival, abstention, and q pathology.
PEHE is secondary with the head-only scope above.

Primary pooled comparisons use scientific-seed block resampling. Regime switching is summarized by
regime mean ATE risk, not noisy per-task winners. Any subgroup defined by a post-hoc mean-risk winner
is labeled descriptive and conditional, with subgroup construction not re-estimated in its interval.
G2 pooled summaries retain one shared zero-severity baseline. Holm and maximum-variance results are
sensitivity analyses, never substitutions for the primary rule.

Success or failure will be reported without changing the method. No repaired output can be merged
with frozen Round-2 evidence or described as a rerun of the old confirmatory program.

## Pre-final amendment

The observed-only convergence audit completed 32/32 tasks and 128/128 budget rows at source commit
`13e6ab0b17480e86c739dae407681f17fe1c4082`, using logical device `cuda:0` bound by the queue to
physical GPU 1. No oracle ATE error, PEHE, bridge truth, or candidate winner was loaded. None of the
three adjacent budget transitions met the frozen 5% h/q stability criterion, so the frozen fallback
rule selected the 300-epoch cap. Final seed generation is therefore authorized; final execution is
not.

Frozen convergence evidence:

- development seed registry SHA-256:
  `30dc5779c30f3912118bb01f42a93067186bd47a1f662066db941937b81795b3`;
- HAMD input SHA-256:
  `b40d7c17e687170c2c17c3eb892fd90a73a51e3b1e96864430ffdc2ce7264609`;
- 32 shard JSON files plus 32 checksum sidecars, canonical path/hash-list SHA-256:
  `2c8291d1b3d265fcb32dc849da568db961b8127692b9beaf10f14d8992642f9a`;
- `observed_convergence.csv` SHA-256:
  `05bf43c4dc809e3d64e61652f785c6f9db1b9cb4fa31320724316dc629af7c3f`;
- `summary.csv` SHA-256:
  `2f9b0412cefe51811f9bdff98513a36df91d0dec2419f8a04cdc544257e2b644`;
- `CONVERGENCE_DECISION.json` SHA-256:
  `b1c044e93893a0d79fa13cc7e79d1f3318be76206b00c2d4a1b3d09ab742abc9`; and
- `CONVERGENCE_AUDIT.md` SHA-256:
  `323c58a480376723095876ae8a075498b0112061800f96b2c49ee1b4d2f1cbee`.

The release sequence is split to avoid a manifest/source-commit self-reference. First, a clean
**scientific source commit** must contain the final seed registry, program-specific builder and
release gates, all executable scientific code, and their tests. The final manifest is then generated
from that commit and every row pins it as `source_commit`. After an independent manifest review, a
later **release-only authorization commit** may add only the exact manifest, authorization metadata,
this preregistration's release appendix, and the independent review report. That authorization must
record the scientific source commit, exact seed-registry and manifest SHA-256 values, 6,700-row
count, selected budget 300, convergence and batch-sensitivity evidence hashes, and review verdict.

The program-specific launcher and every worker must verify that the current clean HEAD is the
release-only authorization commit, that the scientific source commit is its ancestor, and that the
tracked diff between them contains only the exact release-artifact allowlist recorded in the
authorization. Any code, configuration, seed-registry, or study-definition change between those
commits is fatal. Until the batch sensitivity is frozen and this release-only authorization exists,
the builder may prepare evidence but the launcher must refuse final execution. The generic GPU-idle
watcher is only a resource waiter and never constitutes scientific authorization.
