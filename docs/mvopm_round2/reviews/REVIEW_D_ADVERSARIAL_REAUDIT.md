# Reviewer D adversarial re-audit: engineering, design, and manuscript evidence chain

**Audit date:** 2026-08-24  
**Repository:** `/home/xqin5/进化的opm`  
**Audited branch/HEAD:** `mvopm-round2-confirmatory`, `aaef262`  
**Frozen execution checkpoint:** `aa983a23ab34d58b9381ea60dbe502b261bb58d5`  
**Reviewer posture:** independent and adversarial; prior reviews were treated as claims to re-check, not as evidence.  
**Mutation policy:** read-only review of scientific code/results. This report is the only repository file added.

## Verdict

**REQUIRES_MAJOR_REMEDIATION_BEFORE_METHOD/PAPER RELEASE.**

The final numerical corpus is substantially better preserved than the code design and manuscripts imply:
all 6,820 raw tasks exist, all provenance fields in the current corpus match the manifest, and an independent
full reconstruction reproduced `raw.csv`, `raw_candidates.csv`, and `AGGREGATION_AUDIT.json` byte-for-byte.
The headline negative conclusion—MSES is not a reliable near-oracle selector—is numerically robust to the
issues found here.

However, the implementation/paper package should not receive an engineering or methods `PASS` yet. Two
adversarial checks expose untested code-design defects: the kernel bridge minimizes an indefinite, potentially
unbounded empirical U-statistic objective, and a candidate's formal p-value changes when an unrelated earlier
candidate is added to the library. The real-data CIs are post-selection full-OOF CIs even though a nested
estimator was computed. The aggregator is not fail-closed against stale/swapped result files, the execution
environment is not pinned or recorded, and the older “complete” manuscripts are now materially superseded by
Round 2 but do not say so.

Severity convention: **HIGH** can invalidate an implementation claim, inferential claim, or release-level
reproducibility; **MEDIUM** materially weakens interpretation or auditability; **LOW** is a bounded correctness
or presentation defect. No `BLOCKER` is assigned because the negative MSES verdict survives the checks below.

## What independently passed

### P1. Full manifest-to-raw reconstruction passed

I loaded all 6,820 `result.json` files named by `manifest_final.jsonl` and rebuilt the aggregate artifacts in
`/tmp/mvopm_reviewer_d_reaggregate` with the checked-in aggregator.

| Artifact | Original SHA-256 | Rebuilt SHA-256 | Result |
|---|---|---|---|
| `raw.csv` | `308a228d7afe63a28e37e211b291825c3095a5c2f4d3917ecca3627edb5f0130` | same | exact |
| `raw_candidates.csv` | `45b76de690cfb020bdb73938136756f9fbad080410e7404493bc3f21a86c0746` | same | exact |
| `AGGREGATION_AUDIT.json` | `0b96b4d93bb38df788f069d44f22a56dd3e7de95948ccd790f8a3d3cecdc42a3` | same | exact |

The reconstructed counts were 6,820 manifest rows, 6,820 successful task rows, 10,020 candidate rows, zero
missing rows, and zero failed rows. `raw.csv` has shape `(6820, 261)` and `raw_candidates.csv` has shape
`(10020, 87)`. Manifest task keys, output paths, bootstrap seeds, and bank seeds are each unique.

### P2. Current corpus provenance is internally consistent

A second full scan compared every raw record with its manifest row. There were zero mismatches for:

- `array_task_id`;
- exact serialized `config`;
- study/dispatch;
- scientific, split, bootstrap, and bank seeds;
- result-level scientific seed;
- `status=ok`, `exit_code=0`;
- frozen checkpoint `aa983a2`.

This is stronger evidence than the current aggregator itself provides (Finding D-04).

### P3. Reported primary numbers and local links passed

The principal counts, MSES median ratio 1.324651, catastrophic rate 0.422446, seven D2 abstentions, D2
40/43 catastrophic evaluations, mechanism slope 1.022842/R2 0.810843, paired block-bootstrap intervals, RHC
selection frequencies, and RHC summary statistics agree with the committed tables/evidence JSON at the stated
rounding. All 16 local Markdown evidence links in the specified manuscripts resolve. The literature-boundary
URLs checked for Causal Q-Aggregation, the relative-error evaluator, P-learner, and the 2026 deconditional GP
paper resolve to papers whose abstracts support the narrow positioning used in the document.

### P4. Existing test suite passed, but does not cover the findings below

The current environment collected 36 tests and the full suite passed. Passing tests do not contradict the
adversarial failures below: `tests/test_kernel_ustat.py:8-24` verifies only equality to the naive U-statistic,
and `tests/test_mvopm_invariants.py:108-115` checks fitted h predictions under permuted input order, not the
formal p-value's invariance to library composition.

## Findings

### D-01 — HIGH — The kernel bridge training loss is indefinite and can be unbounded below

**Code:** `opm/bridges/kernel_moment.py:37-42`, `151-174`  
**Manuscripts:** `docs/技术全志_1_数学与实现.md:76-98`, `187-204`;
`docs/两天历程_OPM到MVOPM.md:60-78`  
**Test gap:** `tests/test_kernel_ustat.py:8-24`

The population RKHS moment norm is nonnegative, but the code trains by directly minimizing the unbiased
finite-sample U-statistic after removing the kernel diagonal:

`vals @ (K - diag(K)) @ vals / (B(B-1))`.

`K-diag(K)` is generally indefinite. A two-observation adversarial check with a Gaussian-kernel off-diagonal
value 0.6065307 has eigenvalues `[-0.6065307, +0.6065307]`; for residuals `[c,-c]`, the implemented loss is
`-0.6065307*c^2`: -0.6065 at c=1, -60.65 at c=10, and -6065.31 at c=100. Because h is unbounded, weight decay
and early stopping are heuristics, not a lower-bound proof. The convergence audit records only the positive
held-out RFF residual named `optimizer_best_resid`, not the actual training U-loss, parameter norms, or
prediction extrema, so the final corpus cannot establish that the pathological direction was never used.

This does not falsify the stored numerical results, but it invalidates the manuscript implication that the
unbiased U-statistic can simply be substituted as a safe empirical minimization objective. It also weakens
fairness claims involving the kernel candidate and interpretations of solver switching.

**Fix:** train on a provably nonnegative V-statistic/RKHS squared norm, a regularized min-max/GMM objective, or
another objective with an explicit lower bound; reserve the unbiased U-statistic for evaluation. Add tests for
nonnegativity/boundedness of the optimized objective, gradient-scale stress, parameter/prediction norms, and
training-loss traces. A changed solver requires new, clearly post-Round-2 experiments; it cannot silently
replace the frozen confirmatory run.

### D-02 — HIGH — Formal candidate p-values depend on which other candidates are present

**Code:** `opm/validation/moment_tests.py:142-156`  
**Related claim:** `opm/candidates/library.py:8-10`; `docs/mvopm_round2/ROUND2_PREREG.md:76-82`  
**Test gap:** `tests/test_mvopm_invariants.py:108-115`, `192-200`

`test_many_candidates` sorts the current prediction dictionary, enumerates it, and assigns a bootstrap seed as
`base_seed + 10000*i`. Therefore the same candidate/data/predictions/statistic receives a different multiplier
stream when a lexicographically earlier candidate is added or removed.

Adversarial reproduction with identical exact-bridge predictions and identical observed statistics:

- `sieve1_sieve1` alone: h seeds `[77,79]`, `p_h=[0.17,0.62]`, `p_q=[0.81,0.32]`;
- after adding `kernel_kernel`: h seeds `[10077,10079]`, `p_h=[0.26,0.64]`, `p_q=[0.83,0.37]`;
- observed h statistics were exactly equal; p-values were not.

This matters directly to I2's restricted libraries and to any future library ablation. With only 999 draws and
thresholds 0.0125/0.025, Monte Carlo variation can flip screen membership near a boundary.

**Fix:** derive streams from a frozen global candidate-name mapping (or a stable cryptographic name hash), not
the current subset index. Prefer common multiplier draws across candidates for lower-noise comparisons. Add a
test that adding/removing unrelated candidates leaves the target candidate's statistic, seed, p-value, and
screen decision unchanged.

### D-03 — HIGH — RHC effect CIs are post-selection full-OOF CIs, not nested-selection inference

**Code:** `opm/experiments/mvopm_round2/studies.py:162-209`, especially `170-190`, `191-208`;
`opm/experiments/mvopm_round2/core.py:41-98`; `opm/estimator/tau_head.py:102-120`  
**Report:** `results/mvopm_round2/FINAL_REPORT.md:205-225`

RHC computes both a full-sample OOF MSES decision and `nested_mses`. But the returned `ate`, `ate_lo`, and
`ate_hi` come from `rows[selected]` for the full-sample OOF-selected candidate. The nested estimator's ATE is
discarded; only its fold choices are saved. The candidate's moment tests and variance selection use the same
OOF population from which its mean and naive `mean +/- 1.96*sd/sqrt(n)` CI are reported.

Thus the statements that all 30 “95% CIs” exclude zero are descriptive post-selection intervals, not validated
selective or nested cross-fit confidence intervals. The report carefully separates full-sample and outer-fold
selection frequencies, but it does not disclose this inferential distinction.

**Fix:** preserve/report the nested estimator's ATE and a variance calculation justified for fold-wise adaptive
selection, or use an honest selection/evaluation split/repeated cross-fitting with an explicit theorem. Until
then, relabel existing RHC intervals “naive candidate-wise OOF intervals after adaptive selection” and avoid
coverage/95%-confidence language.

### D-04 — HIGH — Aggregation is not fail-closed against stale, swapped, or wrong-code results

**Code:** `scripts/cluster/aggregate_round2.py:37-73`, `114-146`  
**Report claim:** `results/mvopm_round2/FINAL_REPORT.md:25-45`

For a manifest path, the aggregator checks only existence and `status == ok`, then labels the row with study,
config, and seeds copied from the *expected manifest*, not validated from the raw record. It never verifies:

- `prov.config == expected`;
- `array_task_id` corresponds to the manifest row;
- provenance seeds/study/dispatch match;
- `git_commit` equals the frozen checkpoint;
- result-level study/seed agree;
- output path is unique or safe;
- a manifest or result content hash.

A stale or swapped `status=ok` file can therefore be silently relabeled as the expected task. The current corpus
passed my independent full validation, so this is a latent fail-closed defect rather than evidence of actual
contamination.

**Fix:** make all mismatches fatal by default, report them separately, and hash the canonical manifest plus every
raw record. Add a deliberately swapped-result test that aggregation must reject.

### D-05 — HIGH — The execution environment is neither pinned nor captured

**Code/config:** `pyproject.toml:5-21`; `scripts/cluster/worker.py:21-27`, `86-103`  
**Report:** `results/mvopm_round2/FINAL_REPORT.md:25-32`

Dependencies such as NumPy, pandas, SciPy, scikit-learn, and matplotlib are unpinned; PyTorch is only
`torch>=2.2`. No lockfile/container digest is committed. Raw provenance records the short git commit, hostname,
and thread environment, but not Python/package/BLAS versions, CPU model, OS, CUDA determinism, locale, manifest
hash, or dirty-worktree status. `_git_commit()` can report a clean-looking checkpoint while uncommitted source
changes are being executed.

The model-fitting results cannot be independently reproduced from git alone with a defined software stack.
This is especially material for neural optimization, BLAS-backed sieve solves, bootstrap numerics, and pandas
aggregation.

**Fix:** commit a lockfile plus environment/container specification; record full commit, dirty diff hash/status,
manifest SHA-256, interpreter and package versions, BLAS backend, CPU/GPU, and determinism flags in each run or
a signed run-level environment record.

### D-06 — MEDIUM — Result writes are non-atomic and duplicate workers can race

**Code:** `scripts/cluster/worker.py:127-135`; `scripts/cluster/launch.py:23-37`, `69-86`

The worker writes canonical `result.json` directly. Process/node failure can leave a partial JSON; two workers
for the same row can race while archiving and overwriting it. Resume will rerun a malformed file, but no lock,
temporary-file `fsync`, atomic rename, or checksum prevents a valid prior result from being corrupted.

**Fix:** write to a same-directory temporary file, flush/fsync, atomically rename, create a per-row lock, and
store a content hash. Preserve every attempt under unique attempt IDs before promoting one canonical record.

### D-07 — MEDIUM — “ATE error” uses inconsistent truth semantics across DGPs

**Code:** `opm/dgp/synthetic_main.py:53-60`, `103-109`; `opm/dgp/bridge_complexity.py:81-93` versus
`opm/dgp/nonlinear_bridge.py:51-60`; `opm/eval/oracle.py:23-24`  
**Prereg/report:** `docs/mvopm_round2/ROUND2_PREREG.md:130-139`;
`results/mvopm_round2/FINAL_REPORT.md:99-130`

S1/S2, HAMD semi-synthetic, and bridge-complexity tasks define `ate_true` as the realized sample average of
`tau(X_i)`. D2 nonlinear defines it as the population constant 2 even though `tau(X)=2+0.8X_1`. Therefore the
pooled metric mixes empirical-X and population ATE errors without naming the difference.

An adversarial D2 recomputation using empirical-X truth changed the global oracle winner on 2/50 seeds. Among
43 evaluable MSES tasks, median ratio changed from 52.66 to 41.55 and the catastrophic count from 40 to 41. The
D2 failure and overall negative verdict survive, but exact reported ratios/winners depend on inconsistent target
semantics.

**Fix:** freeze one target definition. Ideally retain both `ate_population_error` and
`ate_empirical_X_error`, use one consistently for primary selection, and update all oracle/catastrophic metrics.

### D-08 — MEDIUM — High-degree sieve candidates are frequently numerically clipped/pathological

**Code:** `opm/bridges/sieve.py:91-107`  
**Report:** `results/mvopm_round2/FINAL_REPORT.md:126-149`, `178-188`

Candidate-level scanning found no nonfinite diagnostic values, but hard q clipping is pervasive:

- `sieve2_sieve2`: q reaches exactly 50 in 767/1,520 rows (50.5%); 1,066 rows have q-balance error >0.1;
- `sieve3_sieve3`: q reaches 50 in 1,145/1,520 rows (75.3%); all 1,520 rows have q-balance error >0.1;
- maximum sieve3 q-balance error is 2.224 and maximum pseudo-outcome variance is 2.81e8.

The six-way “every candidate wins” story includes five seed-wise wins by sieve3 and 67 by sieve2. Those are
literal sample-wise argmin results, but not necessarily evidence of six scientifically useful solver regimes.
The library is partly a numerical stress test.

**Fix:** explicitly classify clipped/imbalanced solvers as pathological diagnostics; report winner counts after
predefined q-sanity restrictions as sensitivity (not as replacement confirmation). For future work, scale ridge
penalties coherently, use condition-number diagnostics, and prespecify candidate validity gates.

### D-09 — MEDIUM — Seed-wise “genuine switching” overstates structural switching

**Analysis/report:** `scripts/analyze_round2.py:311-371`;
`results/mvopm_round2/FINAL_REPORT.md:7-10`, `134-149`

The switching claim counts each seed's smallest realized ATE error as an oracle winner and calls a change in that
argmin across paired n/noise tasks a transition. In R, 37.5% of best-versus-second gaps are <0.005, 59.8% are
<0.01, and 27.2% have a relative gap <10%. The 600 seed-wise winner frequencies contain all six candidates, but
regime-level mean-risk winners contain only four: kernel+sieve1 (12 regimes), kernel+kernel (6), sieve1+kernel
(1), sieve2 (1). No uncertainty/test distinguishes structural changes from ranking noise.

**Fix:** retain the literal seed-wise table but call it “sample-wise oracle argmin instability.” Add regime mean
risk, paired loss differences with block uncertainty, practical-equivalence bands, and winner stability under
the empirical-X/population target sensitivity before using “genuine structural switching.”

### D-10 — MEDIUM — Formal screen calibration with fitted nuisances is not established

**Code:** `opm/validation/oof.py:14-90`; `opm/validation/moment_tests.py:49-99`;
`opm/validation/selector_v2.py:24-36`  
**Claims:** `docs/mvopm_round2/ROUND2_PREREG.md:48-74`;
`results/mvopm_round2/FINAL_REPORT.md:190-203`

The multiplier bootstrap treats centered observation moments as if the OOF nuisance predictions were fixed and
row-independent; overlapping cross-fit training sets and nuisance estimation are not represented in the
bootstrap. Existing tests check determinism/formulas, not size or power under fitted bridge classes.

As a diagnostic, exact population bridges on 60 samples had zero candidate false rejections. But when a
degree-2 sieve capable of representing the finite-proxy bridge was OOF-fitted, q-side p<0.025 occurred in 80%
at n=1,000, 47.5% at n=4,000, and 12.5% at n=16,000 (20 replications per n in the scaling check); the union-null
candidate screen rarely rejected because h usually passed. This may reflect finite-sample fitted-bridge error
rather than a mathematically invalid test, but it demonstrates that the p-values are not calibrated as tests of
“candidate class adequacy.” The manuscript sometimes slides between fitted-function compatibility and library
adequacy.

**Fix:** define the precise null (fixed fitted function versus model class/existence), supply a nuisance-aware
calibration argument, and add Monte Carlo size/power tests with exact and correctly specified estimated bridges.
Keep the final report's narrow falsification wording.

### D-11 — MEDIUM — Oracle isolation is a convention, not a physical security boundary

**Code:** `opm/data/views.py:15-34`, `48-66`  
**Test:** `tests/test_mvopm_invariants.py:35-51`, `149-153`

The view removes a blacklist of exact metadata keys; truth under a new alias remains in `meta`. Arrays are direct,
mutable references despite the “read-only” class description. The import test checks only three Round-1 modules,
not `selector_v2`, `oof`, or `nested_mses`. Current Round-2 selection code does not actually read oracle truth,
so this is a defense/claim gap, not observed leakage.

**Fix:** construct an allowlisted immutable metadata object, expose read-only array views/copies, and recursively
audit imports/call graphs for all Round-2 selection modules. Add renamed-key smuggling tests.

### D-12 — LOW — `summary_selectors.csv` is not bitwise deterministic

**Code:** `scripts/cluster/aggregate_round2.py:124-137`

Independent reconstruction produced identical keys, means, medians, and quantiles, but a different SHA-256 for
`summary_selectors.csv`. The only differences were 1e-16-scale SD tails (maximum absolute difference
2.27e-13), plausibly BLAS/reduction behavior. It has no effect on the report.

**Fix:** if byte reproducibility is claimed, round output columns to a documented precision and pin the numeric
stack; otherwise claim numerical, not byte-level, determinism for derived summaries.

### D-13 — HIGH (manuscript package) — The two technical volumes are superseded but presented as complete/current

**Files:** `docs/技术全志_1_数学与实现.md`, `docs/技术全志_2_实验与结果.md`

Material conflicts include:

- Volume 2 `:188` says final seeds are “未跑”; Round 2 has now completed 6,820 tasks.
- Volume 2 `:262-263` summarizes the project as pilot HOLD plus an encouraging near-oracle biasvar direction and
  stable 100%-sieve1 RHC. Round 2 found MSES unsupported, biasvar not generally successful, and very different
  RHC selection frequencies.
- Volume 1 `:243-246` documents obsolete RNG code (`seed+17*si+1`, global NumPy seeding), whereas current code
  uses fixed solver-name offsets and `torch.random.fork_rng` (`opm/candidates/library.py:32`, `75-89`).
- Volume 1 `:283-291` documents the Round-1 manifest/aggregator and 21 tests, not the Round-2 pipeline and current
  36 tests.
- Volume 1 `:89-98`, `187-204` presents the optimized U-statistic without the indefinite-objective caveat in
  Finding D-01.

**Fix:** add an unmistakable superseded/historical banner at line 1 of each volume, then produce revised volumes
covering Round 2, the negative verdict, current code, and this re-audit. Do not silently overwrite historical
numbers without a changelog.

### D-14 — HIGH (manuscript package) — “Two-day journey” ends before the confirmatory study and now misdirects readers

**File:** `docs/两天历程_OPM到MVOPM.md`

The title and opening claim a complete narrative (`:1-12`), but the story ends at the pilot/fresh-seed stage.
Lines `:229-243` characterize bias+variance as near-oracle and “de-risked”; lines `:252-255` recommend running the
pre-registration that has since been run and failed. Lines `:196-199` call the old 25-split, 100%-sieve1 RHC
result stable, conflicting with the 30-split Round-2 full/nested decisions and instability discussion.

**Fix:** mark it “historical through Round 1/pilot” or add a dated Round-2 epilogue with the `NOT_SUPPORTED`
verdict, D2 failure, revised RHC results, and the distinction between diagnostic support and selector failure.

### D-15 — MEDIUM (manuscript) — Final report omits implementation defects and overstates some evidence

**File:** `results/mvopm_round2/FINAL_REPORT.md`

The stored numbers are accurate, but revisions are needed:

- `:85-97` calls candidate training fairer without disclosing the indefinite kernel training objective.
- `:134-149` calls six-candidate seed-wise wins “genuine” switching without practical-equivalence/gap analysis.
- `:178-188` explains D2 via screen behavior but omits the pervasive sieve2/3 clipping/imbalance diagnostics.
- `:205-225` reports “95% CIs” without labeling them post-selection naive candidate intervals.
- `:25-45` implies a fail-closed provenance chain that the aggregator does not enforce, although this re-audit
  independently verified the current corpus.

The first-line `NOT_SUPPORTED` verdict should remain. These issues mostly strengthen caution rather than rescue
MSES.

### D-16 — MEDIUM (manuscript/theory) — CI and efficiency language is too categorical

**Files:** `docs/技术全志_1_数学与实现.md:71-74`, `114-142`;
`opm/estimator/cate_inference.py:1-12`, `46-53`;
`docs/mvopm_round2/CATE_SELECTION_ANALYSIS.md:27-42`

The Round-2 CATE boundary document is appropriately cautious. The older volume and code comments are not: they
state that cross-fitting/Neyman orthogonality guarantees nominal ATE/CATE coverage and call the pseudo-outcome
contrast the efficient influence function without spelling out nuisance rates, regularity, selection, repeated
cross-fit dependence, bandwidth bias, or proximal semiparametric conditions. A wider local-linear bandwidth is
not generally bias-free merely because the specific E10 truth is linear.

**Fix:** align older theory prose with the cautious Round-2 boundary; state sufficient conditions and distinguish
simulation behavior from a proved coverage theorem.

### D-17 — MEDIUM — RHC proxy assignment still carries an unresolved verification marker

**Code:** `opm/data/rhc.py:1-10`, especially `:9`; `DECISIONS.md:14-15`  
**Manuscripts:** `results/mvopm_round2/FINAL_REPORT.md:205-225` and older RHC claims

The loader itself says `VERIFY against the published paper`, and the decisions log records that the verification
comment was intentionally left. Yet multiple manuscripts call the split faithful/real proxies. This is a source
validation gap for the only real-data anchor.

**Fix:** cite the exact paper/table/page and preprocessing provenance for `V=(pafi1,paco21)` and
`W=(ph1,hema1)`, record a dataset checksum and column dictionary, or downgrade the result to a provisional
real-data demonstration.

## Required remediation order

1. **Do not change or rerun the frozen Round-2 corpus.** Preserve it as negative evidence.
2. Fix D-04/D-05/D-06 first so future results are fail-closed, atomic, and environment-reproducible.
3. Fix and test D-01/D-02 on a new development branch; any new solver/selector evidence must use new seeds and a
   new protocol.
4. Correct RHC inference/labels (D-03/D-17) before any public real-data claim.
5. Reframe switching and numerical-pathology statements (D-08/D-09) and document target semantics (D-07).
6. Add calibration/security tests (D-10/D-11) and then revise the final report.
7. Mark both technical volumes and the two-day journey as historical/superseded before distributing the repo.

## Minimal new tests

- optimized kernel loss lower-bound/nonnegative-objective test and scale-stress test;
- same-candidate p-value invariance under candidate-set expansion/reordering;
- aggregate rejects swapped result, wrong seed/config/commit, dirty execution, and duplicate path;
- atomic-write interruption and duplicate-worker race test;
- exact and fitted-nuisance screen size/power Monte Carlo;
- population versus empirical-X ATE target consistency test;
- nested RHC output retains nested ATE/variance and cannot substitute global OOF CIs;
- q clipping, condition number, balance, and pseudo-outcome variance gates;
- full Round-2 oracle import/metadata-smuggling audit;
- Markdown stale-status and evidence-link check in CI.

## Bottom line

The evidence files are not fabricated, missing, or misaggregated: the entire raw chain was independently
reconstructed and the final negative MSES verdict is credible. The codebase nevertheless contains real design
defects that existing tests/reviews did not catch, and the manuscript set mixes a careful Round-2 final report
with older documents that now state obsolete conclusions. Treat Round 2 as a well-preserved negative experiment,
not as a release-ready validation of the solver, formal screen, RHC inference, or complete paper package.
