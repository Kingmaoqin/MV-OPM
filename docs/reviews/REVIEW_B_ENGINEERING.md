# Reviewer B — Engineering / Reproducibility (MV-OPM Phase 2)

Isolated review pass (agent spawning unavailable — same fallback as Reviewer A). Scope:
candidate library, validator bank, nested CF indices, worker/launcher/aggregator, tests.

## Findings

- **B-1 [CRITICAL → RESOLVED].** Candidate-order / determinism. The kernel candidate's network
  init draws from torch's GLOBAL RNG, so sequential fits in one process were not reproducible and
  `test_candidate_order_invariance` FAILED. Fixed in `candidates/library.py`: seed
  `torch.manual_seed` and `np.random.seed` per solver before each fit (frozen solver internals
  untouched). Test now passes; `test_seed_reproducibility` passes.
- **B-2 [MAJOR → RESOLVED].** Shared validation bank. `build_bank` is deterministic in a fixed
  candidate-independent seed and built from held-out instruments only; `evaluate_candidate` and
  the exact-DGP study call `discrepancy_from_hq` with the SAME bank across all candidates
  (`test_common_rff_bank_deterministic_and_candidate_independent`, `test_moment_score_matches_manual_formula`).
- **B-3 [MAJOR → RESOLVED].** Split index audit. Outer/inner disjointness, inner⊂outer_train, and
  outer_test∉inner are recorded per fold and asserted by `test_nested_split_audit_clean`. An
  adversarial injection (`test_adversarial_outer_test_leak_is_detected`) confirms the audit flag
  flips when a leak is introduced.
- **B-4 [OK].** Feature standardization for the bank is fit on inner-validation instruments only
  (never outer-test). Candidate solver standardizers are fit on their own train fold.
- **B-5 [OK].** Arm ordering / tensor alignment: predict_h/predict_q return (n,K) columns in arm
  order (`test_arm_order_and_alignment`). q clipping to [0,q_max] preserved from frozen solvers.
- **B-6 [MAJOR → RESOLVED].** Provenance. `worker.py` saves per-run `result.json` with study,
  config snapshot, explicit seed, array index, git commit, hostname, thread count, runtime, exit
  code, and (on error) traceback. Seeds are explicit in the manifest, never derived solely from the
  array index. `aggregate.py` reconstructs raw.csv + per-candidate raw from these files and runs
  completeness checks (missing/duplicate seeds, NaN columns, per-study counts).
- **B-7 [OK].** Concurrency. `launch.py` spawns one isolated subprocess per row, caps concurrency,
  and sets OMP/MKL/OPENBLAS/NUMEXPR threads per worker to prevent BLAS/torch oversubscription;
  resume-aware; failed rows stay visible (rerun the SAME seed).
- **B-8 [NOTE].** Real-data path (Study J) tolerates missing oracle (`has_truth` guard); PEHE set
  to NaN, selection via product moments only, ATE/CI reported.
- **B-9 [NOTE].** No Slurm on host → local manifest+worker+launcher is the array equivalent;
  documented.

## Verdict
No unresolved CRITICAL/MAJOR after B-1/B-2/B-3/B-6 fixes. 21/21 tests pass (9 legacy + 12 MV-OPM).
**PASS to development pilot.**
