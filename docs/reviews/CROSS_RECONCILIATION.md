# Cross-Reconciliation — Reviewer A × Reviewer B (MV-OPM Phase 2)

Both reviews were isolated passes (independent reviewer-agents unavailable; documented fallback).
Each reviewer then inspected the other's findings.

## A reads B
- B-1 (determinism) is also a *methodological* validity issue: non-reproducible candidate fits
  would confound any selection comparison. A concurs the fix is necessary and sufficient; the
  invariant test guards it.
- B-3 (index audit) directly enforces A-4 (nested validity). A confirms the audited flags match
  the methodological requirement that OUTER_TEST never touches selection/bank/standardization.
- No B finding is only-documented-not-implemented: each has a passing test.

## B reads A
- A-1 (ATE-error as primary target) has no engineering blocker: both `pehe::*` and `ate::*` are
  emitted per row, so the analysis can select the primary target at aggregation without rerunning.
- A-2 (normalization) is implemented exactly as documented (within-arm sd), not just described.
- A-5 (oracle isolation) is enforced in code (view + import guard test), not merely asserted.

## Issue ledger
| ID | Raised by | Severity | Status | Fix | Verification |
|----|-----------|----------|--------|-----|--------------|
| A-1 | A | CRITICAL(design) | RESOLVED | Primary target = ATE error (theory-aligned); PEHE secondary; product score unchanged | frozen pre-pilot; both targets computed |
| A-2 | A | MAJOR | RESOLVED | within-arm sd normalization, documented | moments.py docstring + test |
| B-1 | B | CRITICAL | RESOLVED | per-solver torch/np seed in CandidateSet.fit | test_candidate_order_invariance, test_seed_reproducibility |
| B-2 | B | MAJOR | RESOLVED | shared deterministic bank | test_common_rff_bank_*, test_moment_score_matches_manual_formula |
| B-3 | B | MAJOR | RESOLVED | per-fold index audit + adversarial test | test_nested_split_audit_clean, test_adversarial_outer_test_leak_is_detected |
| B-6 | B | MAJOR | RESOLVED | provenance in worker + aggregate completeness | worker.py/aggregate.py |

**No CRITICAL/MAJOR item remains unresolved. Both reviewers: PASS to development pilot (Phase 3).**

## Prespecified Go/No-Go gate (frozen BEFORE inspecting the pilot aggregate)
Evaluated on the pilot (dev seeds), PRIMARY target = **ATE error**:
1. Rank association (Spearman of product score vs ATE error, across candidates) is **positive on
   average and > 0.2 in ≥ 3 of 4 pilot scenario families** (S1, S2, nonlinear, HAMD).
2. Product-selector **median oracle-ratio (ATE) < the fixed-kernel median oracle-ratio** overall.
3. Product-selector **catastrophic-selection rate (ATE-err > 1.5× oracle) not systematically
   worse than fixed-kernel**, and no whole scenario family collapses.
4. The mechanism study (A) shows the observed moment product **increases with the true bridge-error
   product** (positive Spearman) and single-side corruption behaves per Theorem 2.
If (1)+(4) fail → **STATUS = HOLD** + failure analysis (do not invent new scores). One prespecified
secondary comparison among {product, h, q, sum, max} is permitted; PEHE is reported but is not the
gate.
