# Reviewer B — engineering/reproducibility pre-run audit

**Verdict: PASS_WITH_OPERATIONAL_GATES after fixes.** No remaining major code or reproducibility
finding remained. All operational gates below were completed before final launch.

The isolated reviewer found and prompted fixes for:

* development tasks accidentally referencing final seeds;
* development/final aggregation overwrite risk;
* loss of prior failure attempts on rerun;
* non-nested headline MSES and adaptive-comparator metrics;
* incomplete nested raw artifacts;
* thread caps initialized too late in the convergence driver;
* exact inner indices being dropped from raw nested artifacts;
* an infeasible six-candidate HAMD library instead of the frozen four-candidate subset;
* the need for a clean git checkpoint before final execution.

The reviewer also adversarially verified deterministic torch initialization, candidate-order
invariance, and correct mixed h/q candidate composition. The final checkpoint and verdict are
recorded before final launch.

Final verification covered exact generated-manifest equivalence, reserved development seeds,
999-draw final bootstrap, HAMD feasible library, exact inner/outer indices, shared nested selector
folds/refits, raw nested reconstruction artifacts, old-result immutability, oracle isolation,
RNG isolation, bootstrap determinism, training-only scalers/bandwidths, failure archiving,
phase-scoped aggregation, and thread caps. The full suite passed 36/36.

Operational gates:

1. development smoke: 106/106 successful, zero missing/failures;
2. convergence audit: 32/32 successful, final cap frozen at 300 by observed-data criteria;
3. representative selection benchmark: 17.42 s wall time, 755,424 KiB maximum RSS, 159% CPU
   with two threads; safe launch setting chosen as 12 workers × 2 threads on 48 logical CPUs and
   503 GiB RAM;
4. the source/results checkpoint is committed immediately before final launch.

Minor nonblocking note: numerical moment-test warnings remain reconstructible inside candidate and
nested payloads rather than being duplicated into the worker's top-level warning list.
