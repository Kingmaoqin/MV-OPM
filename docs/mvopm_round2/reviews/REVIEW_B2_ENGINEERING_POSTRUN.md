# Reviewer B2 engineering/provenance post-run audit

**Final verdict: PASS**

**Audit date:** 2026-08-13
**Scope:** final manifest, immutable per-task records, reconstructed task/candidate CSVs,
analysis tables, evidence summary, and six figures.
**Scientific-output changes made by this reviewer:** none.

The confirmatory simulation evidence has a complete, reproducible provenance chain. I found no
missing or substituted task, no failed task hidden by aggregation, no seed overlap, and no mismatch
in the sampled JSON-to-CSV reconstruction. The initial audit found one minor reconstruction issue
confined to the secondary J2 RHC repeated-split display. It has now been corrected and independently
re-audited: full-sample deployment screening, full-sample deployment selection, and outer-fold
selection are separately named and exactly reconstructed in the candidate CSV, RHC table, Figure
6, and analysis README. No open engineering finding remains.

## Findings

### MAJOR findings

None.

### RESOLVED MINOR-1: J2 global deployment indicators were mislabeled as nested-fold frequencies

In the original J2 implementation, `scripts/cluster/aggregate_round2.py` entered its
`elif result.get("study") == "J2"` branch and wrote the top-level full-sample OOF deployment screen
and decision into `nested_screen_survival_fraction` and `nested_selected_fraction[MSES]`. The
original analysis then averaged those columns in `rhc_candidate_frequencies.csv` and used them for
the left panel of Figure 6.

Those quantities are valid deployment summaries, but they are not the nested decisions stored in
`result["nested_selected_by_fold"]`. The difference is material enough to require explicit labels:

| Candidate | Full-sample deployment frequency in the original display | Nested-fold selection frequency (120 outer folds) |
|---|---:|---:|
| `kernel_kernel` | 22/30 = 0.7333 | 64/120 = 0.5333 |
| `kernel_sieve1` | 2/30 = 0.0667 | 9/120 = 0.0750 |
| `sieve1_kernel` | 6/30 = 0.2000 | 40/120 = 0.3333 |
| `sieve1_sieve1` | 0/30 = 0.0000 | 7/120 = 0.0583 |

Only 51/120 = 0.425 of the nested-fold choices equal their split's full-sample deployment choice.
All 30 source records have `nested_status == "ok"`, so the difference is not missingness.

The original audit required these reporting/reconstruction fixes:

1. Preserve the current J2 indicators with explicit names such as
   `global_oof_screen_survival_indicator` and `global_oof_selected_indicator`.
2. Reconstruct true nested-fold selection frequencies from `nested_selected_by_fold` under
   separately named columns.
3. Label the RHC frequency table and Figure 6 bars as global deployment or nested-fold quantities;
   ideally report both.
4. Correct the analysis README statement that `nested_selected_by_fold` is used: the original
   analysis script did not read that field.

This was MINOR because J2 is a secondary, non-independent repeated-split analysis with no oracle
truth, its top-level ATEs are already explicitly described as deployment ATEs, and the issue does
not alter any confirmatory simulation selector metric.

**Resolution verification.** The corrected aggregator now writes
`deployment_screen_survival`, `deployment_selected[MSES]`, and the true
`nested_selected_fraction[MSES]` separately. I checked all 30 J2 source records and all 120
candidate rows: every corrected value matches `survivors`, `selected`, and
`nested_selected_by_fold` exactly; there were zero mismatches. Each split's deployment-selection
indicators sum to one, each split's nested fractions sum to one, and every nested fraction is a
multiple of 1/4. The obsolete `nested_screen_survival_fraction` is null for every J2 row rather
than being populated with a global indicator. All three new fields are present for all 120 rows.

The corrected RHC table reports the three quantities under explicit column names and reproduces
the source counts exactly. Its values are deployment-screen 23/30, deployment-selection 22/30,
and outer-fold selection 64/120 for `kernel_kernel`; 22/30, 2/30, and 9/120 for
`kernel_sieve1`; 30/30, 6/30, and 40/120 for `sieve1_kernel`; and 30/30, 0/30, and 7/120 for
`sieve1_sieve1`. A fresh analysis run reproduced the corrected table byte for byte and Figure 6
pixel for pixel. Visual inspection confirms its three bar series are labeled "full-sample screen
survives", "full-sample deployment", and "outer-fold choices". The README now accurately states
the distinction. MINOR-1 is closed.

## Exact corpus and manifest checks

I independently parsed the final manifest and seed registry rather than relying on the generated
aggregation audit.

* Manifest: 6,820 rows, 6,820 unique `(study, config_id, scientific_seed)` keys, and 6,820 unique
  output paths; every path exists.
* Task CSV: 6,820 data rows and no duplicated task key.
* Candidate CSV: 10,020 data rows and no duplicated
  `(study, config_id, scientific_seed, candidate)` key.
* Status audit: the generated audit records 6,820 `ok`, zero missing, and zero failed. Sampled
  source records additionally had `status == "ok"` and `exit_code == 0`.
* Every manifest row explicitly contains scientific, bootstrap, split, and RFF-bank seeds.
  Bootstrap seeds are unique across all 6,820 tasks; bank seeds are also unique across all 6,820.
* All split seeds equal their registered scientific seed, as frozen.
* The eleven study-specific scientific-seed sets are pairwise disjoint, and their union is
  disjoint from development seeds 220000--220007.
* Every configuration uses exactly its registered study seed set. Counts are A2 25 x 200 = 5,000;
  B2/C2/D2/E2 each 1 x 50; R 20 x 30 = 600; F2 4 x 30 = 120; G2 16 x 30 = 480;
  H2 8 x 30 = 240; I2 3 x 50 = 150; J2 1 x 30.
* Source records inspected across the manifest identify checkpoint `aa983a2`, host
  `dhai.bme.e.uh.edu`, and the four two-thread caps. The embedded source `config` equals the
  corresponding manifest object in the audited records.

File hashes at audit time:

| Artifact | SHA-256 |
|---|---|
| `manifest_final.jsonl` | `53acb6c62b0a24f1ff7dc236244f9ca6c287e767c8ed52c1ccc596cf5162820d` |
| `final/AGGREGATION_AUDIT.json` | `0b96b4d93bb38df788f069d44f22a56dd3e7de95948ccd790f8a3d3cecdc42a3` |
| `final/raw.csv` | `308a228d7afe63a28e37e211b291825c3095a5c2f4d3917ecca3627edb5f0130` |
| `final/raw_candidates.csv` | `45b76de690cfb020bdb73938136756f9fbad080410e7404493bc3f21a86c0746` |
| `final/analysis/EVIDENCE_SUMMARY.json` | `173a87c91c78c4a2a1107304557a14f2530ca200ab5cd464c689be3bd15b0ffe` |
| `final/analysis/tables/rhc_candidate_frequencies.csv` | `b6f7b586772ed1bf71d8cff1d11d6b3aa210fbfb76bb2ff27f148722558792e7` |
| `final/analysis/figures/figure6_rhc.png` | `c08469dad32378450175b45295caca8664d7134523da0647d2e589830efcc610` |
| `final/analysis/README.md` | `7babc4d54e6964e536b2a20c020e98f767e142c1ccb356ed268b7f406a343de9` |

## JSON-to-CSV trace

I selected a deterministic stratified sample using seed 20260813: the first, middle, and last task
of every one of the 81 configurations, plus a fixed pseudorandom fill to 400 tasks. The sample
covers every study and every configuration (A2 203, B2 3, C2 4, D2 3, E2 5, R 73, F2 15, G2 55,
H2 25, I2 11, J2 3).

For each sampled task I rebuilt the task row from:

* the manifest seed/key fields;
* record-level provenance (`git_commit`, host, runtime, status, exit code, array index, timestamps,
  and warnings);
* every non-private result field; and
* every flattened `selector::*` field.

For every sampled candidate I also rebuilt all raw candidate fields, all finite-list min/mean/max
scalars, and all nested screen/selection fractions from the exact stored outer folds. In total,
144,361 flattened field comparisons across 400 task rows and 1,099 candidate rows produced zero
provenance, task-row, or candidate-row mismatches (floating-point tolerance 2e-13).

Representative end-to-end traces include:

* A2 `rh0.4_rq0.4_n4000`, seed 200199, array row 4999:
  `predicted_bias=-0.0630871185`, `incremental_ate_bias=-0.0731237239`,
  `D_h=0.0209829`, `D_q=0.0181470`, and `moment_product=0.0003807` agree between source and
  `raw.csv`.
* B2 `S1_n4000`, seed 201000, array row 5000: source and CSV agree on nested MSES error
  0.0423159, oracle ratio 0.991878, mean survivor count 4.25, and the four outer-fold choices.
  Its six candidate rows, formal p-values, screen indicators, variance, errors, and nested
  fractions also match.
* R `stress_Q_n3000_low_p40`, seed 205029, array row 5739: source and CSV agree that the fixed
  candidate oracle is `kernel_sieve1`, nested MSES selects `sieve1_kernel` in all four folds,
  MSES error is 0.2880646, oracle ratio is 1.0658955, and mean survivor count is 3.5.
* H2 `N_n4000_low`, seed 208019, array row 6599: source and CSV agree on four-fold
  `kernel_kernel` selection, error 0.0105159, oracle ratio approximately 1, and all six nested
  candidate fractions.
* J2 seed 210000 agrees exactly on its full-sample selected candidate `kernel_sieve1`, its stored
  nested choices (four `kernel_kernel` choices), and deployment ATE/CI
  -1.3913422 [-2.1088158, -0.6738685]. Its corrected candidate rows record deployment and nested
  choices in separate fields.

## CSV-to-table and figure checks

I loaded only `final/raw.csv` and `final/raw_candidates.csv` into a fresh temporary analysis
directory and regenerated all artifacts.

* In the initial audit all 15 expected table files were regenerated: twelve were byte-identical,
  while three differed only in last-bit decimal serialization (at most the order of 1e-16). After
  the MINOR-1 correction and final artifact regeneration, a new clean reconstruction made all 15
  tables byte-identical to their stored versions.
* All six current regenerated PNGs are pixel-identical to the stored figures. This includes the
  corrected three-series RHC Figure 6 and the full-tail Figure 5.
* The independently regenerated `EVIDENCE_SUMMARY` matches the stored JSON with no difference
  above 1e-13.

Direct table recomputations confirm, among other values:

* Core-plus-switching scope contains 800 tasks. MSES is evaluable on 793, has median oracle ratio
  1.3246513, catastrophic rate 0.4224464, top-1 rate 0.2834375, top-2 rate 0.505, mean survivor
  count 4.6178125, and oracle survival 0.86125.
* The corresponding product, variance-only, fixed-sieve1, and fixed-kernel median oracle ratios
  are 1.6972435, 1.3001032, 1.7504820, and 1.4763111, respectively; their table values match raw
  nested selector columns.
* A2 cell `(r_h,r_q)=(0.4,0.4)` has 200 replicates, predicted mean bias -0.0630871 and observed
  mean incremental bias -0.0646355.
* R stress-Q/p40 has modal oracle `kernel_sieve1` in 27/30, modal nested MSES
  `sieve1_kernel`, and MSES-oracle match 1/30; source rows and switching table agree.
* I2 `N_kernel_overregularized` has 35/50 = 0.70 abstention and mean survivor count 0.72.
* The stored J2 ATE summary correctly computes the top-level deployment estimates: mean
  -1.3838290, SD 0.2993782, 30/30 negative, and 30/30 deployment CIs excluding zero. This ATE
  calculation was unaffected by the resolved selection-frequency issue.

Figure inputs are traceable as follows: Figure 1 uses A2 raw rows and its cell table; Figure 2
uses B2--E2/R candidate moment and variance columns; Figure 3 is fixed in advance to B2 seed
201000; Figure 4 uses the 20 reconstructed R regimes; Figure 5 uses unprefixed (nested) selector
oracle ratios from B2--E2/R; and corrected Figure 6 uses J2 deployment ATEs plus explicitly
separated deployment-screen, deployment-selection, and outer-fold candidate frequencies.

## Failure and provenance handling

The aggregator does not silently convert failures to successes. Missing output paths are listed in
`missing`; records whose status is not `ok` are listed in `failures`; neither is added to the task
or candidate tables. The downstream analysis refuses missing or failed final rows unless the
operator explicitly supplies `--allow-incomplete`. Because the final audit is complete, that gate
was not bypassed. Warnings and execution provenance remain in both reconstructed CSVs.

## Final engineering assessment

The primary Round-2 confirmatory simulation results have an auditable manifest-to-source-to-table
chain and may be interpreted as generated. The corrected RHC artifacts now distinguish global
deployment from outer-fold selection and pass exact reconstruction checks. No rerun or scientific
method change is indicated by this engineering audit, and no engineering/provenance finding
remains open.
