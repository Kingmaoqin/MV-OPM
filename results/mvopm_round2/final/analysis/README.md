# Round-2 analysis artifacts

All files here are deterministic reconstructions from `final/raw.csv` and
`final/raw_candidates.csv`, which are themselves reconstructed from the immutable per-task
`result.json` records listed in the final manifest. The aggregation audit gives the exact
complete/missing/failure counts.

The unprefixed selector metrics are the primary nested outer-fold evaluations. Columns beginning
`global_oof_` and candidate flags named `selected_by_*` are descriptive full-sample OOF
diagnostics; `nested_selected_fraction[*]` records the confirmatory outer-fold choices. The
`biasvar_mult` and `biasvar_add` rows remain `EXPLORATORY_POSTHOC_FROM_ROUND1` throughout.

For the non-oracle RHC repeated-split study (J2), candidate-frequency tables explicitly separate
the full-sample OOF deployment screen/decision from the four outer-fold choices stored in
`nested_selected_by_fold`. The reported deployment ATEs are not an independent validation sample.

`tables/` contains mechanism, candidate, selector, screening, switching, robustness, abstention,
and RHC summaries. `figures/` contains the six claim-directed figures required by the Round-2
protocol. `EVIDENCE_SUMMARY.json` is a compact machine-readable basis for the final report.
