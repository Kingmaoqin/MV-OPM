# E5 — Semi-synthetic on real HAMD covariates

**Overall: PASS**  (2/2 acceptance criteria met)

## Acceptance criteria

- ✅ PASS — **OPM beats every ignorability baseline on PEHE (Wilcoxon, BH<0.05)**: S-learner: mean_diff=-0.054 adjp=0.0195 ✓; T-learner: mean_diff=-0.474 adjp=0.0026 ✓; X-learner: mean_diff=-0.266 adjp=0.0026 ✓; DR-learner: mean_diff=-1.314 adjp=0.0026 ✓; CausalForest: mean_diff=-0.083 adjp=0.0112 ✓; R-learner: mean_diff=-0.436 adjp=0.0026 ✓; TARNet: mean_diff=-0.290 adjp=0.0026 ✓; DragonNet: mean_diff=-0.281 adjp=0.0026 ✓
- ✅ PASS — **OPM competitive with sieve-PDR (kernel-vs-sieve gap reported, within 20%)**: OPM=0.603 vs PDR-sieve=0.562 (gap +7.2%)

## PEHE on semi-synthetic HAMD covariates (mean ± sd over seeds)

| method       |   pehe_mean |   pehe_sd |
|:-------------|------------:|----------:|
| OPM          |      0.6026 |    0.0401 |
| S-learner    |      0.6567 |    0.0416 |
| T-learner    |      1.0763 |    0.0522 |
| X-learner    |      0.8687 |    0.0427 |
| DR-learner   |      1.9162 |    1.1358 |
| CausalForest |      0.6856 |    0.0484 |
| R-learner    |      1.0384 |    0.0701 |
| TARNet       |      0.8921 |    0.1218 |
| DragonNet    |      0.8835 |    0.1125 |
| POR-sieve    |      0.6778 |    0.0479 |
| PIPW-sieve   |      0.6533 |    0.1893 |
| PDR-sieve    |      0.5623 |    0.049  |
| NMMR         |      0.6326 |    0.0389 |

Baselines not run: DFPV; P-learner; DiffPO
