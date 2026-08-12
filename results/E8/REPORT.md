# E8 — Diagnostics validity study

**Overall: PASS**  (2/2 acceptance criteria met)

## Acceptance criteria

- ✅ PASS — **|Spearman(rho_min, PEHE)| > 0.4 across corruption levels in >=1 family**: additive:0.5; missing:-0.3
- ✅ PASS — **|Spearman(bridge_resid, PEHE)| > 0.4 across corruption levels in >=1 family**: additive:0.6; missing:-0.3

## Diagnostic–PEHE Spearman correlations (per-level means over seeds)

| family   |   spearman(rho_min,PEHE) |   spearman(bridge_resid,PEHE) |   rho_min(per-point) |   bridge_resid(per-point) |
|:---------|-------------------------:|------------------------------:|---------------------:|--------------------------:|
| additive |                      0.5 |                           0.6 |                0.056 |                     0.264 |
| missing  |                     -0.3 |                          -0.3 |               -0.156 |                     0.066 |

Expected signs: rho_min ↓ (weaker proxies) and bridge_resid ↑ as corruption grows, both tracking higher PEHE. Per-point columns include seed scatter.
