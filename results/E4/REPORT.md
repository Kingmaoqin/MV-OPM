# E4 — Real-data anchor (RHC proximal)

**Overall: PASS**  (3/3 acceptance criteria met)

## Acceptance criteria

- ✅ PASS — **Linear-POR gate: finite, plausible NEGATIVE effect on survival days**: POR ATE=-1.940 days [-2.734,-1.147] (RHC reduces survival — Connors 1996 / Cui 2023)
- ✅ PASS — **OPM proximal CI overlaps linear-POR CI**: OPM ATE=-1.319 [-1.855,-0.782] vs POR [-2.734,-1.147]
- ✅ PASS — **Weights bounded (max q_hat<=50, En[1{T=k}q_k] in [0.9,1.1])**: max_q=9.83; En[1Tq]=0:0.998, 1:0.972

Dataset: RHC, n=5735, K=2 (RHC vs no-RHC), outcome=Y_survival_days_30 (days to death/censor at 30; negative ATE = RHC shortens survival).

## ATE estimates (mean over cross-fit seeds, days)

| estimator         |      ate |   ci_low |   ci_high |
|:------------------|---------:|---------:|----------:|
| linear-POR (gate) | -1.94046 | -2.73407 | -1.14686  |
| OPM proximal      | -1.31858 | -1.8553  | -0.781866 |
| OPM dr_fallback   | -1.0785  | -1.68855 | -0.46844  |

OPM proximal ESS per arm: 0:3443, 1:1796

Proxies present (V=pafi1,paco21; W=ph1,hema1) so the proximal q-bridge's moment identity B2 keeps En[1{T=k}q_k]≈1 by construction — the strict weight band holds here (unlike the no-proxy HAMD secondary anchor).
