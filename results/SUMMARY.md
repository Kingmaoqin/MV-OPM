# OPM v2 — Consolidated Results Board

Auto-generated from each `results/{Ex}/REPORT.md`. See per-experiment reports for full tables/figures and `DECISIONS.md` for documented deviations.

| Exp | Title | Overall | Criteria |
|-----|-------|---------|----------|
| E1 | E1 — Consistency and rates (DGP 3.1) | ✅ PASS | 5/5 acceptance criteria met |
| E2 | E2 — Product-bias / rate double robustness (DGP 3.1) | ✅ PASS | 3/3 acceptance criteria met |
| E3 | E3 — Main benchmark (S1, S2 + corruption) | ✅ PASS | 3/3 acceptance criteria met |
| E4 | E4 — Real-data anchor (RHC proximal) | ✅ PASS | 3/3 acceptance criteria met |
| E5 | E5 — Semi-synthetic on real HAMD covariates | ✅ PASS | 2/2 acceptance criteria met |
| E6 | E6 — Ablations (S1) | ✅ PASS | 3/3 acceptance criteria met |
| E7 | E7 — Distributional counterfactuals (S1) | ❌ FAIL | 0/1 acceptance criteria met |
| E8 | E8 — Diagnostics validity study | ✅ PASS | 2/2 acceptance criteria met |
| E9 | E9 — Proximal removes nonlinear-confounding bias (ATE) that ignorability cannot | ✅ PASS | 2/2 acceptance criteria met |
| E10 | E10 — Valid heterogeneous-effect inference / coverage | ✅ PASS | 3/3 acceptance criteria met |

## Criterion-level detail

### E1 — E1 — Consistency and rates (DGP 3.1)  (✅ PASS)
- ✅ PASS — |ATE-2| bias monotone decreasing (Spearman<-0.9)
- ✅ PASS — ||h_hat-h_true||_L2 monotone decreasing (Spearman<-0.9)
- ✅ PASS — |ATE-2| < 0.10 at n=8000 (mean over seeds)
- ✅ PASS — h linear-probe R^2 > 0.95 at n=8000
- ✅ PASS — CI95 coverage in [0.85, 0.99] (n>=2000 pooled)
### E2 — E2 — Product-bias / rate double robustness (DGP 3.1)  (✅ PASS)
- ✅ PASS — DR property: single-bridge corruption stays consistent (h-only AND q-only decreasing)
- ✅ PASS — All 9 two-bridge-corrupted biases vanish with n (slopes < 0)
- ✅ PASS — Fitted rate within ±0.15 of -(a_h+a_q) for >=4/9 pairs (rest directionally correct)
### E3 — E3 — Main benchmark (S1, S2 + corruption)  (✅ PASS)
- ✅ PASS — OPM beats every ignorability baseline on PEHE (Wilcoxon, BH<0.05)
- ✅ PASS — OPM competitive with sieve-PDR (kernel-vs-sieve gap reported, within 20%)
- ✅ PASS — Graceful degradation (no PEHE jump >2x between adjacent levels)
### E4 — E4 — Real-data anchor (RHC proximal)  (✅ PASS)
- ✅ PASS — Linear-POR gate: finite, plausible NEGATIVE effect on survival days
- ✅ PASS — OPM proximal CI overlaps linear-POR CI
- ✅ PASS — Weights bounded (max q_hat<=50, En[1{T=k}q_k] in [0.9,1.1])
### E5 — E5 — Semi-synthetic on real HAMD covariates  (✅ PASS)
- ✅ PASS — OPM beats every ignorability baseline on PEHE (Wilcoxon, BH<0.05)
- ✅ PASS — OPM competitive with sieve-PDR (kernel-vs-sieve gap reported, within 20%)
### E6 — E6 — Ablations (S1)  (✅ PASS)
- ✅ PASS — (f) c_U=0 sanity: proximal ≈ fallback (|gap|<0.05)
- ✅ PASS — (f) proximal gain grows with c_U (gap non-decreasing & end>start)
- ✅ PASS — All ablation tables generated (a,b,c,d,e,f)
### E7 — E7 — Distributional counterfactuals (S1)  (❌ FAIL)
- ❌ FAIL — distilled mean bias < 50% of factual-DDPM on every arm k>=1
### E8 — E8 — Diagnostics validity study  (✅ PASS)
- ✅ PASS — |Spearman(rho_min, PEHE)| > 0.4 across corruption levels in >=1 family
- ✅ PASS — |Spearman(bridge_resid, PEHE)| > 0.4 across corruption levels in >=1 family
### E9 — E9 — Proximal removes nonlinear-confounding bias (ATE) that ignorability cannot  (✅ PASS)
- ✅ PASS — OPM-proximal ATE error < 50% of every ignorability baseline AND its own dr_fallback (Wilcoxon BH<0.05)
- ✅ PASS — Report PEHE & kernel-vs-sieve honestly (informational)
### E10 — E10 — Valid heterogeneous-effect inference / coverage  (✅ PASS)
- ✅ PASS — Subgroup-ATE coverage in [0.90, 0.975] at largest n (n=8000)
- ✅ PASS — Subgroup-ATE coverage increases (or holds) with n (asymptotic validity)
- ✅ PASS — Global ATE CI: no under-coverage at largest n (coverage >= 0.85)
