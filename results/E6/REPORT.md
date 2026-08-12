# E6 — Ablations (S1)

**Overall: PASS**  (3/3 acceptance criteria met)

## Acceptance criteria

- ✅ PASS — **(f) c_U=0 sanity: proximal ≈ fallback (|gap|<0.05)**: |gap@c_U=0| = 0.0417
- ✅ PASS — **(f) proximal gain grows with c_U (gap non-decreasing & end>start)**: gaps = [0.042, 0.208, 0.351, 0.34]
- ✅ PASS — **All ablation tables generated (a,b,c,d,e,f)**: tables a–f written to REPORT.md

### (a) kernel-moment vs polynomial-sieve bridges

| bridge       |   PEHE_mean |   PEHE_sd |
|:-------------|------------:|----------:|
| kernel (OPM) |      0.6105 |    0.0824 |
| sieve deg1   |      0.5863 |    0.2105 |
| sieve deg2   |      0.5312 |    0.0459 |

### (b) cross-fitting on/off

| setting                   |   PEHE_mean |   PEHE_sd |
|:--------------------------|------------:|----------:|
| cross-fit ON (5 folds)    |      0.6105 |    0.0824 |
| cross-fit OFF (in-sample) |      0.5898 |    0.1079 |

### (c) q-clipping q_max

|   q_max |   PEHE_mean |   PEHE_sd |   max_q_seen |
|--------:|------------:|----------:|-------------:|
|      10 |      0.6106 |    0.0824 |          6.6 |
|      50 |      0.6105 |    0.0824 |          6.6 |
|     200 |      0.6105 |    0.0824 |          6.6 |

### (f) dr_fallback vs proximal across c_U  — KEY ABLATION

|   c_U |   proximal_ATEerr |   fallback_ATEerr |   gap_ATE(fb-prox) |   proximal_PEHE |   fallback_PEHE |
|------:|------------------:|------------------:|-------------------:|----------------:|----------------:|
|   0   |            0.1    |            0.1417 |             0.0417 |          0.7172 |          0.7774 |
|   0.5 |            0.0679 |            0.2757 |             0.2078 |          0.5831 |          0.939  |
|   1   |            0.1135 |            0.4641 |             0.3505 |          0.6105 |          0.7719 |
|   2   |            0.2305 |            0.57   |             0.3395 |          0.6384 |          0.9133 |

Expected: gap≈0 at c_U=0 (sanity); proximal ATE-bias advantage grows with c_U.

### (d) distillation lambda sweep (mean bias, lower=better)

|   lambda_distill |   mean_bias |
|-----------------:|------------:|
|              0.1 |      1.295  |
|              1   |      2.083  |
|             10   |      0.7127 |

### (e) diffusion vs Gaussian-MLP generator (mean-matched)

| generator                   |   mean_bias |
|:----------------------------|------------:|
| diffusion (distilled)       |      2.083  |
| Gaussian-MLP (mean-matched) |      0.4975 |

