# MV-OPM PILOT analysis

Rows analyzed: 394. Primary target = **ATE error** (theory-aligned); PEHE secondary.

## Go / No-Go gate (frozen criteria; primary target = ATE error)

- C1 rank assoc Spearman(product, ATE err) > 0.2 in >=3/4 families: **FAIL** (1/4 families; per-family {'S1': 0.13, 'S2': 0.1, 'nonlinear': 0.07, 'hamd': 0.45})
- C2 product median oracle-ratio < fixed-kernel: **PASS** (product=1.384 vs kernel=4.640)
- C4 mechanism: moment product tracks true error product & |ATE bias|: **PASS**

### GATE DECISION: **HOLD**

## Study A — exact product mechanism

- spearman(moment_product, true_err_product) = **0.826**
- spearman(D_h, true_h_err) = **0.856**
- spearman(D_q, true_q_err) = **0.904**
- spearman(abs_ate_bias, moment_product) = **0.133**

Mean |ATE bias| by corruption pattern (DR should keep single-side small):
- neither (r_h=0,r_q=0): 0.0541
- h-only  (r_h>0,r_q=0): 0.0551
- q-only  (r_h=0,r_q>0): 0.0634
- both    (r_h>0,r_q>0): 0.0638

## Selection studies — primary target = ATE error

| scenario   |   n |   spearman_product |   ratio[oracle] |   cat[oracle] |   ratio[product] |   cat[product] |   ratio[h_only] |   cat[h_only] |   ratio[q_only] |   cat[q_only] |   ratio[fixed_kernel] |   cat[fixed_kernel] |   ratio[fixed_sieve1] |   cat[fixed_sieve1] |   ratio[random] |   cat[random] |
|:-----------|----:|-------------------:|----------------:|--------------:|-----------------:|---------------:|----------------:|--------------:|----------------:|--------------:|----------------------:|--------------------:|----------------------:|--------------------:|----------------:|--------------:|
| S1         |  12 |              0.129 |               1 |             0 |            1.46  |          0.5   |           1.664 |         0.667 |           1.695 |         0.667 |                 4.085 |               0.917 |                 1.188 |               0.333 |           1.695 |         0.667 |
| S2         |  12 |              0.1   |               1 |             0 |            1.21  |          0.417 |           1.606 |         0.5   |           2.155 |         0.583 |                 5.195 |               1     |                 1.052 |               0.083 |           2.317 |         0.583 |
| nonlinear  |  12 |              0.071 |               1 |             0 |            1.519 |          0.5   |          20.428 |         0.833 |           6.364 |         0.833 |                14.089 |               1     |                 1     |               0.333 |           2.383 |         0.667 |
| hamd       |  10 |              0.446 |               1 |             0 |            1.308 |          0.4   |           1.891 |         0.6   |           1.336 |         0.3   |                 3.237 |               1     |                 1.308 |               0.4   |           1.392 |         0.4   |

### Secondary (PEHE) — variance stress metric


| scenario   |   n |   spearman_product |   ratio[oracle] |   cat[oracle] |   ratio[product] |   cat[product] |   ratio[h_only] |   cat[h_only] |   ratio[q_only] |   cat[q_only] |   ratio[fixed_kernel] |   cat[fixed_kernel] |   ratio[fixed_sieve1] |   cat[fixed_sieve1] |   ratio[random] |   cat[random] |
|:-----------|----:|-------------------:|----------------:|--------------:|-----------------:|---------------:|----------------:|--------------:|----------------:|--------------:|----------------------:|--------------------:|----------------------:|--------------------:|----------------:|--------------:|
| S1         |  12 |              0.043 |               1 |             0 |            1.255 |          0.25  |           1.285 |         0.333 |           1.194 |         0.417 |                 1.517 |                 0.5 |                 1.042 |               0.083 |           1.182 |         0.333 |
| S2         |  12 |              0.124 |               1 |             0 |            1.183 |          0.333 |           1.465 |         0.417 |           1.317 |         0.333 |                 1.945 |                 1   |                 1     |               0.083 |           2.442 |         0.667 |
| nonlinear  |  12 |              0.005 |               1 |             0 |            2.018 |          0.75  |           2.023 |         0.583 |           2.108 |         0.75  |                 2.307 |                 1   |                 1.436 |               0.417 |           1.78  |         0.667 |
| hamd       |  10 |              0.389 |               1 |             0 |            1.045 |          0.1   |           1.431 |         0.5   |           1.078 |         0.1   |                 1.183 |                 0.1 |                 1.045 |               0.1   |           1.354 |         0.4   |

