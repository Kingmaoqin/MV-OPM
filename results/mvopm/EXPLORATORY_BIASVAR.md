# MV-OPM — EXPLORATORY bias×variance probe (POST-HOC, NON-confirmatory)

Fresh seeds 500–514 (disjoint from pilot). Primary target = ATE error. This was invented
after the pilot HOLD and **cannot be reported as confirmatory** (MV-OPM §15/§29); it only
tests whether the prereg direction is worth a future confirmatory study.

## Median oracle-ratio (ATE) and mean rank-Spearman(score, ATE err), by scenario

| scenario   | score         |   median_ratio |   mean_spearman |   cat_rate |   top1_rate |
|:-----------|:--------------|---------------:|----------------:|-----------:|------------:|
| S1         | product       |          1.183 |           0.093 |      0.4   |       0.267 |
| S1         | biasvar_mult  |          1.06  |           0.699 |      0.333 |       0.4   |
| S1         | biasvar_add   |          1.06  |           0.699 |      0.333 |       0.4   |
| S1         | variance_only |          1.279 |           0.509 |      0.4   |       0.267 |
| S1         | fixed_sieve1  |          1.067 |           0.41  |      0.267 |       0.333 |
| S1         | fixed_kernel  |          3.339 |          -0.201 |      0.933 |       0     |
| S2         | product       |          1.012 |           0.269 |      0.267 |       0.467 |
| S2         | biasvar_mult  |          1.023 |           0.733 |      0.133 |       0.4   |
| S2         | biasvar_add   |          1.023 |           0.733 |      0.133 |       0.4   |
| S2         | variance_only |          1.095 |           0.39  |      0.2   |       0.333 |
| S2         | fixed_sieve1  |          1     |           0.498 |      0.133 |       0.533 |
| S2         | fixed_kernel  |          6.202 |          -0.532 |      1     |       0     |
| nonlinear  | product       |          4.069 |           0.192 |      0.6   |       0.333 |
| nonlinear  | biasvar_mult  |          1.882 |           0.745 |      0.733 |       0.133 |
| nonlinear  | biasvar_add   |          1.882 |           0.745 |      0.733 |       0.133 |
| nonlinear  | variance_only |          1.882 |           0.493 |      0.733 |       0.133 |
| nonlinear  | fixed_sieve1  |          1     |           0.602 |      0.067 |       0.8   |
| nonlinear  | fixed_kernel  |          7.324 |          -0.218 |      1     |       0     |
| hamd       | product       |          1.346 |           0.44  |      0.267 |       0.4   |
| hamd       | biasvar_mult  |          1.011 |           0.73  |      0.267 |       0.467 |
| hamd       | biasvar_add   |          1.011 |           0.73  |      0.267 |       0.467 |
| hamd       | variance_only |          3.931 |           0.112 |      1     |       0     |
| hamd       | fixed_sieve1  |          1.274 |           0.375 |      0.2   |       0.4   |
| hamd       | fixed_kernel  |          3.205 |          -0.288 |      1     |       0     |

## Exploratory read (labelled exploratory)
- product (prereg, bias only): median oracle-ratio 1.26, mean Spearman 0.23
- biasvar_mult (bias×var):     median oracle-ratio 1.04, mean Spearman 0.73
- biasvar_add  (bias+var):     median oracle-ratio 1.04, mean Spearman 0.73
- fixed_sieve1 (strong baseline): median oracle-ratio 1.03

