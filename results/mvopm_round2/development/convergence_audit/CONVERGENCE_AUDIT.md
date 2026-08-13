# Round-2 kernel convergence audit

This audit used observed-data criteria only; no oracle PEHE or ATE error was loaded.

Plateau-rule suggested cap: **300 epochs**.
Per the preregistered observed-data rule, no earlier plateau was found, so the final confirmatory
cap is frozen at 300 epochs with early stopping. No final seed had been launched when this choice
was recorded.

| scenario   |   budget |   median_D |   median_h_change |   median_q_change |   median_objective |
|:-----------|---------:|-----------:|------------------:|------------------:|-------------------:|
| S1         |       45 |  0.393539  |        nan        |         nan       |           0.687223 |
| S1         |       90 |  0.277085  |          3.19135  |           7.52645 |           0.509866 |
| S1         |      180 |  0.152111  |          2.13099  |           5.53809 |           0.344634 |
| S1         |      300 |  0.11214   |          0.96868  |           2.28184 |           0.260325 |
| S2         |       45 |  0.416282  |        nan        |         nan       |           0.720006 |
| S2         |       90 |  0.294984  |          2.11633  |           8.23623 |           0.535866 |
| S2         |      180 |  0.152986  |          1.96846  |           6.46893 |           0.337343 |
| S2         |      300 |  0.0982452 |          0.698727 |           2.59888 |           0.27221  |
| hamd       |       45 |  0.401791  |        nan        |         nan       |           0.679861 |
| hamd       |       90 |  0.283723  |          2.87095  |           8.68294 |           0.496853 |
| hamd       |      180 |  0.146822  |          2.064    |           6.18302 |           0.306387 |
| hamd       |      300 |  0.101114  |          0.817446 |           2.02848 |           0.247991 |
| nonlinear  |       45 |  0.304834  |        nan        |         nan       |           0.622807 |
| nonlinear  |       90 |  0.265088  |          2.06593  |           3.82899 |           0.570485 |
| nonlinear  |      180 |  0.213802  |          1.60912  |          18.134   |           0.432581 |
| nonlinear  |      300 |  0.122809  |          0.726415 |           6.85829 |           0.260161 |
