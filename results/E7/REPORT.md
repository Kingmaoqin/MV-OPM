# E7 — Distributional counterfactuals (S1)

**Overall: FAIL**  (0/1 acceptance criteria met)

## Acceptance criteria

- ❌ FAIL — **distilled mean bias < 50% of factual-DDPM on every arm k>=1**: arm1: ratio 1.31; arm2: ratio 1.31; arm3: ratio 0.78

## Per-arm mean bias and 1-Wasserstein (mean over seeds)

|   arm |   distilled_meanbias |   factual_meanbias |   ratio_d/f | <50%?   |   distilled_W1 |   factual_W1 |
|------:|---------------------:|-------------------:|------------:|:--------|---------------:|-------------:|
|     1 |               0.9876 |             0.7521 |       1.313 | False   |         0.9694 |       0.7289 |
|     2 |               0.6733 |             0.5119 |       1.315 | False   |         0.7532 |       0.5402 |
|     3 |               0.6771 |             0.8638 |       0.784 | False   |         0.7495 |       0.8912 |

### Honest interpretation (optional Stage-3 module)

The strict `<50%` target requires the factual generative baseline to carry a LARGE confounding-driven mean bias that distillation removes. Here the factual-DDPM mean bias is already small (0.5–0.86) — at CPU-feasible DDPM quality it is dominated by the generator's OWN conditional-mean error, not by removable confounding — so a 50% reduction is not attainable and the noisy through-sampling distillation gradient even degrades 2 of 3 arms. The mechanism is directionally real (E6(d): raising `lambda_distill` to 10 lowers the mean bias to 0.71), but the numeric acceptance is not met at this scale. Stage 3 is explicitly OPTIONAL (spec §0/1.7); the core estimator (Stages 1–2) is fully validated in E1–E6, E8. W1 also reflects the documented variance mismatch: distillation debiases the MEAN; higher moments inherit the factual conditional shape (spec 1.7). See DECISIONS.md.
