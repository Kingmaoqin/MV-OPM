# OPM v2 — Contributions & Limitations (honest positioning)

A one-page, evidence-grounded read of what this codebase does and does not establish, to
guide venue choice and paper framing. Every claim below is backed by a `results/{Ex}/REPORT.md`.

## What is solidly established (defensible contributions)

1. **A unified multi-treatment proximal doubly-robust CATE framework with honest degradation.**
   Arm-wise kernel-moment bridges (Stage 1) → out-of-fold DR pseudo-outcomes (STAR, Stage 2)
   → CATE head + influence-function CIs, with a `dr_fallback` mode that degrades to a standard
   multi-treatment DR-learner when proxies are absent. Cross-fitting is leakage-proof
   (unit-tested), and the closed-form bridge, q-moment, product double-robustness, and
   no-confounding equivalence are all verified (tests 1–7, E1 5/5).

2. **Proximal identification removes confounding bias that ignorability fundamentally cannot —
   sharpest under NONLINEAR confounding (E9).** With interaction/high-frequency confounding
   phi(U)=U1U2+0.6(U1²−1)+0.8 sin(U1+U2), OPM-proximal's **ATE error is ~1/4 of every
   ignorability baseline and of its own no-proxy fallback** (0.14 vs ~0.55; Wilcoxon BH<0.01).
   This is the cleanest "why proximal" evidence in the suite.

3. **A faithful real-data proximal result (E4/RHC).** On RHC (n=5735, real proxies) the linear
   POR gate and full proximal OPM both recover the literature's negative effect of RHC on
   survival (−1.9 / −1.3 days), CIs overlap, and the proximal weights are well-calibrated by
   construction (En[1{T=k}q]=0.998/0.972).

4. **Valid heterogeneous-effect inference (E10).** Subgroup-ATE CIs from the proximal
   pseudo-outcome attain near-nominal coverage that approaches 95% as n grows — inference
   beyond the global ATE.

5. **Honest, reproducible engineering.** 10 experiments, one-command reproduction, per-experiment
   PASS/FAIL vs numeric criteria, and every deviation logged in `DECISIONS.md`.

## What is NOT established (limitations — stated plainly, not hidden)

1. **The kernel bridge is not empirically superior to a simple sieve.** Across E3, E5, E6(a),
   and E9, a linear/low-degree polynomial sieve matches or slightly beats the kernel-moment
   bridge on PEHE (e.g. E9: sieve1 0.44 vs kernel 0.56). The kernel's value is **robustness**
   (no basis/degree selection; degree-2/3 sieves are high-variance and unstable), **not**
   accuracy. Do not headline the kernel bridge.

2. **The PEHE/CATE advantage over ignorability is modest; the clean win is on bias/ATE.** Where
   the effect surface is simple, low-variance ignorability learners are competitive on PEHE
   even when biased on the level (their variance beats the proximal estimator's).

3. **No new theory is proven here.** Identification/rate results are arm-wise applications of
   Miao et al. 2018 / Cui et al. 2023 / Kennedy 2020. The repo validates them empirically;
   it does not contribute new theorems (e.g. semiparametric efficiency or uniform inference
   for the multi-arm proximal CATE).

4. **The Stage-3 diffusion module (E7) does not meet its target.** At CPU-feasible DDPM quality
   the generator's own conditional-mean error dominates the confounding bias, so distillation
   cannot halve the factual mean bias. Stage 3 is optional; treat it as preliminary.

5. **Some baselines are represented, not re-implemented** (DFPV, DiffPO; P-learner shown to
   reduce to PDR here). Real-data is RHC + a HAMD trial; MIMIC-IV is designed-for but unused.

## Venue read

- **As-is (workshop / applied track / CLeaR / strong preprint):** ready — the empirical study,
  the RHC anchor, and the E9 "proximal necessity under nonlinear confounding" figure are a
  coherent, honest package.
- **Top-tier main track (NeurIPS/ICML/AISTATS) or a stats journal:** needs a genuine
  methodological contribution on top of this scaffold. The highest-leverage additions, in
  order: (i) a NEW theoretical result — semiparametric-efficient and/or uniformly-valid
  multi-arm proximal CATE inference; (ii) a setting where the neural bridge provably/robustly
  beats sieves (or drop the kernel-superiority claim entirely and sell the framework +
  identification); (iii) a real application where proximal reveals a finding ignorability
  misses, with domain validation.

**Bottom line:** the honest contribution is the *proximal identification framework and the
empirical characterization of when it helps* (a lot on nonlinear-confounding bias; modestly on
CATE), not a superior bridge estimator. Framed that way, it is truthful and defensible; framed
as "a better CATE learner via kernel bridges", it is contradicted by our own results.
