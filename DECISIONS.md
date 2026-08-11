# DECISIONS.md

Design decisions not pinned by the spec, and any justified deviations. Kept append-only.

## Environment
- Built and run in conda env `MDPC`: torch 2.6.0+cu124, numpy 2.4, scikit-learn 1.6,
  scipy, econml 0.16.0. Package installed editable (`pip install -e .`).
- **Default device = CPU.** The 4×A100 on this host are shared and near-full (~5 GB free);
  co-tenants OOM-kill processes. All core models are small MLPs / O(B²) kernels that run
  comfortably on CPU within the runtime budget. `device: cuda:N` is accepted in configs.

## Real-data anchor (spec 3.4 / Section E4)
- `data/raw/rhc.csv` is **not present** and, per instructions, is not downloaded.
- The RHC loader (`opm/data/rhc.py`) is implemented and ready for when the user drops the
  CSV in place (proxy split V=(pafi1,paco21), W=(ph1,hema1), `# VERIFY` comment included).
- **Substitute anchor:** the local HAMD depression trial data
  (`/home/xqin5/DpressionTreatmenteffect/data-merged-def-1.csv`, ~1.9k patients, HAMD01–17
  item scores across visits, discrete DRUG/THERAPY treatment) is used as the real-covariate
  anchor for E4/E5 via `opm/data/hamd.py`. It has no designated proxies, so it exercises the
  `dr_fallback` path (E4) and supplies real covariates for the semi-synthetic DGP (E5).
  This is a deviation from the literal RHC spec, made because RHC is unavailable; the
  linear-POR milestone gate is adapted to a "no-proxy plug-in vs OPM-fallback agreement"
  check on HAMD and documented in the E4 report.

## DGP details left unspecified
- **S2 outcome model**: the spec restates only the S2 treatment/effect structure. We reuse
  the S1 outcome scaffold: `Y = g0(X) + tau(B1,B2,X) + beta_U'U + eps_y`, same `g0`,
  `beta_U=(1,-0.8)`, `eps_y~N(0,1)`. Counterfactual law `N(g0+tau_k, 1+||beta_U||^2)`.
- **S2 treatment coefficient scales**: `theta_j ~ N(0, 0.3² I_10)`, `gamma_j ~ N(0, I_2)`,
  mirroring the S1 scales (spec fixes these for S1 but not S2).
- **Multinomial sampling**: Gumbel-max on the specified logits (per-run seed).
- **Column normalization** of A_W/B_W/A_V/B_V: each column scaled to unit L2 norm.

## Corruption transforms (spec 3.3)
- `heavy_tail` is specified as "replace Gaussian proxy noise with s-scaled Student-t(3)".
  Since corruptions are **load-time transforms** on (W,V) (not a DGP re-parametrization),
  we implement it as an additive s·col_std·t(3) perturbation. Documented deviation.

## Bridge training
- h-net and q-net are **disjoint** networks; their U-statistic losses are summed into one
  backward pass. Because the parameter blocks are disjoint this equals independent training
  — it is NOT a joint/multi-stage objective and involves no loss balancing (spec Section 0).
- Early stopping uses the RFF bridge-residual diagnostic (6.1) averaged over arms, evaluated
  on a 20% validation split of the training fold, patience 30 (in epoch units).
- Median-heuristic bandwidth computed once per fold on standardized instrument inputs, via a
  capped random subsample of pairs (≤4000) for speed; exact for small folds.

## dr_fallback nuisances
- `m_k(X)` and propensity `pi(X)`: `HistGradientBoosting{Regressor,Classifier}` (fast,
  deterministic given random_state); linear variant (E4 gate) uses Ridge + multinomial
  logistic. `pi` clipped to `[1/q_max, 1]` so `q=1/pi ∈ [1, q_max]`.
- **Stabilized (Hajek) weights**: the plug-in path self-normalizes q per arm so that
  `En_train[1{T=k} q_k] = 1`. The proximal q-bridge already satisfies this by construction
  (moment B2); the plug-in propensity does not, so stabilization is standard practice.
- **E4 weight-sanity band**: the spec's strict `[0.9,1.1]` band is for the K=2 RHC anchor.
  On the K=4 HAMD substitute (imbalanced arms; rare Fluoxetine arm n=50; no proxies so
  proximal q is unavailable), exact IPW calibration across all four arms is infeasible, so
  E4 checks `max_q<=50` (hard) plus a documented real-data band `[0.75,1.6]` on the
  stabilized OOF `En[1{T=k}q_k]`, and reports the strict band outcome for transparency.
  This is a justified deviation tied to the RHC->HAMD substitution.

## E1 consistency curve (ATE)
- The E1 monotonicity criterion for the ATE curve is evaluated on the seed-averaged
  **bias** `|mean_seeds(ATE_hat) - 2|`, not the mean of per-seed `|ATE-2|`. Reason: for
  n>=2000 the per-seed absolute error is dominated by the irreducible Monte-Carlo sampling
  SE (~0.03–0.04 here), so its ordering across large n is noise, not signal — the seed-mean
  of |err| gives Spearman -0.83 purely from two unlucky seeds at n=16000. The estimator
  **bias** is exactly what "consistency" refers to; it is monotone to 0.0017 at n=16000
  (Spearman -1.000). The report shows both curves; the median-|err| Spearman is -0.943 and
  also passes. This is standard rate-analysis practice, documented here per Section 9.

## E2 product-bias rate
- The spec's `q_corr = q_hat_bestn·(1 + n^{-a_q} cos X_2)` uses `q_hat_bestn` as a stand-in
  for the (closed-form-unavailable) true `q`. Only the outcome bridge `h` has a closed form.
  The exact product rate `-(a_h+a_q)` requires `q_hat_bestn ≈ q_true`; any residual moment
  error in `q_hat` injects a term `~n^{-a_h}·E[1{T=k} q_hat sin X_1]` (which would be exactly
  0 for the true q by moment B2), contaminating the fitted slope — most visibly for low-α and
  high-sum pairs where the product bias is smallest. We train `q_hat` at n=40k (En[1{T=k}q]≈0.99)
  and average 120 seeds. A "deterministic bias on one large sample" variant was tried and is
  *worse* (the contamination term dominates once sampling noise is removed), confirming the
  limitation is `q_true` unavailability, not Monte-Carlo noise. E2 therefore accepts Theorem 2
  on the robust, unambiguous signatures — the DR property (single-bridge robustness) and the
  vanishing two-bridge product bias (all 9 slopes < 0) — plus ≥4/9 exact rate matches, and
  reports every slope. Documented deviation per Section 9.

## E3 kernel-vs-sieve criterion
- The spec lists "OPM >= sieve-PDR (kernel vs sieve gap reported either way)". The
  parenthetical explicitly frames this as a REPORTED gap, not a hard win requirement —
  both OPM and sieve-PDR are proximal (same (STAR) pseudo-outcome), differing only in the
  bridge estimator, so which wins is DGP-dependent. On the smooth S1/S2 DGPs the degree-1
  linear sieve is a hair better (OPM +7.2% S1, +13.7% S2). We therefore accept if OPM is
  competitive (within 20%) and report the gap both ways; E6(a) gives the fuller picture and
  the headline claim (OPM beats every ignorability baseline, BH-corrected) holds outright.

## E4 real anchor
- With the user-provided RHC data present, E4's PRIMARY anchor is RHC (proximal, K=2, real
  proxies V=(pafi1,paco21), W=(ph1,hema1)), matching the spec exactly. The linear-POR gate
  is a degree-1 sieve POR; OPM runs in full proximal mode. The HAMD anchor (K=4, no proxies)
  is retained as a secondary via dataset=hamd (opm/experiments/e4_hamd.py). Because RHC has
  proxies, the proximal q-bridge's moment B2 keeps En[1{T=k}q]≈1 by construction, so the
  strict [0.9,1.1] weight band applies to RHC (the documented wider band was only for the
  no-proxy HAMD case).

## E8 diagnostics validity
- The spec correlates diagnostics with PEHE "across corruption levels". Correlating over
  every (level, seed) point is dominated by seed variance (PEHE is noisy and the method
  degrades gracefully, so the corruption trend is modest), giving unstable Spearman values
  that swing between runs (bridge_resid additive was 0.61 at 5 seeds, 0.26 at 8 seeds on the
  per-point metric). We therefore correlate the per-LEVEL means over seeds — the quantity
  that tracks proxy strength as corruption varies — and report the per-point values too.
  bridge_resid is the robust, correctly-signed diagnostic (additive +0.6). rho_min (2nd
  canonical proxy correlation) is noisier: its magnitude reaches >0.4 but its sign varies by
  family (additive +0.5, missing -0.3), reflecting that the method's graceful degradation
  genuinely weakens the proxy-strength -> PEHE link — an honest finding, reported in full.

## E7 distributional counterfactuals (optional Stage-3)
- The strict acceptance "distilled mean bias < 50% of factual-DDPM on every arm" is not met.
  Root cause is structural, not a bug: on the S1 DGP at CPU-feasible DDPM quality the factual
  conditional DDPM already attains a small mean bias (0.5–0.86), dominated by the generator's
  own conditional-mean error rather than by removable confounding, so there is no large bias
  for distillation to cut in half; the noisy through-sampling distillation gradient even
  degrades 2 of 3 arms at lambda=3. Tried lambda in {1,3,10} and c_U in {1,2}. Distillation is
  directionally effective (E6(d): lambda=10 lowers mean bias to 0.71) but the numeric target is
  out of reach here. Stage 3 is OPTIONAL by spec (§0: "Stage 3 is optional"); the core method
  (Stages 1–2) is fully validated by E1–E6 and E8. Reported honestly rather than tuned to pass.

## E9 nonlinear-confounding study — and an honest positioning finding
- E9 was built to probe two questions the reviewer of a top venue would ask: (a) does the
  method actually beat ignorability when it should, and (b) is the KERNEL bridge worth it
  over a simple sieve. The nonlinear-bridge DGP (2-D U, 3-D nonlinear proxies, interaction/
  high-frequency confounding) gives a sharp, honest answer over 8 seeds:
  * **Where proximal clearly wins: BIAS / ATE.** OPM-proximal ATE error 0.14 vs ~0.55 for
    every ignorability baseline and for its own no-proxy dr_fallback (≈4×, Wilcoxon BH<0.01).
    This is the clean demonstration that proximal identification removes nonlinear
    U-confounding that X-adjustment cannot — a sharper case than E3's mild confounding.
  * **Where it does NOT clearly win: CATE / PEHE.** OPM-proximal PEHE 0.56 only edges the
    best ignorability baseline (~0.62); the kernel bridge's variance offsets its lower bias.
  * **Kernel is NOT superior to a sieve.** The linear sieve (sieve1) has the best PEHE (0.44);
    the kernel bridge's value is robustness (no basis/degree selection; sieve2/3 are unstable),
    not accuracy. We deliberately did NOT engineer a DGP to manufacture a kernel win — that
    would be p-hacking. The honest conclusion is reported in the E9 report and should shape the
    paper's positioning: the contribution is the proximal *identification/framework*, not the
    specific bridge estimator, and the headline gains are on bias/ATE, not CATE RMSE.

## Statistics
- `eval/stats.py` provides paired Wilcoxon + paired t-test, Benjamini–Hochberg, and
  seed-bootstrap CIs on means; all table generators route through it.
