# Review B — Code-Correctness Audit of MV-OPM Round-3 Selector Sweep

**Reviewer:** Review Agent B (independent, skeptical, code-correctness angle)
**Repo:** `/home/xqin5/进化的opm/`
**Python:** `/home/xqin5/.conda/envs/MDPC/bin/python` (env MDPC)
**Date:** 2026-08-28
**Method:** read the full pipeline, then RAN diagnostic scripts (`reviewB_*.py`) that reproduce
every headline number from scratch. Nothing below is taken from the report or docstrings.

---

## OVERALL VERDICT

**The pipeline is CORRECT. The bad Round-3 results are FAITHFUL to the data and the method —
they are NOT caused by implementation bugs.**

Both headline "failures" reproduce exactly and are explained by genuine statistical mechanisms,
not miscoded features, mis-indexing, or leakage:

- **`nonlinear` MSES median oracle-ratio 44×** — the identifying-moment screen has *no statistical
  power* against the two catastrophically over-fit polynomial-sieve candidates (their residuals are
  so large/noisy that no moment violation is detectable), so the screen *passes exactly the wrong
  candidates* and rejects the good kernel ones. MSES then min-variance-selects among the survivors
  and lands on a candidate whose ATE is off by ~3.6 (true ATE ≈ 2). This is a real, deterministic
  method failure. I verified the moment test itself is *correctly calibrated* (uniform p under the
  true bridge) and that the power asymmetry is genuine.

- **`hamd` variance_only 100% catastrophic** — the degree-3 sieve's treatment bridge *collapses*
  (self-normalization `E_n[1{T=k}q_k]` ≈ 0.17 instead of 1, `q_balance` ≈ 0.90), which deflates its
  OOF pseudo-outcome variance to the global minimum (1.95 vs 14–31 for the good candidates).
  Minimum-variance selection is therefore deterministically gamed by a degenerate near-zero-q
  bridge whose ATE is biased (error 0.32). The `q_balance` diagnostic *correctly flags* it, which is
  exactly why the q-aware selectors avoid it.

Every selection number I checked equals what the (correctly-computed) features dictate. The
features are extracted from the right quantities and correctly aligned to each candidate.

The one interpretation caveat (not a code bug): the **`oracle_ratio` metric is fragile because the
oracle-best denominator is often near-zero** (median 0.020 in nonlinear; 8/16 seeds < 0.02), which
inflates the ratio headline. But the **absolute** errors independently confirm the failures are
real (MSES nonlinear median absolute ATE error = 1.53 on a true ATE of 2).

---

## SEVERITY-RANKED FINDINGS

### CRITICAL — none.
No correctness bug that produces wrong selection numbers, wrong pseudo-outcomes, wrong variances,
wrong p-values, or oracle leakage was found.

### MAJOR

**M1 — `oracle_ratio` headline is denominator-inflated (metric design, not a coding bug).**
`scripts/mvopm_variant_sweep.py:102` computes `oracle_ratio = err / (best_err + 1e-12)`. When the
oracle-best candidate nails the ATE (near-zero `best_err`), the ratio explodes even for a benign
pick.
Evidence (`reviewB_*` recompute from `results/mvopm_variants/raw_*.jsonl`):
- nonlinear/MSES: median ratio 44.06, mean 754.9, **max 10,590**; oracle_best_error median 0.0196,
  **8/16 seeds < 0.02**. BUT median *absolute* error of the pick = **1.530** (true ATE ≈ 2) →
  the failure is genuine, not merely a small-denominator artifact.
- hamd/variance_only: median ratio 5.04, catastrophic 1.000; but absolute error is a steady
  ~0.315 (true ATE vector `[1.0, 0.116, -0.016]`). Here "100% catastrophic" is partly
  metric-amplified — variance_only is consistently the *worst* selector, but 0.315 absolute is
  moderate, not apocalyptic.
Impact: the "44×" / "100% catastrophic" phrasing overstates severity relative to absolute error.
The qualitative conclusions (MSES fails in nonlinear; variance_only fails in hamd) are sound and
reproduce; the magnitude framing should be read alongside absolute error. Recommend reporting
`regret` (absolute) as the primary headline; the code already logs it.

### MINOR

**M2 — `mses_varcap` docstring contradicts its code; the variance cap is toothless exactly when
needed.** `opm/validation/selector_variants.py:108-124`. The docstring promises "If the cap (or the
screen) empties the set, fall back to the **global minimum-variance candidate**." The code does
`pool = capped or survivors`, i.e. when the cap empties but the screen has survivors it falls back
to the **survivor set** (= plain MSES), never to the global min-variance candidate. In the nonlinear
D2 case the only survivors are `sieve2_sieve2` / `sieve3_sieve3` (var 52 492 / 70 713), both far
above the cap `3·median = 130.6`, so `capped = []`, `pool = survivors`, and the pick is
`sieve2_sieve2` — identical to MSES. Verified live and synthetically (`reviewB_trace.py`,
plus a direct reconstruction). The report's claim that "`mses_varcap` ≡ MSES on these data" is
therefore **true**, and I confirmed *why*; but the in-code docstring is misleading and the
mechanism gives zero protection in the very regime it was designed for. Documentation/design issue,
not a results-corrupting bug.

**M3 — Cosmetic: `pseudo_outcome.py` header says "STAR" but references (spec 1.4/1.6) elsewhere spell
it "STARSTAR"** (`opm/estimator/tau_head.py:1`). No functional impact.

---

## VERIFIED-CORRECT (trustworthy)

1. **Pseudo-outcome `compute_phi` (STAR / doubly-robust) — CORRECT.**
   `opm/estimator/pseudo_outcome.py:9`. `φ_k = 1{T=k}·q_k·(Y−h_k)+h_k`.
   - Exact hand match on a crafted n=4/K=2 case (`reviewB_phi_unit.py` TEST 1).
   - Reference arm = arm 0, correctly aligned (`core.py:66` uses `phi[:,k]-phi[:,0]`).
   - True-bridge ATE recovery on the exact DGP: STAR ATE = 1.94–2.03 across seeds (incl.
     77000000 → 2.031), true = 2.0. The unrealized-arm entry equals `h_k` exactly, and only the
     realized arm uses `q` — both confirmed.

2. **Bridge fitting — CORRECT implementation; sieve degeneracy is REAL, not a bug.**
   (`reviewB_bridge_fit.py`, seed 77000000)
   - Kernel bridge: healthy everywhere. Self-normalization `E_n[1{T=k}q_k]` dev = 0.03 (nonlinear) /
     0.014 (hamd); q strictly positive, well-bounded (max 3–8); h correlates 0.64–0.90 with Y. No
     NaN, finite objective, EMA + early stopping converge.
   - Sieve1: acceptable. Sieve2/Sieve3: **genuinely degenerate on the hard DGPs** — nonlinear
     sieve3 h-corr collapses to 0.16 (RMSE 15 vs base 2.3); hamd sieve3 q self-norm collapses to
     0.17 (`q_balance` 0.83) and h interpolates (RMSE 0, in-sample). This is the intended
     curse-of-dimensionality behavior of a polynomial sieve, and it is what drives both failures.
     The huge OOF variances (52 000 / 70 000) are the correct consequence of q hitting the 50-clip
     × large residuals — variance_only correctly avoids them; the moment screen incorrectly passes
     them.

3. **Feature extraction & alignment — CORRECT.**
   `scripts/mvopm_variant_sweep.py::_features` and `core.py::evaluate_candidates_round2`.
   Live asserts (`reviewB_trace.py`) confirm `feat[n]["var"] == rows[n]["po_var_mean"]` and
   `feat[n]["q_balance"] == rows[n]["q_balance"]` for every candidate; `p_h/p_q/D_h/D_q/max_q` are
   read per-name from a dict keyed by candidate, so no cross-candidate swap is possible.
   `q_balance = mean_k |E_n[1{T=k}q_k] − 1|` is computed correctly in
   `diagnostics.py::q_sanity` (`En_1Tk_qk = (onehot*q).mean(0)`), matching the documented formula.

4. **Cross-fitting / leakage — CLEAN.**
   - Inner OOF (`opm/validation/oof.py`): every observation predicted exactly once by a model that
     excluded it (`coverage == 1` invariant enforced, raises otherwise).
   - Oracle isolation: `ObservedDatasetView` physically blocks `tau_true/ate_true/mu_true/cf_mean/U/
     h_true` (all raise `AttributeError`) and returns read-only arrays (`X[0,0]=…` raises) —
     verified directly. Truth (`opm/eval/oracle.py`) enters ONLY the *scoring*, never the selector's
     `feat`. The selectors' oracle-blindness is additionally enforced by the poisoned-feature test
     (`tests/test_selector_variants.py::test_oracle_blind`, passes).
   - The Round-3 global-OOF path scores `rows[chosen]["ate_err"]`, an OOF quantity; selection and
     scoring share the global-OOF φ but the *selector inputs contain no truth*, so there is no
     oracle leakage into selection. The separate nested path (`nested_mses.py`) properly excludes
     the outer-test fold from both selection and refit (audited).

5. **Moment-test calibration — CORRECT.** (`reviewB_calib.py`)
   `moment_tests.py::studentized_rff_max_test` — centered Gaussian-multiplier bootstrap, shared
   sample SE for observed & bootstrap stats, `(1+exceed)/(n_boot+1)` p-value.
   - Under the TRUE bridge on the exact DGP (40 seeds): `p_h` mean 0.562, frac<0.05 = 0.000;
     `p_q` mean 0.450, frac<0.05 = 0.075. ≈ uniform, **not** systematically tiny. Calibrated.
   - Power asymmetry is genuine: true h → p ≈ 0.75; structured-bias h → p = 0.002 (correctly
     rejects); huge-noise garbage h → p ≈ 0.28–0.66 (correctly *loses power*). This is precisely the
     mechanism that makes the screen anti-informative for over-fit sieves — a real property of the
     method, not a coding error.

6. **Selector library — CORRECT.** `tests/test_selector_variants.py` (32 tests) all pass; hand
   re-derivation of every selector's pick on both target seeds matches the code
   (`reviewB_trace.py`). `mses_varcap` equivalence to MSES confirmed with mechanism (see M2).

---

## PER-CANDIDATE TRACE — `nonlinear`, seed 77000000
(live reproduction via `evaluate_candidates_round2`, n=2000, n_boot=299; matches recorded
`raw_nonlinear.jsonl` to ~3 decimals)

| candidate | ate_err | po_var_mean | pehe | screen_pass | p_dr_min | q_balance | max_q | ATE_hat |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|
| kernel_kernel | 0.0086 | 10.89 | 0.88 | **False** | 0.0033 | 0.022 | 9.12 | 2.004 |
| kernel_sieve1 | **0.0064** | 29.07 | 0.81 | **False** | 0.0033 | 0.192 | 36.84 | 2.006 |
| sieve1_kernel | 0.0516 | 16.62 | 0.65 | **False** | 0.0067 | 0.022 | 9.12 | 2.064 |
| sieve1_sieve1 | 0.0487 | 58.03 | 1.07 | **False** | 0.0033 | 0.192 | 36.84 | 2.061 |
| sieve2_sieve2 | 3.6047 | 52492.8 | 4.12 | **True** | 0.0400 | 0.384 | 50.0 | 5.617 |
| sieve3_sieve3 | 3.7006 | 70713.5 | 14.57 | **True** | 0.1533 | 0.696 | 50.0 | 5.713 |

- Oracle-best = `kernel_sieve1` (err 0.0064). True ATE = 2.012.
- **MSES**: screen (Bonferroni α/K = 0.025) survivors = **{sieve2_sieve2, sieve3_sieve3}** — the two
  worst candidates. min-variance among them → **sieve2_sieve2**, err **3.6047** (ATE_hat 5.62).
  `mses_varcap[1.5]`/`[3.0]` → same **sieve2_sieve2** (both survivors exceed the cap; see M2).
- **variance_only** → `kernel_kernel` (global-min var 10.89), err **0.0086**. Correct.
- **q_gated_variance / min_q_balance / varq_combo / soft_screen / complexity_var** → `kernel_kernel`
  (0.0086). The q/variance signals all avoid the catastrophe; only the moment-trusting MSES family
  fails.
- **Explanation:** NOT a bug. The powerless moment screen passes exactly the over-fit high-variance
  sieves and rejects the well-fit kernels (which have small residuals in which the test *can* detect
  a tiny remaining violation). Confirmed by the calibration + power-asymmetry experiment.

## PER-CANDIDATE TRACE — `hamd`, seed 77000000

| candidate | ate_err | po_var_mean | pehe | screen_pass | p_dr_min | q_balance | max_q |
|---|---:|---:|---:|:--:|---:|---:|---:|
| kernel_kernel | 0.1345 | 14.36 | 0.563 | True | 0.2467 | 0.033 | 6.89 |
| kernel_sieve1 | **0.0066** | 28.77 | 0.549 | True | 0.2733 | 0.140 | 24.59 |
| sieve1_kernel | 0.0382 | 15.23 | 0.553 | True | 1.0000 | 0.033 | 6.89 |
| sieve1_sieve1 | 0.0817 | 31.47 | 0.557 | True | 1.0000 | 0.140 | 24.59 |
| sieve2_sieve2 | 1.0828 | 7986.2 | 4.926 | True | 0.0133 | 1.236 | 50.0 |
| sieve3_sieve3 | 0.3177 | **1.947** | 0.916 | **False** | 0.0033 | 0.900 | 4.42 |

- Oracle-best = `kernel_sieve1` (err 0.0066). True ATE vector = [1.0, 0.116, −0.016].
- **variance_only** → `sieve3_sieve3` (global-min var **1.947**), err **0.3177**. FAILURE — sieve3's
  q collapsed (self-norm ≈ 0.17 ⇒ q_balance 0.90 ⇒ φ ≈ h ⇒ artificially low contrast variance).
- **MSES** → screen rejects sieve3 (p_dr 0.0033 < 0.0125); survivors exclude it; min-var → 
  `kernel_kernel`, err **0.1345** (better than variance_only here — screen helps in this regime).
- **min_q_balance / q_gated_variance / varq_combo** → `kernel_kernel` (0.1345): the `q_balance`
  signal correctly identifies sieve3 as pathological and avoids it.
- **biasvar_mult / biasvar_add / product_moment / rank_sum / varq_moment_combo** → `sieve1_kernel`
  (0.0382), the second-best; **soft_screen[0.5] / complexity_var / min_ci_width / var_tiebreak**
  follow variance_only into `sieve3_sieve3` (0.3177).
- **Explanation:** NOT a bug. Minimum-variance is deterministically gamed by a degenerate low-q
  bridge; the pipeline *measures this correctly* and the q-diagnostic *flags it correctly*.

---

## REPRODUCTION COMMANDS
```
P=/home/xqin5/.conda/envs/MDPC/bin/python
S=/tmp/claude-1159798523/-home-xqin5/557f8a62-8b64-4d80-90ae-0102b85e51e3/scratchpad
$P $S/reviewB_phi_unit.py     # compute_phi correctness + ATE recovery
$P $S/reviewB_bridge_fit.py   # single-fit bridge diagnostics (q self-norm, h fit, NaN)
$P $S/reviewB_calib.py        # moment-test calibration + power asymmetry
$P $S/reviewB_trace.py        # live per-candidate trace, both target seeds, feature-alignment asserts
$P -m pytest tests/test_selector_variants.py tests/test_dr_fallback_equals_proximal.py \
   tests/test_crossfit.py tests/test_mvopm_invariants.py -q   # all pass
```

## BOTTOM LINE
The user's suspicion that the bad Round-3 numbers come from *code bugs* is **not supported**. The
STAR pseudo-outcome, bridge fits, OOF cross-fitting, oracle isolation, moment test, feature
extraction, and selector library are all implemented correctly and reproduce the reported numbers.
The failures are honest, correctly-measured consequences of (a) polynomial-sieve bridge degeneracy
on the hard DGPs, (b) a correctly-calibrated but power-asymmetric moment screen that cannot reject
high-variance over-fit bridges, and (c) minimum-variance selection being gamed by a q-collapsed
bridge. The only substantive caveats are the denominator-inflated `oracle_ratio` headline (M1) and
the misleading `mses_varcap` docstring/toothless cap (M2) — neither corrupts a result.
```
