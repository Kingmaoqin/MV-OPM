# Round-3 code & design review — two independent reviewers, reconciled

**Trigger.** The user suspected the Round-3 "failures" (MSES catastrophic in `nonlinear`;
`variance_only` catastrophic in `hamd`) were caused by **bugs, bad experiment design, or degenerate
datasets**. Two agents reviewed the same code *independently*, formed separate verdicts, and ran
their own diagnostics (env `/home/xqin5/.conda/envs/MDPC/bin/python`):

- **Review A** — methods / statistics / DGP-validity lens.
- **Review B** — code-correctness lens (audited `compute_phi`, bridge fits, features, cross-fitting,
  moment-test calibration, oracle isolation, and traced two concrete seeds).

Both reproduced every headline number from scratch. Their full findings are in
`scratchpad/review_A_findings.md` and `scratchpad/review_B_findings.md`.

---

## Bottom line

**The user's "it's a code bug" hypothesis is mostly NOT supported — but the review was still worth
it, because it found one real bug, one real metric flaw, and one wrong interpretation in my Round-3
write-up. None of these were corrupting the raw pipeline; two of them changed conclusions once fixed.**

| Claim under test | Verdict |
|---|---|
| The bad results come from a buggy *pipeline* (pseudo-outcome, bridges, features, cross-fitting, leakage) | **NO** — Review B verified all correct and reproduced the numbers |
| The datasets are degenerate / the effect is ~0 / no valid bridge exists | **NO** — Review A verified all 5 DGPs have real overlap, non-trivial ATE, informative proxies, and an adequate library |
| The reported *failures* are real | **YES** — genuine in absolute terms (MSES nonlinear = 76% of the true effect; variance_only hamd = 66%) |
| The reported *severity numbers* ("44×", "5.04×", "100% catastrophic") are trustworthy | **NO** — the oracle-ratio metric is denominator-inflated and unnormalized by effect size (both reviewers, independently) |
| `mses_varcap` genuinely can't fix the nonlinear catastrophe | **NO** — that was a **bug**; fixed, it now repairs it (44.06 → 1.157) |
| "Moment compatibility is intrinsically anti-correlated with causal error" | **NO** — mislabels a *studentized-screen power/variance confound*; the test itself is correctly calibrated |

---

## Where the two reviewers AGREE (high confidence)

1. **No result-corrupting code bug.** Review B verified, by running code:
   - `compute_phi` (STAR / doubly-robust pseudo-outcome `φ_k = 1{T=k} q_k (Y−h_k) + h_k`) — exact
     hand-match; recovers ATE ≈ 2.0 on `exact` with true bridges; reference arm aligned.
   - Bridge fitting — kernel bridges healthy (q positive, self-normalized to ~1, h correlates
     0.64–0.90 with Y); **sieve2/sieve3 degeneracy is REAL over-fitting, not a numerical bug**
     (this resolves Review A's open question Q1).
   - Feature extraction aligned (`feat[n]["var"] == rows[n]["po_var_mean"]`, etc., asserted live);
     no cross-candidate swap.
   - Cross-fitting clean (`coverage==1` enforced); `ObservedDatasetView` physically blocks
     `tau_true/ate_true`; truth enters only scoring. Oracle isolation is real.
   - Moment test correctly calibrated: under the true bridge, p-values ≈ uniform (mean 0.56 / 0.45,
     frac<0.05 ≈ 0).
   - Aggregation reproduces exactly (Review A independently reproduced 44.061 and 5.040).

2. **DGPs are valid, not degenerate** (Review A): arm-1 ATE = 1.0 everywhere (2.0 in exact/nonlinear);
   min per-arm mean propensity ≈ 0.23 (no positivity violation); proxies carry latent-U signal
   (canonical corr(W,V) ≈ 0.73); at least one candidate is near-oracle in every scenario, so
   "selection failure" is a *meaningful* concept. `hamd` uses the full real table n=1892 — no
   degenerate subsample.

3. **The failures are genuine in ABSOLUTE terms** — not denominator illusions:
   - `nonlinear` MSES: median **absolute** ATE error **1.53 on true ATE 2.0 (76% of the effect)**.
   - `hamd` `variance_only`: median absolute regret **≈0.25 on mean|ATE| 0.377 (66% of the effect)**.

4. **The `oracle_ratio` / "catastrophic > 1.5× best" metric is inflated** (Review A MAJOR, Review B
   MAJOR-M1). `oracle_best_error` is often tiny (nonlinear median 0.0196; 5–8/16 seeds < 0.02), so
   the ratio explodes and benign picks read as "catastrophic". The same ~0.25 absolute regret is
   reported as "5.4× / 94% catastrophic" in BOTH `exact` (ATE 2.0 → benign 12.5%) and `hamd`
   (|ATE| 0.38 → real 66%); `variance_only` `nonlinear` "37.5% catastrophic" is a 0.033 error =
   **1.6% of the effect** (essentially oracle).

## The one place they DISAGREE — and the reconciliation

**The moment-screen mechanism.**
- Review A (CRITICAL C1): the studentized screen's pass/fail is a **power artifact** — it studentizes
  each candidate by *its own* residual scale, so a wild over-fit candidate (huge residual variance)
  evades detection and **passes**, while a precise candidate's tiny remaining bias becomes
  "significant" and it **fails**. The screen ≈ inverse-variance in disguise. Live evidence
  (`nonlinear` seed 77000003): raw scale-free discrepancies are near-identical (D_h 0.017 vs 0.018)
  for a 0.00-error candidate and a 9.3-error candidate, but the screen flips them purely on the
  2500× variance difference. So my "moment compatibility is anti-correlated with causal error" label
  is wrong.
- Review B: the moment test is **correctly calibrated** (uniform p under the true bridge) and the
  power asymmetry is a **genuine statistical property, not a bug**.

**Reconciliation — both are right about different things.** The test is a *valid hypothesis test*
(correct size under H0 — Review B). But its *power* is confounded with residual variance, so using it
as a **hard selection screen** is pathological: pass/fail tracks precision, not bias, and
systematically admits high-variance over-fit candidates while rejecting precise slightly-biased ones.
This is **not a code bug and not a calibration error — it is a design flaw in using a studentized
moment test as a hard gate.** The correct statement of the Round-3 finding is therefore *sharper*, not
weaker: soft-weighting the moment signal beats hard-gating it because the soft score keeps the
variance/precision information the hard gate throws away.

---

## Concrete per-candidate evidence (Review B traces)

**`nonlinear`, seed 77000000** — the screen passes exactly the two worst candidates:

| candidate | ate_err | po_var | screen_pass | q_balance | max_q |
|---|---:|---:|:--:|---:|---:|
| kernel_kernel | 0.0086 | 10.9 | **False** | 0.022 | 9.1 |
| kernel_sieve1 (**oracle**) | **0.0064** | 29.1 | **False** | 0.192 | 36.8 |
| sieve2_sieve2 | **3.60** | 52 493 | **True** | 0.384 | 50.0 |
| sieve3_sieve3 | **3.70** | 70 714 | **True** | 0.696 | 50.0 |

MSES survivors = {sieve2, sieve3} (the two worst) → min-variance → **sieve2** (err 3.60). `variance_only`
→ kernel_kernel (0.0086, correct). The catastrophe is caused by the **screen gate**, and `variance_only`
is *right* here.

**`hamd`, seed 77000000** — variance is fooled by a collapsed-q bridge:

| candidate | ate_err | po_var | screen_pass | q_balance |
|---|---:|---:|:--:|---:|
| kernel_sieve1 (**oracle**) | **0.0066** | 28.8 | True | 0.140 |
| kernel_kernel | 0.1345 | 14.4 | True | 0.033 |
| sieve3_sieve3 | 0.3177 | **1.95** | **False** | **0.900** |

`variance_only` → sieve3 (global-min var 1.95, but its q collapsed: self-norm ≈0.17 ⇒ φ≈h ⇒ biased ATE,
err 0.32). The **screen and `q_balance` both correctly reject sieve3**; only pure variance is fooled.
Note the causes are *asymmetric*: in `nonlinear` the screen is wrong and variance is right; in `hamd`
the screen is right and variance is wrong.

---

## Fixes applied after the review

1. **BUG FIX — `mses_varcap` fallback** (Review B-M2). The code did `pool = capped or survivors`, so
   when the variance cap emptied the survivor set it fell back to the *survivor set* (= plain MSES),
   not the global minimum-variance candidate its docstring promised — making the cap toothless in
   exactly the D2 regime. Fixed to degrade to global min-variance when `capped` is empty. **Effect
   (post-hoc over the 16 n=2000 nonlinear seeds): median oracle-ratio 44.06 → 1.157; absolute error
   1.530 → 0.033.** `mses_varcap` now repairs the nonlinear catastrophe; other scenarios unchanged.
   Regression test added (`test_varcap_falls_back_to_global_when_survivors_all_exceed_cap`).
   *This overturns my earlier claim that "`mses_varcap` failed / the catastrophic survivors are
   low-variance" — they are HIGH variance (52k/70k), and the rule was simply implemented against its
   own spec.*

2. **METRIC FIX — effect-size normalization** (Review A-M2 / B-M1). The sweep now stores the true
   effect scale (`mean|ATE|`) and the aggregator reports, as the PRIMARY headline,
   `median_norm_regret = median(regret)/mean|ATE|`, `median_abs_regret`, `median_abs_error`, and a
   `norm_catastrophic` flag (`regret > 0.25 × effect`). `oracle_ratio` is retained but demoted and
   labelled denominator-inflated. Tables re-ranked by normalized regret.

3. **INTERPRETATION FIX** (Review A-C1). The Round-3 narrative doc is being corrected: the
   moment-screen failure is reframed as a *studentized-test power/variance confound used as a hard
   gate* (the test is calibrated), not an intrinsic anti-correlation of identifying moments; and the
   asymmetric causes of the two failures are stated explicitly.

## Minor / non-blocking

- `n_boot = 299` gives coarse p-value resolution (~1/300) near the screen threshold (Review A-m4);
  not the driver of any failure, but ≥999 is advisable for anything used to certify a decision.
- Sample size (n=2000, ~410–530/arm; 4-fold OOF) is adequate, not a flaw (Review A-m5) — the good
  candidates reach near-oracle error under these settings.
- Cosmetic STAR/STARSTAR naming inconsistency (Review B-M3), no functional impact.

## What this means for the Round-3 conclusions

- **The pipeline and datasets are sound; the qualitative story survives** but with corrected magnitudes
  and one corrected mechanism. MSES *is* catastrophic in nonlinear (screen-driven); `variance_only`
  *is* fooled in hamd (collapsed-q); q-aware / soft-fusion selectors *are* more robust.
- **Two of my write-up's claims were wrong and are being corrected**: `mses_varcap` is fixable (and now
  fixed), and the "moment anti-correlation" is really a hard-gate power/variance confound.
- **Severity was overstated**: replace "44× / 100% catastrophic" with effect-normalized regret
  (nonlinear MSES ≈ 0.76 of effect; hamd variance_only ≈ 0.66; the many "benign-but-flagged" cases
  drop out).
