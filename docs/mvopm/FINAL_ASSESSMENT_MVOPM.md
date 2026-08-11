# MV-OPM — Final Scientific Assessment (Pilot phase; STATUS = HOLD)

This answers the MV-OPM §16 questions from the completed pilot evidence. The Go/No-Go gate returned
**HOLD** (`results/mvopm/FAILURE_ANALYSIS.md`); per protocol the expensive final matrix was NOT run.
All numbers reconstruct from `results/mvopm/raw.csv` + `raw_candidates.csv`; reviews and the frozen
gate are in `docs/reviews/`.

## Direct answers to §16 A–H

**A. Do held-out identifying-moment violations predict causal error?**
*Bias — YES; per-instance CATE selection — no.* On the exact-bridge DGP the moment discrepancies
track the true bridge errors almost perfectly (`Spearman(D_h,‖ĥ−h‖)=0.86`, `D_q=0.90`,
`moment_product↔true_err_product=0.83`). But across candidate families the moment product ranks
*CATE* error only weakly (Spearman 0.03–0.45) because it is variance-blind (see D below).

**B. Does product-moment validation beat h-only / q-only / simple heuristics?**
*Yes vs weak baselines; not vs the strong one.* Median oracle-ratio (ATE): product 1.2–1.5 <
h-only 1.6–20, q-only 1.3–6.4, random 1.4–2.4, and ≪ fixed-kernel 3–14. But product does **not**
beat fixed-sieve1 (≈1.0–1.3), which is often the oracle-best.

**C. How close to oracle selection?**
Product median oracle-ratio ≈ **1.3** (ATE); oracle = 1.0; fixed-kernel ≈ 5. Substantially better
than naive selection, but ~30% above oracle and with a 0.4–0.5 catastrophic-selection rate — not
near-oracle.

**D. When does proximal adjustment materially beat ignorability?**
Reconfirmed from the prior suite (E6f/E9, archived): under nonlinear/stronger hidden confounding
proximal cuts ATE bias ~4× vs ignorability. In MV-OPM's candidate view, the *best proximal candidate*
is far better than fixed-kernel; the open problem is *choosing* it from observed data.

**E. When do proxies become too weak/corrupted?**
Study G (additive/missing s∈{0,.1,.2,.3}): product oracle-ratio and `min_total_D` are flat — mild
corruption barely moves the selection benefit or the moment level at this scale.

**F. Can the method detect failure regimes (abstention)?**
*No, as currently defined.* Study I: `min_total_D` for deliberately broken proxies (shuffle_W /
noise_V / weaken) is indistinguishable from valid proxies (0.079 vs 0.076). The normalized moment can
still be satisfied by some candidate, so it does not falsify broken-proxy regimes. Honest negative
result; the variance-aware direction (D/exploratory) is the natural fix.

**G. Does it persist across settings?**
The *bias-measurement* result is robust (mechanism DGP). The *selection-vs-naive* benefit persists
across S1/S2/nonlinear/HAMD. The *near-oracle selection* and *abstention* claims do **not** hold in
any family. **Real RHC (Study J, 25 repeated splits, no oracle): the product selector chose
`sieve1_sieve1` in 100% of splits (perfectly stable), giving ATE = −2.07 days (sd 0.70; median CI
[−2.93, −0.97]; ESS 2104; max q 18.7)** — literature-consistent (RHC shortens survival) and
well-behaved. So the *stability* of the selected estimator on real data is a positive result, even
though synthetic near-oracle selection is not achieved. (Repeated splits are algorithmic stability,
not independent replications.)

**H. Strongest legitimate final thesis supported by the evidence**
> **Held-out proximal identifying-moment violations are a reliable, counterfactual-label-free measure
> of proximal BRIDGE bias, and thus a principled bridge-ADEQUACY / falsification diagnostic. They are
> NOT, on their own, sufficient to select reliable heterogeneous-effect estimators, because CATE
> error is bias + variance and the moment is variance-blind. A bias×variance validator is the
> prespecified hypothesis for confirmation.**
This is a *model-validation framework* contribution, explicitly not a "near-oracle selector" or a
"better bridge" claim.

## Exploratory validation of the prereg direction (POST-HOC — NOT confirmatory)
Run on **fresh seeds 500–514** (disjoint from the pilot dev bank 0–19 that generated the
hypothesis), primary target = ATE error (`results/mvopm/EXPLORATORY_BIASVAR.md`,
`scripts/mvopm_exploratory.py`). Adding an observed-data variance signal (OOF pseudo-outcome
contrast variance) to the moment diagnostics:

| score | median oracle-ratio (ATE) | mean Spearman(score, ATE err) |
|---|---|---|
| product (prereg, bias only) | 1.26 | 0.23 |
| **biasvar (bias × / + variance)** | **1.04** | **0.73** |
| variance_only | mixed (fails nonlinear/HAMD) | 0.11–0.51 |
| fixed_sieve1 (strong baseline) | 1.03 | 0.41 |

Per-scenario rank Spearman rises to **0.70 / 0.73 / 0.75 / 0.73** (S1/S2/nonlinear/HAMD; all ≥0.70,
vs 0.09–0.44 for product-alone), and the oracle-ratio falls to ~1.04 (**near-oracle**), matching the
strong fixed-sieve1 baseline *as a data-driven selector*. `variance_only` alone is insufficient — it
is the **combination** of identifying-moment adequacy (bias) and estimation variance that works,
consistent with error = bias + variance. **This is exploratory (the score was chosen after the
pilot); it cannot be reported as confirmatory (§15/§29), but it strongly de-risks a preregistered
confirmatory study.** Honest caveat: sieve1 is oracle-best in most tested scenarios, so beating
fixed-sieve1 requires settings where the oracle-best candidate *varies* (e.g. regimes where kernel or
mixed candidates win) — a required ingredient of the confirmatory design.

## Is this ICLR-main-track strong as-is?
**No — HOLD.** As it stands the confirmed contribution (moment = valid bridge-bias diagnostic) is
sound and useful but narrow, and the headline application (near-oracle proximal-estimator selection)
is **not** supported by the pilot; the abstention capability fails. A main-track submission would need
the prespecified follow-up to succeed:
1. a **bias×variance** validator (moment adequacy + estimation-variance signal) that achieves
   near-oracle selection in a *fresh, preregistered* confirmatory study (pilot exploratory Spearman
   0.67–0.75 is promising but post-hoc);
2. a working **falsification/abstention** test for unsupported proxy regimes;
3. ideally a theory linking held-out conditional-moment discrepancy (a weak norm) to bridge error
   under the ill-posed inverse problem (MV-OPM §27 T1–T3), or an explicit statement that the
   contribution is empirical model-validation.

## What was delivered (Phase 2–3, honest scope)
- Full MV-OPM pipeline: common candidate library (kernel/sieve1-3 + mixed), **solver-independent
  shared-bank moment validator** (RFF-L2/RFF-max/MMR), prespecified selector (product primary),
  **nested cross-fitting** with index audit, **oracle isolation** (`ObservedDatasetView`),
  **exact-bridge DGP** with real `h_true`/`q_true` (replaces E2's debt).
- 12 scientific-invariant tests + adversarial injection (21/21 total pass); the suite **caught and
  forced a fix** to a real determinism bug.
- Two isolated reviews + cross-reconciliation; frozen Go/No-Go gate.
- Local cluster harness (manifest/worker/launcher/aggregator, full provenance) + 394-row pilot +
  gate report + this failure analysis.
- Everything reconstructs from raw per-run files.
