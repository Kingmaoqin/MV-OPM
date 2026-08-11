# MV-OPM — Phase 0 Project Audit (no source changes)

**Governing prompt:** `固定的主线和修补指南` (MASTER AGENT PROMPT — MV-OPM).
**Phase:** 0 (audit only). **No source code has been modified.** Only this document and the
`docs/mvopm/` folder were created.

**New thesis (frozen target of the pivot):**
> Proximal heterogeneous-effect estimation needs not only flexible bridge estimation but an
> *observed-data* mechanism for deciding which learned bridge solution is reliable. **MV-OPM**
> uses **held-out proximal identifying-moment violations** to diagnose, **select**, and when
> necessary **reject** candidate proximal estimators — without any counterfactual validation
> labels (no `tau_true`, `mu_true`, PEHE, oracle ATE).

The existing repo is an OPM-v2 "proximal DR CATE learner" (10 experiments, 9/10 PASS). This
audit maps what is reusable, what is debt, what must be built vs frozen, the dependency graph,
cost, risks, and preliminary Reviewer-A/B concerns, then determines whether Phase 1 is authorized.

---

## 1. Answers to the seven Section-0 audit questions

**Q1. What code is already reusable?**
- **Bridge solvers** (become MV-OPM *candidates*): `opm/bridges/kernel_moment.py::KernelBridges`
  (kernel/MMR h **and** q), `opm/bridges/sieve.py::SieveBridges(degree=d)` (sieve h **and** q,
  deg 1/2/3). Both expose the same `predict_h(W,X)/predict_q(V,X) → (n,K)` API — the seed of the
  common candidate interface. **Gap:** they pair h and q *within one solver*; MV-OPM also needs
  **mixed** candidates (`kernel_sieve1`, `sieve1_kernel`), which require a new composition wrapper.
- **Pseudo-outcome + Stage-2**: `estimator/pseudo_outcome.py::compute_phi` (STAR), `estimator/
  tau_head.py` (τ head, `ate_with_ci`), `estimator/learner.py::OPM` (single-level OOF pipeline).
  Reusable as the *downstream* estimator once a candidate is selected.
- **Cross-fitting**: `estimator/crossfit.py::CrossFitter` (single level, leakage-proof, unit-tested).
  Reusable as the **inner or outer** primitive; nested orchestration must be built on top.
- **Diagnostics** (see Q2).
- **DGPs**: `linear_gaussian` (closed-form h_true only), `synthetic_main` (S1/S2 + corruption),
  `semisynth` (real HAMD X), `nonlinear_bridge` (E9). Reusable as scenarios. **Gap:** none has an
  exact *q_true*; the exact-bridge DGP (§9 of the prompt) must be built.
- **Eval/stats**: `eval/stats.py` (paired Wilcoxon/t, BH, bootstrap), `eval/metrics.py` (PEHE, ATE
  err, policy value). Reusable in the **oracle evaluation package only**.
- **Real data**: `data/rhc.py` (RHC proximal loader), `data/hamd.py`. Reusable for Study H.

**Q2. What functionality already exists for bridge residual diagnostics?**
This is the most important reuse and it is *substantial but not yet solver-independent*:
- `opm/bridges/kernel_moment.py::_val_residual` — computes, on a validation split, for each arm k,
  the **RFF-max** score `max_j |E_n[R g_j]| / sd(R)` for the outcome side (R = 1{T=k}(Y−h_k)) and
  the analogous q side (S = 1{T=k}q_k − 1). **This is exactly the RFF-max diagnostic (prompt §5B).**
  **Two problems for MV-OPM:** (i) the RFF bank is **candidate-specific** (`_rff` seeds from the
  bridge's own `cfg.seed`), so scores are **not comparable across candidate families**; (ii) it is
  computed on the candidate's *own* validation split and used for early stopping — using the same
  statistic to also *rank* candidates would be the unfair-comparison failure the prompt warns about (§5).
- `opm/diagnostics/diagnostics.py`:
  - `bridge_residual_score(R, inst, bw)` — standalone RFF-max score. Reusable core of the validator.
  - `q_sanity` — `En[1{T=k}q_k]`, max q, ESS (Kish). Reusable as competing heuristics (Study C rows 8).
  - `proxy_strength` — residual-CCA `rho_min` (proxy strength). Reusable as competitor (Study C row 7).
  - `arm_overlap`. Reusable diagnostic.
- `opm/experiments/e8.py` — **already correlates `bridge_resid` and `rho_min` with PEHE across
  corruption levels.** This is a *single-candidate prototype of Observation D* (moment violations
  predict causal error). It must be generalized to **multiple candidate families** and turned into a
  *selection* signal, not just a correlation.
- **Missing:** RFF-**L2** aggregate (`sqrt(mean_j m_j²)`, prompt §5A); a solver-independent **kernel/MMR**
  discrepancy validator (§5C); a **shared fixed feature bank** across candidates; residual normalization.

**Q3. Which experiments directly support the new thesis?**
- **E6(f)** (`experiments/e6.py`, c_U ∈{0,0.5,1,2}) → **Observation A** (hidden confounding creates a
  failure X-adjustment can't remove; proximal advantage grows with c_U). Directly reusable/strengthen → Study D.
- **E9** (`experiments/e9.py`, nonlinear DGP; kernel≈sieve, sieve1 best on PEHE, sieve2/3 unstable) →
  **Observation B** (flexibility ≠ causal reliability). This is now **motivating evidence for validation**,
  not an embarrassment. Reusable → Figure 1 / Study B.
- **E8** (diagnostic–PEHE correlation) → **Observation C/D** prototype. Reusable → Study B foundation.
- **E3/E5** (candidate families kernel vs sieve vs neural) → candidate library exists.

**Q4. Which experiments are secondary?**
- **E1** (consistency/rates) — supporting sanity, not the thesis.
- **E4** (RHC) — becomes **Study H** (real data, no oracle; report stability/diagnostics).
- **E2** (product-bias) — *conceptually central* (product mechanism) but its current implementation is
  debt (uses estimated q as q_true). It is **superseded by the exact-bridge DGP Study A**; archive.
- **E10** (CATE inference) — secondary capability; explicitly **NOT the main novelty** (prompt §3).
  Retain in appendix only; if reported, needs ≥100–200 reps/n (prompt §12), not 6 seeds.

**Q5. Which modules must be frozen rather than extended?**
- **`opm/generator/*` (DDPM, Gaussian-MLP) + E7** — diffusion is explicitly *out of the story* (prompt
  §3, §29.14). Freeze/archive; do not extend; not in Figure 1/abstract/title.
- **`opm/estimator/cate_inference.py` + `experiments/e10.py`** — secondary; freeze as-is (appendix).
- **Solver internals** `kernel_moment.py` (training loop) and `sieve.py` (2SLS) — freeze the *training*
  logic for reproducibility; MV-OPM wraps their `predict_h/predict_q` behind the candidate API rather
  than modifying them. (The early-stopping `_val_residual` must **not** be repurposed as the
  cross-candidate selector — a *separate* validator is built.)

**Q6. Which existing claims are no longer appropriate under the new framing?**
- "OPM (kernel) beats every ignorability baseline" (E3/E5) — true but **not the thesis**; demote to a
  motivating/background result. Do **not** headline a "better CATE learner" (prompt §29.12/13).
- Any implication that the **kernel bridge is superior** — prohibited (§29.5); E9 shows sieve1 wins on
  PEHE. `POSITIONING.md` already states this honestly — that stance is now the *asset* that motivates
  MV-OPM (validation matters *because* no single solver dominates).
- E2's "product-bias verified" — weakened by the q_true-proxy limitation; re-establish only via the
  exact-bridge DGP.
- Nothing may be called "first"/"near-oracle" until the literature matrix (§28) and the quantitative
  evidence exist.

**Q7. Which current experiments contain statistical or implementation debt?**
- **E2**: uses `q_hat_bestn` as a stand-in for the (closed-form-unavailable) `q_true`; contaminates the
  pure product rate (documented in `DECISIONS.md`). → replace with exact DGP.
- **E10**: 6-seed coverage reported; prompt §12 forbids treating this as strong CI evidence. → ≥100–200 reps if kept.
- **E8**: diagnostic–PEHE correlation is **noisy/unstable** (per-point Spearman swung 0.61→0.26 between 5
  and 8 seeds; `rho_min` sign flips by family). Single-candidate only. → the MV-OPM validator must use
  shared instruments, more seeds, and be evaluated *across candidate families*.
- **RFF bank not shared across candidates** (`_rff` per-bridge seed) → unfair cross-candidate ranking.
- **No nested cross-fitting** — selection adaptivity is unaccounted for; single OOF is insufficient (§7).
- **No oracle isolation** — `CausalDataset` exposes `tau_true/mu_true/ate_true/cf_mean` directly; a
  selector could read them. Needs `ObservedDatasetView` + guard test (§8, §22).
- **Candidate mixing unsupported** — cannot currently build `kernel_sieve1` / `sieve1_kernel`.

---

## 2. Current architecture map (file → role → MV-OPM disposition)

| Module | Role today | MV-OPM disposition |
|---|---|---|
| `bridges/kernel_moment.py` | kernel h+q solver + RFF early-stop | **candidate** `kernel_kernel`; freeze training; extract residual logic into shared validator |
| `bridges/sieve.py` | sieve h+q (deg d) via 2SLS | **candidates** `sieveD_sieveD`; freeze |
| `bridges/plugin.py` | dr_fallback (AIPW) | keep as no-proxy reference/competitor |
| `bridges/networks.py` | TrunkNet, EMA | freeze (used by kernel candidate) |
| `estimator/crossfit.py` | single-level OOF folds | **reuse as inner/outer primitive** under new nested orchestrator |
| `estimator/pseudo_outcome.py` | STAR | reuse (outer-OOF pseudo-outcomes) |
| `estimator/tau_head.py` | τ head, ATE-CI | reuse downstream |
| `estimator/learner.py::OPM` | full single-level pipeline | reuse for *fixed-candidate* baselines; new nested selector sits above |
| `estimator/cate_inference.py` | local-linear CATE CI | **freeze** (secondary/appendix) |
| `diagnostics/diagnostics.py` | RFF-max, q_sanity, proxy CCA, overlap | **core reuse** for validator + competitor scores |
| `dgp/*` | S1/S2/nonlinear/semisynth/linear-gaussian | reuse as scenarios; **build exact-bridge DGP** |
| `generator/*`, `experiments/e7.py` | diffusion | **FREEZE/archive** |
| `experiments/e1..e10.py` | chronological experiments | archive; **reorganize into Studies A–H** |
| `eval/*` | metrics, stats | **oracle-only** package boundary must be enforced |
| `data/rhc.py`, `data/hamd.py` | real loaders | reuse (Study H, semisynth-X) |

---

## 3. Components that MUST be BUILT (Phase 2, after spec + review approval)

1. **Common candidate API + library** — `opm/candidates/` : a `Candidate` protocol exposing
   `fit(view)`, `predict_h(W,X)`, `predict_q(V,X)`; registry of
   `kernel_kernel, sieve1_sieve1, sieve2_sieve2, sieve3_sieve3, kernel_sieve1, sieve1_kernel`
   (mixed = compose an h from one solver with a q from another). Unit test: **candidate-order invariance**.
2. **Solver-independent moment validator** — `opm/validation/moments.py` : `D_h_L2, D_q_L2` (RFF-L2),
   `RFF-max`, and a kernel/MMR discrepancy; **one shared fixed random-feature bank** per comparison;
   documented residual normalization. Evaluated only on data **not** used to fit the candidate.
3. **Selector + prespecified scores** — `opm/validation/selector.py` : `score_h, score_q,
   score_sum=std(D_h)+std(D_q), score_max, score_product=log(D_h+ε)+log(D_q+ε)`; **product is the
   prespecified primary**. No post-hoc score changes.
4. **Nested cross-fitting** — `opm/estimator/nested_crossfit.py` : outer/inner splits; inner = fit
   candidates → validate → rank/select; outer = refit selected on full OUTER_TRAIN → outer-OOF STAR →
   downstream τ only after all outer folds. **Explicit index-audit logs.**
5. **Oracle isolation** — `opm/data/views.py::ObservedDatasetView` (no truth fields) + `opm/eval/oracle.py`
   (truth lives here; selector package must not import it). Guard test.
6. **Exact-bridge DGP** — `opm/dgp/finite_proxy_exact.py` : finite/discrete latent U + full-rank
   transition matrices so **both `h_true` and `q_true`** are solvable from population equations;
   controllable completeness; latent confounding; known ATE/CATE; supports `h_hat=h_true+r_h·δ_h`,
   `q_hat=q_true+r_q·δ_q`. Replaces E2.
7. **Provenance + anti-leakage tests** — the §22 test suite (13 tests) + §23 adversarial bug-injection.
8. **Governance docs** — `docs/MV_OPM_SPEC.md`, `docs/CLAIM_EVIDENCE_LEDGER.md`,
   `docs/LITERATURE_NOVELTY_MATRIX.md`, `docs/reviews/{REVIEW_A_METHODS,REVIEW_B_ENGINEERING,CROSS_RECONCILIATION}.md`.

## 4. Components that MUST be FROZEN
`opm/generator/*`, `experiments/e7.py`, `estimator/cate_inference.py`, `experiments/e10.py`,
and the *training* internals of `kernel_moment.py` / `sieve.py` (wrap, don't edit). Existing E1–E10
scripts/results are **archived, not deleted** (prompt §29.8).

---

## 5. Proposed implementation dependency graph

```
ObservedDatasetView ─┐
exact_proxy DGP ─────┼─▶ Candidate API ──▶ (fit on INNER_TRAIN)
shared RFF/MMR bank ─┘                         │
                                               ▼
                        Moment Validator (D_h, D_q on INNER_VALID, shared bank)
                                               │
                                               ▼
                        Selector scores {h,q,sum,max,product}  ──▶ pick candidate
                                               │
        (refit selected on full OUTER_TRAIN) ──┼──▶ predict h/q on OUTER_TEST
                                               ▼
                        outer-OOF STAR pseudo-outcomes ──▶ τ head (downstream)
                                               │
                                               ▼
                 oracle eval (separate pkg): PEHE / ATE err / regret / oracle-best
```
Build order: (1) ObservedView + oracle split → (2) candidate API + order-invariance test →
(3) validator + shared bank → (4) nested cross-fit + index audit → (5) exact DGP → (6) selector +
scores → (7) anti-leakage tests → (8) DEV pilot.

---

## 6. Computational-cost estimate by experiment family (CPU, shared A100 host)

Unit cost anchors (measured this project): one **kernel** bridge fit ≈ 30–120 s (n=2k–8k); one
**sieve** fit ≈ 1–3 s; τ head ≈ 5–15 s. Nested CF multiplies by (outer folds × inner folds × candidates).

| Family | Rough driver | Estimate |
|---|---|---|
| **Study A** (exact DGP, product mechanism, 200 reps) | candidates cheap on finite DGP; mostly arithmetic | **low–moderate** (~1–2 h) |
| **Study B/C** (rank/select, 4 scenarios × 50 seeds × ~6 candidates × nested CF) | dominated by **kernel** candidate × folds | **very high** (tens of hours) — the main cost; needs parallelism + caching |
| **Study D** (c_U sweep, 4 levels × 30 seeds) | as B/C per config | **high** |
| **Study E** (proxy corruption, 4 kinds × 4 levels × 30) | as B/C | **very high** |
| **Study F** (n sweep, 4 n × 30, ≥2 DGPs) | grows with n | **high** |
| **Study G** (failure/abstention, several breaks × 50) | as B/C | **high** |
| **Study H** (RHC, 30 splits) | real n=5735, ~6 candidates | **moderate** |

**Implication:** the pilot-first discipline (§10/§30) is essential. Full matrix is **not** runnable
naively on this shared host; requires candidate/DGP caching, single-GPU-sequential + watchdog, and
possibly reduced candidate set or seeds in exploratory grids. A cost gate must precede Study B–G.

---

## 7. Risks that could invalidate the new thesis

- **R1 (core-hypothesis risk).** Held-out moment violations may **not** reliably predict causal error
  across solver families. E8 already shows the correlation is noisy and one diagnostic (`rho_min`)
  flips sign by corruption family. If Observation D is weak, the thesis fails → the Go/No-Go gate must
  catch this and set HOLD.
- **R2 (product-score risk).** The product score is motivated by the DR remainder `~‖δ_h‖·‖δ_q‖`, but
  (a) moment violation is a **weak norm** that may not control the **strong (L2) bridge norm** in an
  ill-posed inverse problem (theory T2 caveat); (b) `D_h` and `D_q` have different scales/noise, so the
  product may be dominated by one side. Product might not beat `h-only`/`q-only`/fixed-kernel.
- **R3 (adaptivity/leakage).** Selection adds adaptivity; without correct nested CF and oracle
  isolation, apparent success could be leakage. Mitigated by §7/§8/§22 but must be *verified*, not assumed.
- **R4 (scale/normalization).** Cross-candidate ranking is sensitive to residual normalization; a wrong
  normalization can trivially determine the winner (prompt §5 warns).
- **R5 (compute).** The confirmatory matrix may be infeasible on the shared host within reasonable time;
  risks pressure to cut corners. Mitigated by pilot-first + cost gate.
- **R6 (interpretation).** Temptation to call moment adequacy "proof of proxy validity" (§29.11) — must
  use falsification language only.

---

## 8. Reviewer A (methodological) — preliminary concerns

- **A-1 [MAJOR].** The existing residual diagnostic (`_val_residual`) uses a **candidate-specific** RFF
  bank and is the candidate's own early-stopping statistic; using it to rank candidates is unfair.
  Require a **shared, fixed** instrument bank in a solver-independent validator (spec must fix this).
- **A-2 [MAJOR].** E2's product-mechanism evidence relies on estimated q as q_true → not a clean test.
  The exact-bridge DGP is required before any product-mechanism claim (Study A).
- **A-3 [MAJOR].** Selection adaptivity ⇒ **nested** cross-fitting is mandatory; the current single-level
  OOF cannot support an unbiased evaluation of the *selected* estimator.
- **A-4 [MINOR].** Prespecify the primary score (**product**) and the secondary comparison set
  {h,q,sum,max,product} *before* the pilot; no post-hoc score invention (§6/§11).
- **A-5 [NOTE].** Language: "empirical support / falsification", never "proves proxies valid".
- **A-6 [NOTE].** Theory (T1–T4) must not be asserted; if unproven, frame as an empirical validation framework.

## 9. Reviewer B (engineering) — preliminary concerns

- **B-1 [CRITICAL, latent].** `CausalDataset` exposes `tau_true/mu_true/ate_true/cf_mean` directly; the
  selector could read oracle truth. Must introduce `ObservedDatasetView` and a test that **fails** if
  oracle fields are reachable from the selection API (§8/§22).
- **B-2 [MAJOR].** RFF banks are seeded per bridge (`_rff` uses `cfg.seed+key`); ranking could depend on
  candidate identity/order. Need one shared bank + `test_common_rff_bank_across_candidates` +
  `test_candidate_order_invariance`.
- **B-3 [MAJOR].** No nested split index audit exists; must log INNER⊂OUTER_TRAIN, INNER_VALID∩INNER_TRAIN=∅,
  OUTER_TEST never touched by selection/standardizers/bandwidth. Adversarial bug-injection (§23) required.
- **B-4 [MINOR].** Mixed candidates (kernel h + sieve q) need careful arm/tensor alignment and q-clipping;
  add `test_arm_order_preserved`, `test_moment_score_matches_manual_formula`, `test_product_score_formula`.
- **B-5 [MINOR].** Provenance: current experiments save `raw.csv` but not per-run config/seed/split-IDs;
  §17 structure `results/mvopm/<study>/<config>/<seed>/` + `test_raw_results_reconstruct_summary` needed.
- **B-6 [NOTE].** Backward compatibility: freezing solvers while wrapping them; keep the 9/9 unit tests green.

---

## 10. Old → new story mapping

| Old | New role |
|---|---|
| E6(f) confounding sweep | **Finding 1 / Study D** — hidden confounding raises the value of proximal + validation |
| E9 kernel≈sieve, sieve wins | **Finding 2 / Figure 1 / Study B** — flexibility ≠ reliability (motivates validation) |
| E8 diagnostic↔PEHE | **Finding 3 / Study B** — moment violations predict causal error (generalize to candidate families) |
| (new) product-moment selection | **Finding 4 / Study C** — product validation selects near-oracle candidate |
| (new) broken-proxy detection | **Finding 5 / Study G** — falsification / abstention |
| E4 RHC | **Finding 6 / Study H** — stability on real data |
| E2 | archived; replaced by exact-DGP **Study A** (product mechanism) |
| E7 diffusion, E10 inference | frozen; appendix/archive only |

---

## 11. STATUS REPORT (prompt §31 format)

**## Status** — **PASS (Phase 0 complete; Phase 1 authorized, not yet started).**

**## Completed** — Repository read and audited (every `opm/*.py`, tests, configs, results, and the
four `docs/*.md`). Produced this `docs/mvopm/PHASE0_AUDIT.md`: architecture map, reusable-component
inventory (esp. the existing RFF-max bridge diagnostics and the E8 correlation prototype), MV-OPM
technical-debt list, build-vs-freeze file plan, dependency graph, per-family compute estimate,
thesis-invalidating risks, and preliminary Reviewer-A/B concerns. **No source code modified.**

**## Scientific finding so far** — The pivot is well-supported by *existing* assets: bridge-moment
residual diagnostics already exist (RFF-max), and E8/E9 already prototype the two load-bearing
observations (moment violations correlate with error; flexible solvers don't dominate). But every
selection-critical safeguard the thesis needs — solver-independent validator with a *shared* feature
bank, *nested* cross-fitting, *oracle isolation*, and an *exact-bridge* DGP with a real `q_true` — is
**absent** and must be built before any confirmatory claim.

**## Reviewer A** — CRITICAL: none. MAJOR: A-1 (shared validator bank), A-2 (exact DGP for product
mechanism), A-3 (nested CF). MINOR: A-4 (prespecify product primary). NOTE: A-5, A-6 (language/theory).

**## Reviewer B** — CRITICAL: B-1 (oracle reachable from dataset → must isolate) — *latent*, becomes
CRITICAL the moment selector code is written. MAJOR: B-2 (shared RFF bank / order invariance), B-3
(nested index audit + adversarial tests). MINOR: B-4, B-5. NOTE: B-6.

**## Reconciliation** — No CRITICAL/MAJOR items are *code-blocking yet* because no selection code
exists; they are **binding requirements on the Phase-1 spec and Phase-2 build**. B-1 must be resolved
in the very first Phase-2 commit (oracle isolation) and A-1/A-3 must be fixed in the validator/nested-CF
design. Formal two-independent-reviewer *agent* cross-audit (prompt §18–21) is a **Phase-1/Phase-2 gate**
and has **not** been run (the earlier attempt to spawn reviewer agents hit a session limit); it must be
completed and PASS before Phase-2 implementation is authorized.

**## Next authorized action** — Phase 1: write `docs/MV_OPM_SPEC.md` (paper identity, frozen primary
score = product, candidate set, metric definitions, pilot Go/No-Go gate with pre-registered thresholds,
seed policy), plus the empty ledger/literature-matrix scaffolds — **documentation only, no experiments,
no algorithm code**. Then run the two-reviewer spec review and record `CROSS_RECONCILIATION.md`. Phase-2
implementation is **blocked** until that review PASSes.

---

### Phase 0 → Phase 1 authorization decision
Per the protocol, Phase 0 requires only a completed audit with no code changes — **satisfied**.
**Phase 1 (specification) is AUTHORIZED.** Phase 1 remains documentation-only; the expensive Phase-2
build and any experiment run stay **gated** behind the specification review. I will **stop here** and
await confirmation to proceed into Phase 1, per the prompt's "begin with Phase 0 only … then stop"
instruction and because the program's later phases carry large, hard-to-reverse compute commitments.
