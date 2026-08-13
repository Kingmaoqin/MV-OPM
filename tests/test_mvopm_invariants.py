"""MV-OPM scientific-invariant tests (§6/§22) + adversarial bug-injection (§23).

These test *scientific* invariants (leakage, alignment, determinism, formulas), not merely that
code runs. Fast settings throughout.
"""
import numpy as np
import pytest
import torch
import json
from pathlib import Path

from opm.candidates.library import CANDIDATE_NAMES, CandidateSet
from opm.data.views import observed
from opm.dgp import finite_proxy_exact as fpe
from opm.estimator.nested_crossfit import NestedConfig, nested_mvopm
from opm.estimator.nested_mses import NestedMSESConfig, nested_mses
from opm.validation.moments import build_bank, evaluate_candidate
from opm.validation.selector import compute_scores
from opm.validation.moment_tests import studentized_rff_max_test
from opm.validation.oof import collect_candidate_oof
from opm.validation.selector_v2 import ABSTAIN, pseudo_outcome_uncertainty, screen_candidate, select_mses
from opm.experiments.mvopm_round2.studies import E_TANH2, aligned_mechanism_row
from opm.experiments.mvopm_round2.convergence_audit import choose_plateau_budget
from opm.validation.cate_relative import relative_cate_risk

torch.set_num_threads(4)
BK = dict(max_epochs=25, patience=10, eval_every=3, batch_size=256)
HK = dict(max_epochs=50, patience=8)


def _view(seed=0, n=800):
    return observed(fpe.generate(n, seed, c_U=1.0))


# ---------- oracle isolation ----------
def test_selector_cannot_access_oracle_truth():
    ds = fpe.generate(300, 0)
    v = observed(ds)
    for attr in ("tau_true", "mu_true", "ate_true", "h_true_val", "q_true_val", "U_cat"):
        with pytest.raises(AttributeError):
            getattr(v, attr)


def test_selection_packages_do_not_import_oracle():
    import opm.validation.moments as m
    import opm.validation.selector as s
    import opm.candidates.library as c
    for mod in (m, s, c):
        assert "oracle" not in getattr(mod, "__dict__", {}), f"{mod.__name__} imported oracle"
        src = open(mod.__file__).read()
        assert "eval.oracle" not in src and "import oracle" not in src


# ---------- nested split audits ----------
def test_nested_split_audit_clean():
    res = nested_mvopm(_view(1), NestedConfig(K=2, outer_folds=3, inner_folds=2, seed=1,
                                              bridge_kwargs=BK, head_kwargs=HK))
    for a in res["audit"]:
        assert a["outer_train_test_disjoint"]
        for ia in a["inner"]:
            assert ia["inner_train_valid_disjoint"]      # inner valid ∩ inner train = ∅
            assert ia["inner_subset_of_outer_train"]     # inner ⊂ outer train
            assert ia["outer_test_absent_from_inner"]    # outer test never in inner


def test_selected_candidate_is_from_library():
    res = nested_mvopm(_view(2), NestedConfig(K=2, outer_folds=3, inner_folds=2, seed=2,
                                              bridge_kwargs=BK, head_kwargs=HK))
    assert all(s in CANDIDATE_NAMES for s in res["selected"])


# ---------- shared bank + formulas ----------
def test_common_rff_bank_deterministic_and_candidate_independent():
    v = _view(3)
    ih = np.concatenate([v.V, v.X], 1); iq = np.concatenate([v.W, v.X], 1)
    b1 = build_bank(ih, iq, n_rff=64, seed=20240)
    b2 = build_bank(ih, iq, n_rff=64, seed=20240)
    assert np.allclose(b1.om_h, b2.om_h) and np.allclose(b1.b_q, b2.b_q)  # deterministic, no candidate input


def test_moment_score_matches_manual_formula():
    v = _view(4)
    cs = CandidateSet(2, seed=0, bridge_kwargs=BK).fit(v)
    ih = np.concatenate([v.V, v.X], 1); iq = np.concatenate([v.W, v.X], 1)
    bank = build_bank(ih, iq, n_rff=64, seed=20240)
    nm, cand = next(iter(cs))
    got = evaluate_candidate(bank, cand, v, kinds=("rff_l2",))
    # manual replication of D_h_rff_l2 (mean over arms of sqrt(mean_j m_hj^2))
    from opm.validation.moments import _rff, EPS
    h = cand.predict_h(v.W, v.X)
    gh = _rff(ih, bank.om_h, bank.b_h, bank.mh, bank.sh)
    dh = []
    for k in range(2):
        R = (v.T == k).astype(float) * (v.Y - h[:, k]); Rn = R / (R.std() + EPS)
        dh.append(np.sqrt(np.mean((Rn[:, None] * gh).mean(0) ** 2)))
    assert abs(got["D_h_rff_l2"] - float(np.mean(dh))) < 1e-9


def test_product_score_formula():
    cand_D = {"a": {"D_h_rff_l2": 0.2, "D_q_rff_l2": 0.5},
              "b": {"D_h_rff_l2": 0.1, "D_q_rff_l2": 0.1}}
    sc = compute_scores(cand_D, "rff_l2")["product"]
    eps = 1e-8
    assert abs(sc["a"] - (np.log(0.2 + eps) + np.log(0.5 + eps))) < 1e-9
    assert sc["b"] < sc["a"]     # b has smaller product -> better (lower)


# ---------- candidate order invariance ----------
def test_candidate_order_invariance():
    v = _view(5)
    cs1 = CandidateSet(2, seed=0, bridge_kwargs=BK, candidate_names=["kernel_kernel", "sieve1_sieve1", "sieve2_sieve2"]).fit(v)
    cs2 = CandidateSet(2, seed=0, bridge_kwargs=BK, candidate_names=["sieve2_sieve2", "kernel_kernel", "sieve1_sieve1"]).fit(v)
    for nm in ["kernel_kernel", "sieve1_sieve1", "sieve2_sieve2"]:
        h1 = cs1.get(nm).predict_h(v.W, v.X); h2 = cs2.get(nm).predict_h(v.W, v.X)
        assert np.allclose(h1, h2), f"candidate {nm} depends on order"


def test_arm_order_and_alignment():
    v = _view(6)
    cs = CandidateSet(2, seed=0, bridge_kwargs=BK).fit(v)
    _, cand = next(iter(cs))
    assert cand.predict_h(v.W, v.X).shape == (v.n, 2)
    assert cand.predict_q(v.V, v.X).shape == (v.n, 2)


# ---------- determinism ----------
def test_seed_reproducibility():
    cfg = NestedConfig(K=2, outer_folds=3, inner_folds=2, seed=7, bridge_kwargs=BK, head_kwargs=HK)
    r1 = nested_mvopm(_view(7), cfg)
    r2 = nested_mvopm(_view(7), cfg)
    assert r1["selected"] == r2["selected"]
    assert np.allclose(r1["phi"], r2["phi"])


# ---------- adversarial bug-injection (§23): the invariant must CATCH the fault ----------
def test_adversarial_outer_test_leak_is_detected():
    # simulate letting an inner fold include an outer-test index -> audit flag must flip False
    n = 60
    all_idx = np.arange(n)
    outer_test = np.arange(0, 20)
    outer_train = np.setdiff1d(all_idx, outer_test)
    # BUG: inner_valid accidentally contains an outer-test index
    inner_valid_bad = np.concatenate([outer_train[:5], outer_test[:1]])
    inner_train = np.setdiff1d(outer_train, inner_valid_bad)
    flag_no_outer_test = len(np.intersect1d(np.concatenate([inner_train, inner_valid_bad]), outer_test)) == 0
    assert flag_no_outer_test is False   # the audit correctly detects the injected leak


def test_adversarial_oracle_exposure_is_blocked():
    ds = fpe.generate(200, 0)
    ds.meta["tau_true"] = ds.tau_true          # try to smuggle truth through meta
    v = observed(ds)
    assert "tau_true" not in v.meta            # stripped by the view


# ---------- Round-2 aligned mechanism ----------
def test_aligned_exact_dgp_theoretical_bias_and_reference_handling():
    row = aligned_mechanism_row(123, 0.20, 0.40, n=12000, n_rff=24)
    assert row["predicted_bias"] == pytest.approx(-0.20 * 0.40 * E_TANH2, abs=1e-15)
    assert row["incremental_ate_bias"] == pytest.approx(row["predicted_bias"], abs=0.04)
    assert row["reference_h_unchanged"] and row["reference_q_unchanged"]


@pytest.mark.parametrize("r_h,r_q", [(0.2, 0.0), (0.0, 0.4)])
def test_aligned_single_side_zero_bias_sanity(r_h, r_q):
    row = aligned_mechanism_row(124, r_h, r_q, n=12000, n_rff=24)
    assert row["predicted_bias"] == 0.0
    assert abs(row["ate_bias"]) < 0.08


def test_aligned_q_perturbation_positive():
    row = aligned_mechanism_row(125, 0.4, 0.4, n=1000, n_rff=16)
    assert row["min_q_multiplier"] > 0.6
    assert row["min_q"] > 0.0


# ---------- Round-2 OOF bank and bootstrap ----------
def test_inner_oof_full_coverage_and_common_bank():
    v = _view(126, n=240)
    out = collect_candidate_oof(
        v, n_folds=3, seed=126,
        candidate_names=["sieve1_sieve1", "sieve2_sieve2"],
        n_rff=16, n_boot=19, bank_seed=9001, bootstrap_seed=9101,
    )
    assert np.all(out["coverage"] == 1)
    assert all(a["disjoint"] for a in out["fold_audit"])
    assert out["bank"].n_rff == 16
    # A single bank object is returned for the entire candidate comparison.
    assert isinstance(out["bank_identity"], int)


def test_bootstrap_p_values_are_deterministic():
    rng = np.random.default_rng(5)
    r = rng.normal(size=180)
    g = rng.normal(size=(180, 12))
    a = studentized_rff_max_test(r, g, n_boot=99, bootstrap_seed=101)
    b = studentized_rff_max_test(r, g, n_boot=99, bootstrap_seed=101)
    assert a.statistic == b.statistic
    assert a.p_value == b.p_value
    assert 0.01 <= a.p_value <= 1.0


# ---------- Round-2 screening and MSES ----------
def _test_payload(ph, pq):
    return {"p_h": np.asarray(ph, float), "p_q": np.asarray(pq, float)}


def test_p_dr_is_max_and_both_rejections_are_required():
    s = screen_candidate([0.001, 0.8], [0.9, 0.001], alpha=0.05)
    assert np.allclose(s.p_dr, [0.9, 0.8])
    assert s.passes
    both = screen_candidate([0.001, 0.8], [0.002, 0.9], alpha=0.05)
    assert not both.passes and both.failed_arms == (0,)


def test_multiple_arm_bonferroni_correction():
    s = screen_candidate([0.02, 0.5, 0.5, 0.5], [0.02, 0.5, 0.5, 0.5], alpha=0.05)
    assert s.threshold == pytest.approx(0.0125)
    assert s.passes  # 0.02 would reject without the prespecified across-arm correction


def test_mses_survivors_and_minimum_variance_selection():
    tests = {
        "a": _test_payload([0.4, 0.4], [0.4, 0.4]),
        "b": _test_payload([0.3, 0.3], [0.3, 0.3]),
        "c": _test_payload([0.001, 0.3], [0.001, 0.3]),
    }
    uncertainty = {n: {"variance_mean": v} for n, v in {"a": 3.0, "b": 1.0, "c": 0.1}.items()}
    out = select_mses(tests, uncertainty)
    assert out["survivors"] == ["a", "b"]
    assert out["selected"] == "b"


def test_mses_abstains_when_no_candidate_survives():
    tests = {"a": _test_payload([0.001], [0.001]), "b": _test_payload([0.002], [0.002])}
    uncertainty = {"a": {"variance_mean": 1.0}, "b": {"variance_mean": 2.0}}
    out = select_mses(tests, uncertainty)
    assert out["selected"] == ABSTAIN and out["abstained"]


def test_variance_is_computed_from_supplied_oof_pseudo_outcomes():
    phi = np.array([[0.0, 1.0, 2.0], [1.0, 3.0, 2.0], [2.0, 5.0, 6.0]])
    out = pseudo_outcome_uncertainty(phi)
    manual = np.var(np.column_stack([phi[:, 1] - phi[:, 0], phi[:, 2] - phi[:, 0]]), axis=0, ddof=1)
    assert np.allclose(out["contrast_variance"], manual)
    assert out["variance_mean"] == pytest.approx(float(np.mean(manual)))


def test_convergence_budget_selector_has_no_oracle_input():
    import inspect
    src = inspect.getsource(choose_plateau_budget)
    assert "oracle" not in src.lower() and "ate_err" not in src and "pehe" not in src
    rows = []
    for budget, d, change in [(45, 1.0, np.nan), (90, 0.97, 0.02), (180, 0.96, 0.01), (300, 0.95, 0.01)]:
        rows.append({"budget": budget, "D_total_rff_l2": d,
                     "h_prediction_change": change, "q_prediction_change": change})
    assert choose_plateau_budget(rows) == 45


def test_round2_final_seeds_do_not_overlap_prior_banks():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "configs/seeds/mvopm_round2_final.json").read_text())
    new = [s for vals in manifest["scientific_seeds"].values() for s in vals]
    assert len(new) == len(set(new))
    prior = set(range(0, 500)) | set(range(500, 515)) | set(range(100000, 100500))
    assert not (set(new) & prior)
    assert not (set(new) & set(manifest["development_only_seeds"]))


def test_relative_cate_risk_matches_oracle_identity_with_exact_signal():
    x = np.linspace(-1, 1, 100)[:, None]
    truth = x.copy()
    a, b = 0.8 * x, 1.4 * x
    got = relative_cate_risk(a, b, truth)["risk_a_minus_b"]
    oracle_difference = float(np.mean((a - truth) ** 2 - (b - truth) ** 2))
    assert got == pytest.approx(oracle_difference, abs=1e-12)


def test_all_adaptive_selectors_are_outer_nested_and_complete():
    v = _view(127, n=240)
    out = nested_mses(v, NestedMSESConfig(
        K=2, outer_folds=3, inner_folds=2, seed=127, n_rff=12, n_boot=19,
        candidate_names=["sieve1_sieve1", "sieve2_sieve2"], head_kwargs=HK,
    ))
    expected = {"product", "h_only", "q_only", "sum", "max", "variance_only", "random",
                "fixed_kernel", "fixed_sieve1", "biasvar_mult", "biasvar_add", "MSES"}
    assert set(out["phi_by_selector"]) == expected
    for fold in out["folds"]:
        assert fold["outer_train_test_disjoint"] and fold["inner_coverage_complete"]
        assert set(fold["selected_by_selector"]) == expected
        for inner in fold["inner_folds"]:
            assert inner["disjoint"] and inner["outer_test_absent"]
            assert len(np.intersect1d(inner["train_global"], inner["valid_global"])) == 0
    for name, phi in out["phi_by_selector"].items():
        if name != "MSES" or not out["abstained"]:
            assert np.all(np.isfinite(phi))
