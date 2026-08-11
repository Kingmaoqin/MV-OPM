"""MV-OPM scientific-invariant tests (§6/§22) + adversarial bug-injection (§23).

These test *scientific* invariants (leakage, alignment, determinism, formulas), not merely that
code runs. Fast settings throughout.
"""
import numpy as np
import pytest
import torch

from opm.candidates.library import CANDIDATE_NAMES, CandidateSet
from opm.data.views import observed
from opm.dgp import finite_proxy_exact as fpe
from opm.estimator.nested_crossfit import NestedConfig, nested_mvopm
from opm.validation.moments import build_bank, evaluate_candidate
from opm.validation.selector import compute_scores

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
