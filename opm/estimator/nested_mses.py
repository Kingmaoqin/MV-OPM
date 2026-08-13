"""Leakage-resistant nested cross-fitting for the Round-2 MSES estimator."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..candidates.library import CANDIDATE_NAMES, CandidateSet
from ..estimator.crossfit import make_folds
from ..estimator.pseudo_outcome import compute_phi
from ..estimator.tau_head import ate_with_ci, fit_cate_and_ate
from ..validation.oof import fit_mses_on_oof
from ..validation.selector import SCORE_NAMES, compute_scores, select
from ..validation.selector_v2 import ABSTAIN

EPS = 1e-8


def _fold_selectors(fitted: dict, names: list[str], seed: int) -> dict:
    """All adaptive comparators choose from the same inner-OOF validation population."""
    old_scores = compute_scores(fitted["discrepancy"], "rff_l2")
    selected = {s: select(old_scores[s]) for s in SCORE_NAMES}
    selected["variance_only"] = min(
        names, key=lambda n: (fitted["uncertainty"][n]["variance_mean"], n)
    )
    rng = np.random.default_rng(seed)
    selected["random"] = names[int(rng.integers(len(names)))]
    selected["fixed_kernel"] = "kernel_kernel" if "kernel_kernel" in names else names[0]
    selected["fixed_sieve1"] = "sieve1_sieve1" if "sieve1_sieve1" in names else names[0]
    selected["MSES"] = fitted["mses"]["selected"]

    Dh = np.array([fitted["discrepancy"][n]["D_h_rff_l2"] for n in names])
    Dq = np.array([fitted["discrepancy"][n]["D_q_rff_l2"] for n in names])
    var = np.array([fitted["uncertainty"][n]["variance_mean"] for n in names])
    logprod = np.log(Dh + EPS) + np.log(Dq + EPS)
    z = lambda x: (np.log(x + EPS) - np.mean(np.log(x + EPS))) / (np.std(np.log(x + EPS)) + EPS)
    selected["biasvar_mult"] = names[int(np.argmin(logprod + np.log(var + EPS)))]
    selected["biasvar_add"] = names[int(np.argmin(z(Dh * Dq) + z(var)))]
    return selected


@dataclass
class NestedMSESConfig:
    K: int
    outer_folds: int = 4
    inner_folds: int = 3
    seed: int = 0
    split_seed: int | None = None
    bootstrap_seed: int = 7020240
    bank_seed: int = 8020240
    n_boot: int = 999
    n_rff: int = 200
    alpha: float = 0.05
    device: str = "cpu"
    candidate_names: list | None = None
    bridge_kwargs: dict = field(default_factory=dict)
    head_kwargs: dict = field(default_factory=dict)


def nested_mses(view, cfg: NestedMSESConfig) -> dict:
    """Select inside every outer-training split and predict only its untouched outer test."""
    n, K = view.n, view.K
    if K != cfg.K:
        raise ValueError("cfg.K does not match the observed view")
    names = sorted(cfg.candidate_names) if cfg.candidate_names else list(CANDIDATE_NAMES)
    split_seed = cfg.seed if cfg.split_seed is None else cfg.split_seed
    outer = make_folds(n, cfg.outer_folds, split_seed)
    all_idx = np.arange(n)
    selector_names = [*SCORE_NAMES, "variance_only", "random", "fixed_kernel", "fixed_sieve1",
                      "biasvar_mult", "biasvar_add", "MSES"]
    phi_by_selector = {s: np.full((n, K), np.nan) for s in selector_names}
    folds = []
    abstained = False
    for of, outer_test in enumerate(outer):
        outer_train = np.setdiff1d(all_idx, outer_test, assume_unique=False)
        train_view = view.subset(outer_train)
        fitted = fit_mses_on_oof(
            train_view,
            n_folds=cfg.inner_folds,
            seed=cfg.seed + 100000 * of + 101,
            device=cfg.device,
            candidate_names=names,
            bridge_kwargs=cfg.bridge_kwargs,
            n_rff=cfg.n_rff,
            bank_seed=cfg.bank_seed + 1000 * of,
            n_boot=cfg.n_boot,
            bootstrap_seed=cfg.bootstrap_seed + 1000000 * of,
            alpha=cfg.alpha,
        )
        decision = fitted["mses"]
        selected = _fold_selectors(fitted, names, cfg.seed + 900000 + of)
        audit = {
            "fold": of,
            "outer_train": outer_train,
            "outer_test": outer_test,
            "outer_train_test_disjoint": len(np.intersect1d(outer_train, outer_test)) == 0,
            "inner_coverage_complete": bool(np.all(fitted["coverage"] == 1)),
            "inner_folds": [
                {
                    "fold": inner["fold"],
                    "train_local": inner["train"],
                    "valid_local": inner["valid"],
                    "train_global": outer_train[inner["train"]],
                    "valid_global": outer_train[inner["valid"]],
                    "disjoint": inner["disjoint"],
                    "outer_test_absent": len(np.intersect1d(
                        outer_train[np.concatenate([inner["train"], inner["valid"]])], outer_test
                    )) == 0,
                }
                for inner in fitted["fold_audit"]
            ],
            "selected": selected["MSES"],
            "selected_by_selector": selected,
            "survivors": decision["survivors"],
            "survivor_count": decision["survivor_count"],
            "tests": fitted["tests"],
            "uncertainty": fitted["uncertainty"],
            "discrepancy": fitted["discrepancy"],
        }
        folds.append(audit)
        if selected["MSES"] == ABSTAIN:
            abstained = True
        chosen_candidates = sorted({name for name in selected.values() if name != ABSTAIN})
        refit = CandidateSet(
            K,
            # Match the fixed candidate-baseline outer-fold fit seed, so oracle comparisons use
            # the same refit convention and differ only by candidate choice.
            seed=cfg.seed + 1000 * of + 1,
            device=cfg.device,
            bridge_kwargs=cfg.bridge_kwargs,
            candidate_names=chosen_candidates,
        ).fit(train_view)
        test_view = view.subset(outer_test)
        phi_by_candidate = {}
        for name in chosen_candidates:
            cand = refit.get(name)
            h = cand.predict_h(test_view.W, test_view.X)
            q = cand.predict_q(test_view.V, test_view.X)
            phi_by_candidate[name] = compute_phi(h, q, test_view)
        for selector_name, candidate_name in selected.items():
            if candidate_name != ABSTAIN:
                phi_by_selector[selector_name][outer_test] = phi_by_candidate[candidate_name]

    for selector_name, values in phi_by_selector.items():
        if selector_name == "MSES" and abstained:
            continue
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"outer OOF coverage invariant failed for {selector_name}")
    ate_by_selector = {}
    ate_vector_by_selector = {}
    for selector_name, values in phi_by_selector.items():
        if not np.all(np.isfinite(values)):
            ate_by_selector[selector_name] = None
            ate_vector_by_selector[selector_name] = None
            continue
        ate = {k: ate_with_ci(values[:, k] - values[:, 0]) for k in range(1, K)}
        ate_by_selector[selector_name] = ate
        ate_vector_by_selector[selector_name] = np.array([ate[k]["ate"] for k in range(1, K)])

    if abstained:
        return {
            "status": ABSTAIN,
            "abstained": True,
            "phi": phi_by_selector["MSES"],
            "phi_by_selector": phi_by_selector,
            "folds": folds,
            "selected": [f["selected"] for f in folds],
            "ate": None,
            "ate_vector": None,
            "ate_by_selector": ate_by_selector,
            "ate_vector_by_selector": ate_vector_by_selector,
            "predict_cate": None,
        }
    phi = phi_by_selector["MSES"]
    tau_head, g0_head, ate = fit_cate_and_ate(view.X, phi, K, cfg.device, cfg.seed, cfg.head_kwargs)
    return {
        "status": "ok",
        "abstained": False,
        "phi": phi,
        "phi_by_selector": phi_by_selector,
        "folds": folds,
        "selected": [f["selected"] for f in folds],
        "tau_head": tau_head,
        "g0_head": g0_head,
        "ate": ate,
        "ate_vector": np.array([ate[k]["ate"] for k in range(1, K)]),
        "ate_by_selector": ate_by_selector,
        "ate_vector_by_selector": ate_vector_by_selector,
        "predict_cate": lambda X: tau_head.predict(X).reshape(X.shape[0], K - 1),
    }
