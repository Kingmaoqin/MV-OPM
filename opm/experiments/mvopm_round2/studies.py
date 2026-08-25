"""Round-2 study units. Round-1 study code and evidence remain untouched."""
from __future__ import annotations

import numpy as np

from ...data.views import observed
from ...dgp import finite_proxy_exact as fpe
from ...dgp import bridge_complexity
from ...dgp.synthetic_main import corrupt_proxies
from ...estimator.learner import OPM, OPMConfig
from ...estimator.pseudo_outcome import compute_phi
from ...eval import oracle
from ...validation.moment_tests import rff_features
from ...validation.moments import build_bank
from ...validation.selector_v2 import select_mses
from ..mvopm.scenarios import generate as generate_legacy
from .core import Round2Config, evaluate_candidates_round2, run_round2_selection

E_TANH2 = 0.3942944903978411
ATE_TRUE = fpe.ATE


def _nested_rhc_ate_payload(nested: dict, K: int = 2) -> dict:
    """Expose nested adaptive estimates without upgrading naive Wald coverage claims."""
    summaries = (nested.get("ate_by_selector") or {}).get("MSES")
    if nested.get("status") != "ok" or not summaries:
        return {
            "ate": None,
            "ate_lo": None,
            "ate_hi": None,
            "ate_se": None,
            "ate_source": "nested_outer_oof_adaptive_mses",
            "ate_interval_scope": nested.get("ate_interval_scope"),
        }
    ordered = [summaries[k] for k in range(1, K)]
    return {
        "ate": [row["ate"] for row in ordered],
        "ate_lo": [row["ci_low"] for row in ordered],
        "ate_hi": [row["ci_high"] for row in ordered],
        "ate_se": [row["se"] for row in ordered],
        "ate_source": "nested_outer_oof_adaptive_mses",
        "ate_interval_scope": nested.get("ate_interval_scope"),
    }


def aligned_mechanism_row(
    seed: int,
    r_h: float,
    r_q: float,
    *,
    n: int = 4000,
    c_U: float = 1.0,
    n_rff: int = 200,
    bank_seed: int = 8020240,
) -> dict:
    """Study A2: align h/q corruption on treated arm and keep the reference exact."""
    if r_h not in {0.0, 0.05, 0.10, 0.20, 0.40} or r_q not in {0.0, 0.05, 0.10, 0.20, 0.40}:
        raise ValueError("A2 corruption is outside the frozen grid")
    ds = fpe.generate(n, seed, c_U=c_U)
    view = observed(ds)
    h_true = np.asarray(ds.meta["h_true_val"], dtype=float)
    q_true = np.asarray(ds.meta["q_true_val"], dtype=float)
    h_hat, q_hat = h_true.copy(), q_true.copy()
    g = np.tanh(ds.X[:, 0])
    h_hat[:, 1] += r_h * g
    q_multiplier = 1.0 + r_q * g
    q_hat[:, 1] *= q_multiplier
    if np.min(q_hat) <= 0:
        raise FloatingPointError("aligned q perturbation violated positivity")

    inst_h = np.concatenate([view.V, view.X], axis=1)
    inst_q = np.concatenate([view.W, view.X], axis=1)
    bank = build_bank(inst_h, inst_q, n_rff=n_rff, seed=bank_seed)
    gh, gq = rff_features(bank, inst_h, inst_q)
    mask = (view.T == 1).astype(float)
    R = mask * (view.Y - h_hat[:, 1])
    S = mask * q_hat[:, 1] - 1.0
    Rn = R / (np.std(R) + 1e-8)
    Sn = S / (np.std(S) + 1e-8)
    D_h = float(np.sqrt(np.mean(np.mean(Rn[:, None] * gh, axis=0) ** 2)))
    D_q = float(np.sqrt(np.mean(np.mean(Sn[:, None] * gq, axis=0) ** 2)))

    phi_exact = compute_phi(h_true, q_true, view)
    phi = compute_phi(h_hat, q_hat, view)
    exact_sample_ate = float(np.mean(phi_exact[:, 1] - phi_exact[:, 0]))
    ate_hat = float(np.mean(phi[:, 1] - phi[:, 0]))
    predicted_bias = float(-r_h * r_q * E_TANH2)
    h_error = float(np.sqrt(np.mean((h_hat[:, 1] - h_true[:, 1]) ** 2)))
    q_error = float(np.sqrt(np.mean((q_hat[:, 1] - q_true[:, 1]) ** 2)))
    return {
        "study": "A2",
        "seed": int(seed),
        "n": int(n),
        "c_U": float(c_U),
        "r_h": float(r_h),
        "r_q": float(r_q),
        "E_tanh2_quadrature": E_TANH2,
        "predicted_bias": predicted_bias,
        "theoretical_product": float(r_h * r_q * E_TANH2),
        "sample_g2": float(np.mean(g * g)),
        "true_h_error": h_error,
        "true_q_error": q_error,
        "true_error_product": h_error * q_error,
        "D_h": D_h,
        "D_q": D_q,
        "moment_product": D_h * D_q,
        "ate_hat": ate_hat,
        "exact_sample_ate_hat": exact_sample_ate,
        "incremental_ate_bias": ate_hat - exact_sample_ate,
        "ate_bias": ate_hat - ATE_TRUE,
        "abs_ate_bias": abs(ate_hat - ATE_TRUE),
        "min_q_multiplier": float(np.min(q_multiplier)),
        "min_q": float(np.min(q_hat)),
        "reference_h_unchanged": bool(np.array_equal(h_hat[:, 0], h_true[:, 0])),
        "reference_q_unchanged": bool(np.array_equal(q_hat[:, 0], q_true[:, 0])),
    }


def _dataset(scenario: str, n: int, seed: int, c_U: float, regime: dict | None = None):
    if scenario == "bridge_complexity":
        if regime is None:
            raise ValueError("bridge_complexity requires a regime")
        return bridge_complexity.generate(
            n,
            seed,
            family=regime["family"],
            proxy_noise=regime["proxy_noise"],
            p=regime.get("p", 10),
            c_U=c_U,
        )
    return generate_legacy(scenario, n, seed, c_U=c_U)


def selection_study_row(
    study: str,
    scenario: str,
    n: int,
    seed: int,
    cfg: Round2Config,
    *,
    c_U: float = 1.0,
    proxy_mod: dict | None = None,
    regime: dict | None = None,
    evaluate_ignorability: bool = False,
    library_scenario_label: str | None = None,
    library_truth: str | None = None,
) -> dict:
    """One confirmatory simulation task, retaining a complete candidate-level table."""
    ds = _dataset(scenario, n, seed, c_U, regime)
    if proxy_mod:
        W, V = corrupt_proxies(ds.W, ds.V, proxy_mod["kind"], float(proxy_mod["level"]), seed)
        ds.W, ds.V = W, V
        ds.meta.update({"proxy_mod_kind": proxy_mod["kind"], "proxy_mod_level": proxy_mod["level"]})
    result = run_round2_selection(ds, cfg)
    out = {
        "study": study,
        "scenario": scenario,
        "seed": int(seed),
        "n": int(n),
        "K": int(ds.K),
        "c_U": float(c_U),
        "proxy_mod_kind": (proxy_mod or {}).get("kind"),
        "proxy_mod_level": (proxy_mod or {}).get("level"),
        "regime": regime,
        "library_scenario_label": library_scenario_label or library_truth,
        "legacy_library_truth_label": library_truth,
        "selectors": result["selectors"],
        "_candidates": result["candidates"],
        "_nested": result.get("_nested"),
        "arm_counts": np.bincount(ds.T, minlength=ds.K).astype(int).tolist(),
    }
    true_propensity = (ds.meta or {}).get("true_propensity")
    if true_propensity is not None:
        min_probability = np.min(np.asarray(true_propensity, dtype=float), axis=1)
        out.update({
            "true_overlap_min_probability_mean": float(np.mean(min_probability)),
            "true_overlap_min_probability_q05": float(np.quantile(min_probability, 0.05)),
        })
    if evaluate_ignorability:
        ign = OPM(OPMConfig(
            K=ds.K,
            mode="dr_fallback",
            n_folds=cfg.n_folds,
            seed=cfg.seed,
            device=cfg.device,
            head_kwargs=cfg.head_kwargs,
        )).fit(ds)
        out["ignorability"] = {
            **oracle.candidate_causal_error(ign.predict_cate(ds.X), ign.ate_vector(), ds),
            "ate": ign.ate_vector().tolist(),
        }
    return out


def rhc_stability_row(seed: int, cfg: Round2Config) -> dict:
    """One RHC repeated-split task; no oracle quantities are computed or implied."""
    from ...data.rhc import load_rhc
    from ...estimator.nested_mses import NestedMSESConfig, nested_mses

    ds = load_rhc(require_verified_proxy_mapping=True)
    cfg.K = 2
    cfg.candidate_names = ["kernel_kernel", "kernel_sieve1", "sieve1_kernel", "sieve1_sieve1"]
    evaluated = evaluate_candidates_round2(ds, cfg)
    rows = evaluated["rows"]
    tests = {n: {"p_h": rows[n]["p_h"], "p_q": rows[n]["p_q"]} for n in rows}
    uncertainty = {n: {"variance_mean": rows[n]["po_var_mean"]} for n in rows}
    decision = select_mses(tests, uncertainty, alpha=cfg.alpha)
    nested = nested_mses(observed(ds), NestedMSESConfig(
        K=2,
        outer_folds=cfg.n_folds,
        inner_folds=cfg.inner_folds,
        seed=cfg.seed,
        split_seed=cfg.split_seed,
        bootstrap_seed=cfg.bootstrap_seed + 300000000,
        bank_seed=cfg.bank_seed + 300000000,
        n_boot=cfg.n_boot,
        n_rff=cfg.n_rff,
        alpha=cfg.alpha,
        device=cfg.device,
        candidate_names=cfg.candidate_names,
        bridge_kwargs=cfg.bridge_kwargs,
        head_kwargs=cfg.head_kwargs,
        compute_cate_oof=False,
    ))
    selected = decision["selected"]
    selected_metrics = None if selected.startswith("ABSTAIN") else rows[selected]
    nested_ate = _nested_rhc_ate_payload(nested, K=2)
    fold_choices = nested.get("selected") or []
    fold_abstention_rate = float(np.mean([
        str(choice).startswith("ABSTAIN") for choice in fold_choices
    ])) if fold_choices else float("nan")
    return {
        "study": "J2",
        "scenario": "rhc",
        "seed": int(seed),
        "n": ds.n,
        "selected": selected,
        "survivors": decision["survivors"],
        "survivor_count": decision["survivor_count"],
        "nested_status": nested["status"],
        "nested_selected_by_fold": fold_choices,
        "nested_fold_abstention_rate": fold_abstention_rate,
        **nested_ate,
        "deployment_ate": None if selected_metrics is None else selected_metrics["ate"],
        "deployment_ate_lo_naive": None if selected_metrics is None else selected_metrics["ate_lo"],
        "deployment_ate_hi_naive": None if selected_metrics is None else selected_metrics["ate_hi"],
        "deployment_interval_scope": "naive_candidate_oof_after_adaptive_selection",
        "deployment_abstained": decision["abstained"],
        "deployment_ess": None if selected_metrics is None else selected_metrics["ess"],
        "deployment_max_q": None if selected_metrics is None else selected_metrics["max_q"],
        "deployment_q_balance": None if selected_metrics is None else selected_metrics["q_balance"],
        "_candidates": rows,
    }
