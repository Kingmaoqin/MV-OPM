"""Oracle-isolated evaluation layer for Round-2 candidates and selectors."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import kendalltau, spearmanr

from ...data.views import observed
from ...diagnostics.diagnostics import q_sanity
from ...estimator.nested_mses import NestedMSESConfig, nested_mses
from ...estimator.tau_head import fit_cate_and_ate
from ...eval import oracle
from ...validation.oof import collect_candidate_oof
from ...validation.selector import SCORE_NAMES, compute_scores, select
from ...validation.selector_v2 import ABSTAIN, screen_candidate, select_mses

EPS = 1e-8
EXPLORATORY_LABEL = "EXPLORATORY_POSTHOC_FROM_ROUND1"


@dataclass
class Round2Config:
    K: int
    n_folds: int = 4
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
    run_nested: bool = True


def evaluate_candidates_round2(ds, cfg: Round2Config) -> dict:
    """Compute common-bank tests, OOF uncertainty, diagnostics, and oracle errors."""
    view = observed(ds)
    if view.K != cfg.K:
        raise ValueError("Round2Config.K does not match dataset")
    oof = collect_candidate_oof(
        view,
        n_folds=cfg.n_folds,
        seed=cfg.seed if cfg.split_seed is None else cfg.split_seed,
        device=cfg.device,
        candidate_names=cfg.candidate_names,
        bridge_kwargs=cfg.bridge_kwargs,
        n_rff=cfg.n_rff,
        bank_seed=cfg.bank_seed,
        n_boot=cfg.n_boot,
        bootstrap_seed=cfg.bootstrap_seed,
    )
    rows = {}
    has_truth = oracle.has_truth(ds)
    for nm in sorted(oof["phi"]):
        phi = oof["phi"][nm]
        tau_head, _, ate = fit_cate_and_ate(ds.X, phi, cfg.K, cfg.device, cfg.seed, cfg.head_kwargs)
        if has_truth:
            err = oracle.candidate_causal_error(
                tau_head.predict(ds.X), [ate[k]["ate"] for k in range(1, cfg.K)], ds
            )
        else:
            err = {"pehe": float("nan"), "ate_err": float("nan")}
        qd = q_sanity(oof["q"][nm], ds.T, cfg.K)
        tst = oof["tests"][nm]
        scr = screen_candidate(tst["p_h"], tst["p_q"], alpha=cfg.alpha)
        unc = oof["uncertainty"][nm]
        rows[nm] = {
            **oof["discrepancy"][nm],
            **err,
            "p_h": tst["p_h"].tolist(),
            "p_q": tst["p_q"].tolist(),
            "p_DR": tst["p_DR"].tolist(),
            "test_h": tst["h"],
            "test_q": tst["q"],
            "screen_pass": bool(scr.passes),
            "screen_threshold": scr.threshold,
            "failed_arms": list(scr.failed_arms),
            "po_var": unc["variance_mean"],
            "po_var_mean": unc["variance_mean"],
            "po_var_max": unc["variance_max"],
            "po_var_by_contrast": unc["contrast_variance"].tolist(),
            "ate_se_by_contrast": unc["ate_se"].tolist(),
            "ate_ci_width_by_contrast": unc["ate_ci_width"].tolist(),
            "ate": [ate[k]["ate"] for k in range(1, cfg.K)],
            "ate_lo": [ate[k]["ci_low"] for k in range(1, cfg.K)],
            "ate_hi": [ate[k]["ci_high"] for k in range(1, cfg.K)],
            "q_balance": float(np.mean([abs(qd[k]["En_1Tk_qk"] - 1.0) for k in range(cfg.K)])),
            "ess": float(np.mean([qd[k]["ess"] for k in range(cfg.K)])),
            "ess_by_arm": [qd[k]["ess"] for k in range(cfg.K)],
            "max_q": float(max(qd[k]["max_q"] for k in range(cfg.K))),
        }
    return {"rows": rows, "oof": oof}


def _exploratory_scores(rows: dict) -> dict:
    names = sorted(rows)
    Dh = np.array([rows[n]["D_h_rff_l2"] for n in names])
    Dq = np.array([rows[n]["D_q_rff_l2"] for n in names])
    var = np.array([rows[n]["po_var_mean"] for n in names])
    logprod = np.log(Dh + EPS) + np.log(Dq + EPS)
    z = lambda x: (np.log(x + EPS) - np.mean(np.log(x + EPS))) / (np.std(np.log(x + EPS)) + EPS)
    return {
        "biasvar_mult": dict(zip(names, logprod + np.log(var + EPS))),
        "biasvar_add": dict(zip(names, z(Dh * Dq) + z(var))),
    }


def selector_metrics(rows: dict, cfg: Round2Config, *, nested_result: dict | None = None, ds=None) -> dict:
    """Apply all fixed comparators; ATE error is the primary target."""
    names = sorted(rows)
    err = np.array([rows[n]["ate_err"] for n in names], dtype=float)
    oracle_best = names[int(np.nanargmin(err))]
    best_err = float(np.nanmin(err))
    old_scores = compute_scores(rows, "rff_l2")
    tests = {n: {"p_h": rows[n]["p_h"], "p_q": rows[n]["p_q"]} for n in names}
    uncertainty = {n: {"variance_mean": rows[n]["po_var_mean"]} for n in names}
    mses = select_mses(tests, uncertainty, alpha=cfg.alpha)
    rng = np.random.default_rng(cfg.seed + 999)
    selectors = {s: select(old_scores[s]) for s in SCORE_NAMES}
    selectors.update({
        "variance_only": min(names, key=lambda n: (rows[n]["po_var_mean"], n)),
        "fixed_sieve1": "sieve1_sieve1" if "sieve1_sieve1" in names else names[0],
        "fixed_kernel": "kernel_kernel" if "kernel_kernel" in names else names[0],
        "random": names[int(rng.integers(len(names)))],
        "oracle": oracle_best,
    })
    exploratory = _exploratory_scores(rows)
    selectors.update({s: min(v, key=v.get) for s, v in exploratory.items()})

    out = {
        "primary_target": "ate_err",
        "oracle_best": oracle_best,
        "oracle_best_error": best_err,
        "global_oof_mses_selected_deployment": mses["selected"],
        "global_oof_mses_survivors": mses["survivors"],
        "global_oof_mses_survivor_count": mses["survivor_count"],
        "global_oof_mses_oracle_survives": bool(oracle_best in mses["survivors"]),
        "global_oof_mses_abstained": mses["abstained"],
        "exploratory_score_label": EXPLORATORY_LABEL,
    }
    for sname, chosen in selectors.items():
        selected_err = float(rows[chosen]["ate_err"])
        prefix = "" if sname == "oracle" else "global_oof_"
        out[f"{prefix}selected[{sname}]"] = chosen
        out[f"{prefix}error[{sname}]"] = selected_err
        out[f"{prefix}regret[{sname}]"] = selected_err - best_err
        out[f"{prefix}oracle_ratio[{sname}]"] = selected_err / (best_err + 1e-12)
        out[f"{prefix}catastrophic[{sname}]"] = int(selected_err > 1.5 * best_err)
        out[f"{prefix}top1[{sname}]"] = int(chosen == oracle_best)

    # The global-OOF decision describes what would be deployed after using all observations for
    # selection. It is not the primary confirmatory performance estimate.
    if mses["selected"] == ABSTAIN:
        for key in ("error", "regret", "oracle_ratio", "catastrophic", "top1", "top2"):
            out[f"global_oof_{key}[MSES]"] = float("nan")
    else:
        chosen = mses["selected"]
        selected_err = float(rows[chosen]["ate_err"])
        var_order = sorted(mses["survivors"], key=lambda n: (rows[n]["po_var_mean"], n))
        out.update({
            "global_oof_selected[MSES]": chosen,
            "global_oof_error[MSES]": selected_err,
            "global_oof_regret[MSES]": selected_err - best_err,
            "global_oof_oracle_ratio[MSES]": selected_err / (best_err + 1e-12),
            "global_oof_catastrophic[MSES]": int(selected_err > 1.5 * best_err),
            "global_oof_top1[MSES]": int(chosen == oracle_best),
            "global_oof_top2[MSES]": int(oracle_best in var_order[:2]),
        })

    scalar_scores = {**old_scores, "variance_only": {n: rows[n]["po_var_mean"] for n in names}, **exploratory}
    for sname, score in scalar_scores.items():
        values = np.array([score[n] for n in names], dtype=float)
        out[f"spearman[{sname}]"] = float(spearmanr(values, err).correlation)
        out[f"kendall[{sname}]"] = float(kendalltau(values, err).correlation)
        out[f"top2[{sname}]"] = int(oracle_best in set(np.array(names)[np.argsort(values)[:2]]))

    if nested_result is not None:
        selected_by_fold = nested_result["selected"]
        survivor_counts = [f["survivor_count"] for f in nested_result["folds"]]
        oracle_survival = [oracle_best in f["survivors"] for f in nested_result["folds"]]
        top1 = [chosen == oracle_best for chosen in selected_by_fold]
        top2 = []
        for fold in nested_result["folds"]:
            order = sorted(fold["survivors"], key=lambda n: (fold["uncertainty"][n]["variance_mean"], n))
            top2.append(oracle_best in order[:2])
        out.update({
            "status[MSES]": nested_result["status"],
            "selected[MSES]": selected_by_fold,
            "survivor_count_by_fold[MSES]": survivor_counts,
            "survivor_count[MSES]": float(np.mean(survivor_counts)),
            "oracle_survival_rate[MSES]": float(np.mean(oracle_survival)),
            "top1[MSES]": float(np.mean(top1)),
            "top2[MSES]": float(np.mean(top2)),
        })
        if not nested_result["abstained"]:
            if ds is None:
                raise ValueError("ds is required to evaluate a nested simulation result")
            nested_ate_error = oracle.ate_error(nested_result["ate_vector"], ds)
            nested_pehe = oracle.pehe(nested_result["predict_cate"](ds.X), ds)
            out.update({
                "error[MSES]": nested_ate_error,
                "pehe[MSES]": nested_pehe,
                "regret[MSES]": nested_ate_error - best_err,
                "oracle_ratio[MSES]": nested_ate_error / (best_err + 1e-12),
                "catastrophic[MSES]": int(nested_ate_error > 1.5 * best_err),
            })
        else:
            for key in ("error", "pehe", "regret", "oracle_ratio", "catastrophic"):
                out[f"{key}[MSES]"] = float("nan")
        # Every adaptive comparator is selected on the identical inner-OOF population, refit on
        # the identical outer train, and evaluated on the identical outer test as MSES.
        for sname, ate_vector in nested_result["ate_vector_by_selector"].items():
            if sname == "MSES":
                continue
            choices = [f["selected_by_selector"][sname] for f in nested_result["folds"]]
            selected_err = oracle.ate_error(ate_vector, ds)
            out.update({
                f"status[{sname}]": "ok",
                f"selected[{sname}]": choices,
                f"error[{sname}]": selected_err,
                f"regret[{sname}]": selected_err - best_err,
                f"oracle_ratio[{sname}]": selected_err / (best_err + 1e-12),
                f"catastrophic[{sname}]": int(selected_err > 1.5 * best_err),
                f"top1[{sname}]": float(np.mean([c == oracle_best for c in choices])),
            })
            fold_top2 = []
            for fold in nested_result["folds"]:
                if sname in SCORE_NAMES:
                    score = compute_scores(fold["discrepancy"], "rff_l2")[sname]
                    order = sorted(score, key=score.get)
                    fold_top2.append(oracle_best in order[:2])
                elif sname == "variance_only":
                    order = sorted(fold["uncertainty"], key=lambda n: (fold["uncertainty"][n]["variance_mean"], n))
                    fold_top2.append(oracle_best in order[:2])
                elif sname in ("biasvar_mult", "biasvar_add"):
                    inner_rows = {
                        n: {**fold["discrepancy"][n], "po_var_mean": fold["uncertainty"][n]["variance_mean"]}
                        for n in fold["discrepancy"]
                    }
                    score = _exploratory_scores(inner_rows)[sname]
                    order = sorted(score, key=score.get)
                    fold_top2.append(oracle_best in order[:2])
                elif sname in ("fixed_kernel", "fixed_sieve1"):
                    fold_top2.append(fold["selected_by_selector"][sname] == oracle_best)
            out[f"top2[{sname}]"] = float(np.mean(fold_top2)) if fold_top2 else float("nan")
    for nm in names:
        rows[nm].update({
            "selected_by_product": selectors["product"] == nm,
            "selected_by_variance_only": selectors["variance_only"] == nm,
            "selected_by_MSES": mses["selected"] == nm,
            "oracle_best": oracle_best == nm,
            "fixed_sieve1": nm == "sieve1_sieve1",
            "fixed_kernel": nm == "kernel_kernel",
        })
    return out


def run_round2_selection(ds, cfg: Round2Config) -> dict:
    """End-to-end one-seed evaluation with optional nested MSES outer evaluation."""
    evaluated = evaluate_candidates_round2(ds, cfg)
    nested_result = None
    if cfg.run_nested:
        nested_result = nested_mses(observed(ds), NestedMSESConfig(
            K=cfg.K,
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
        ))
    metrics = selector_metrics(evaluated["rows"], cfg, nested_result=nested_result, ds=ds)
    if nested_result is None:
        nested_raw = None
    else:
        nested_raw = {
            "status": nested_result["status"], "abstained": nested_result["abstained"],
            "selected": nested_result["selected"], "phi": nested_result["phi"],
            "phi_by_selector": nested_result["phi_by_selector"],
            "folds": nested_result["folds"], "ate": nested_result.get("ate"),
            "ate_vector": nested_result.get("ate_vector"),
            "ate_by_selector": nested_result.get("ate_by_selector"),
            "ate_vector_by_selector": nested_result.get("ate_vector_by_selector"),
        }
    return {"candidates": evaluated["rows"], "selectors": metrics, "_nested": nested_raw}
