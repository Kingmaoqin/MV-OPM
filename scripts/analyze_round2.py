"""Frozen, deterministic Round-2 tables, diagnostics, and scientific figures.

This script consumes only the reconstructed CSV files written by
``scripts/cluster/aggregate_round2.py``.  It does not refit models, change selector decisions, or
read simulation truth beyond the truth/error columns already retained in those CSVs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUND2 = os.path.join(ROOT, "results", "mvopm_round2")
SELECTORS = [
    "MSES", "product", "variance_only", "fixed_sieve1", "fixed_kernel",
    "biasvar_mult", "biasvar_add", "oracle",
]
COLORS = {
    "MSES": "#1565c0", "product": "#ef6c00", "variance_only": "#00897b",
    "fixed_sieve1": "#6a1b9a", "fixed_kernel": "#c62828",
    "biasvar_mult": "#7cb342", "biasvar_add": "#8d6e63", "oracle": "#212121",
}
SHORT = {
    "kernel_kernel": "K/K", "kernel_sieve1": "K/S1", "sieve1_kernel": "S1/K",
    "sieve1_sieve1": "S1/S1", "sieve2_sieve2": "S2/S2",
    "sieve3_sieve3": "S3/S3", "ABSTAIN_LIBRARY_INADEQUATE": "ABSTAIN",
}


def _selector_col(metric: str, selector: str) -> str:
    return f"selector::{metric}[{selector}]"


def _selector_series(frame: pd.DataFrame, metric: str, selector: str) -> pd.Series:
    """Prefer confirmatory nested metrics; permit global-OOF fallback for dev-only smoke."""
    primary = _selector_col(metric, selector)
    fallback = _selector_col(f"global_oof_{metric}", selector)
    if primary in frame:
        return frame[primary]
    if fallback in frame:
        return frame[fallback]
    return pd.Series(index=frame.index, dtype=float)


def _json(value, default=None):
    if isinstance(value, (dict, list)):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _first_number(value):
    parsed = _json(value, value)
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else np.nan
    try:
        return float(parsed)
    except (TypeError, ValueError):
        return np.nan


def _finite(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values[np.isfinite(values)]


def _safe_stats(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": len(x), "pearson": np.nan, "spearman": np.nan,
                "slope": np.nan, "intercept": np.nan, "r_squared": np.nan}
    fit = stats.linregress(x, y)
    return {
        "n": int(len(x)), "pearson": float(stats.pearsonr(x, y).statistic),
        "spearman": float(stats.spearmanr(x, y).statistic),
        "slope": float(fit.slope), "intercept": float(fit.intercept),
        "r_squared": float(fit.rvalue ** 2),
    }


def _write(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def mechanism_tables(tasks: pd.DataFrame, tables: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    a2 = tasks[tasks.study == "A2"].copy()
    numeric = [
        "r_h", "r_q", "predicted_bias", "theoretical_product", "true_h_error",
        "true_q_error", "true_error_product", "D_h", "D_q", "moment_product",
        "incremental_ate_bias", "ate_bias", "abs_ate_bias", "min_q_multiplier", "min_q",
    ]
    for col in numeric:
        a2[col] = pd.to_numeric(a2[col], errors="coerce")
    metrics = [
        "predicted_bias", "theoretical_product", "true_h_error", "true_q_error",
        "true_error_product", "D_h", "D_q", "moment_product", "incremental_ate_bias",
        "ate_bias", "abs_ate_bias", "min_q_multiplier", "min_q",
    ]
    rows = []
    for (rh, rq), group in a2.groupby(["r_h", "r_q"], sort=True):
        row = {"r_h": rh, "r_q": rq, "n_rep": len(group)}
        for metric in metrics:
            values = _finite(group[metric])
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std(ddof=1)
            row[f"{metric}_se"] = values.std(ddof=1) / math.sqrt(len(values))
        row["incremental_bias_minus_prediction"] = (
            row["incremental_ate_bias_mean"] - row["predicted_bias_mean"]
        )
        rows.append(row)
    cells = pd.DataFrame(rows)
    _write(cells, os.path.join(tables, "mechanism_cells.csv"))

    relations = [
        ("D_h_vs_true_h_error", "true_h_error", "D_h"),
        ("D_q_vs_true_q_error", "true_q_error", "D_q"),
        ("moment_product_vs_true_error_product", "true_error_product", "moment_product"),
        ("predicted_vs_incremental_ate_bias", "predicted_bias", "incremental_ate_bias"),
        ("theoretical_product_vs_negative_incremental_bias", "theoretical_product", "incremental_ate_bias"),
    ]
    diag = []
    for label, x, y in relations:
        row = {"relation": label, "x": x, "y": y, **_safe_stats(a2[x], a2[y])}
        if label.startswith("theoretical_product"):
            # The algebra predicts a negative slope for the unsigned product.
            pass
        diag.append(row)
    diagnostics = pd.DataFrame(diag)
    _write(diagnostics, os.path.join(tables, "mechanism_diagnostics.csv"))
    return cells, diagnostics


def selector_tables(tasks: pd.DataFrame, tables: str) -> pd.DataFrame:
    scopes = {
        "flagship_B2_E2": tasks.study.isin(["B2", "C2", "D2", "E2"]),
        "core_plus_switching": tasks.study.isin(["B2", "C2", "D2", "E2", "R"]),
        "all_simulation_selection": tasks.study.isin(["B2", "C2", "D2", "E2", "R", "F2", "G2", "H2", "I2"]),
    }
    for study in ["B2", "C2", "D2", "E2", "R", "F2", "G2", "H2", "I2"]:
        scopes[study] = tasks.study.eq(study)
    rows = []
    for scope, mask in scopes.items():
        group = tasks[mask]
        for selector in SELECTORS:
            ratio_source = _selector_series(group, "oracle_ratio", selector)
            if ratio_source.empty:
                continue
            ratio = _finite(ratio_source)
            regret = _finite(_selector_series(group, "regret", selector))
            catastrophic = _finite(_selector_series(group, "catastrophic", selector))
            top1 = _finite(_selector_series(group, "top1", selector))
            top2 = _finite(_selector_series(group, "top2", selector))
            error = _finite(_selector_series(group, "error", selector))
            row = {
                "scope": scope, "selector": selector, "n_tasks": len(group),
                "n_evaluable": len(ratio), "abstention_or_missing_rate": 1 - len(ratio) / max(len(group), 1),
                "oracle_ratio_mean": ratio.mean(), "oracle_ratio_median": ratio.median(),
                "oracle_ratio_q25": ratio.quantile(.25), "oracle_ratio_q75": ratio.quantile(.75),
                "error_mean": error.mean(), "error_median": error.median(),
                "regret_mean": regret.mean(), "regret_median": regret.median(),
                "catastrophic_rate": catastrophic.mean(), "top1_rate": top1.mean(),
                "top2_rate": top2.mean(),
            }
            if selector == "MSES":
                survivor = _finite(_selector_series(group, "mses_survivor_count", selector))
                if survivor.empty:
                    survivor = _finite(group.get(_selector_col("survivor_count", selector), pd.Series(dtype=float)))
                oracle_survives = _finite(group.get(_selector_col("oracle_survival_rate", selector), pd.Series(dtype=float)))
                if oracle_survives.empty:
                    oracle_survives = _finite(group.get("selector::global_oof_mses_oracle_survives", pd.Series(dtype=float)))
                row.update({"survivor_count_mean": survivor.mean(),
                            "oracle_survival_rate": oracle_survives.mean()})
            rows.append(row)
    out = pd.DataFrame(rows)
    _write(out, os.path.join(tables, "selector_performance.csv"))
    return out


def paired_selector_tables(tasks: pd.DataFrame, tables: str) -> pd.DataFrame:
    """Paired ATE-error contrasts; positive values favor MSES.

    Scientific seeds, rather than task-regime rows, are the independent resampling unit.  This
    matters for study R, where each of 30 scientific seeds is reused across all 20 registered
    regimes.  The bootstrap is stratified by study and resamples whole scientific-seed blocks,
    preserving every row in a sampled block and each study's registered allocation.  Row-level
    medians and win fractions remain descriptive summaries only.
    """
    core = tasks[tasks.study.isin(["B2", "C2", "D2", "E2", "R"])].copy()
    scopes = {
        "core_plus_switching": core,
        "flagship_B2_E2": core[core.study.isin(["B2", "C2", "D2", "E2"])],
        "R_switching": core[core.study.eq("R")],
    }
    if "selector::oracle_best" in core:
        scopes["oracle_not_sieve1"] = core[core["selector::oracle_best"] != "sieve1_sieve1"]
    rng = np.random.default_rng(20260812)
    rows = []
    for scope, group in scopes.items():
        mes = pd.to_numeric(_selector_series(group, "error", "MSES"), errors="coerce")
        for comparator in ["product", "variance_only", "fixed_sieve1", "fixed_kernel",
                           "biasvar_mult", "biasvar_add"]:
            other = pd.to_numeric(_selector_series(group, "error", comparator), errors="coerce")
            keep = np.isfinite(mes) & np.isfinite(other)
            paired = group.loc[keep, ["study", "scientific_seed"]].copy()
            paired["delta"] = (other[keep] - mes[keep]).to_numpy(float)
            delta = paired["delta"].to_numpy(float)
            if not len(delta):
                continue

            blocks_by_study = []
            for _, study_group in paired.groupby("study", sort=True):
                blocks = [
                    seed_group["delta"].to_numpy(float)
                    for _, seed_group in study_group.groupby("scientific_seed", sort=True)
                ]
                blocks_by_study.append(blocks)
            n_blocks = sum(len(blocks) for blocks in blocks_by_study)
            bootstrap_means = np.empty(5000, dtype=float)
            for iteration in range(len(bootstrap_means)):
                total = 0.0
                count = 0
                for blocks in blocks_by_study:
                    sampled = rng.integers(0, len(blocks), size=len(blocks))
                    for index in sampled:
                        block = blocks[index]
                        total += float(block.sum())
                        count += len(block)
                bootstrap_means[iteration] = total / count
            ci_low, ci_high = np.quantile(bootstrap_means, [.025, .975])
            rows.append({
                "scope": scope, "comparator": comparator,
                "n_paired_task_rows": len(delta),
                "n_independent_scientific_seed_blocks": n_blocks,
                "mean_error_reduction_MSES": delta.mean(),
                "descriptive_row_median_error_reduction_MSES": np.median(delta),
                "stratified_seed_block_bootstrap_95ci_low": ci_low,
                "stratified_seed_block_bootstrap_95ci_high": ci_high,
                "descriptive_fraction_rows_MSES_lower_error": np.mean(delta > 0),
            })
    out = pd.DataFrame(rows)
    _write(out, os.path.join(tables, "paired_selector_comparisons.csv"))
    return out


def screening_tables(tasks: pd.DataFrame, tables: str) -> pd.DataFrame:
    rows = []
    simulation = tasks[tasks.study.isin(["B2", "C2", "D2", "E2", "R", "F2", "G2", "H2", "I2"])]
    for (study, config_id), group in simulation.groupby(["study", "config_id"], sort=True):
        status = group.get(_selector_col("status", "MSES"),
                           pd.Series(index=group.index, dtype=object)).astype(str)
        survivor = _finite(group.get(_selector_col("survivor_count", "MSES"), pd.Series(dtype=float)))
        oracle_survives = _finite(group.get(_selector_col("oracle_survival_rate", "MSES"),
                                            pd.Series(dtype=float)))
        rows.append({
            "study": study, "config_id": config_id, "n_tasks": len(group),
            "abstention_rate": status.eq("ABSTAIN_LIBRARY_INADEQUATE").mean(),
            "survivor_count_mean": survivor.mean(), "survivor_count_median": survivor.median(),
            "oracle_candidate_survival_rate": oracle_survives.mean(),
        })
    out = pd.DataFrame(rows)
    _write(out, os.path.join(tables, "screening_performance.csv"))
    return out


def candidate_tables(candidates: pd.DataFrame, tables: str) -> pd.DataFrame:
    numeric = [
        "D_h_rff_l2", "D_q_rff_l2", "D_h_mmr", "D_q_mmr", "min_p_h", "min_p_q",
        "min_p_DR", "screen_pass", "po_var_mean", "mean_ate_ci_width_by_contrast", "pehe",
        "ate_err", "ess", "max_q", "q_balance", "selected_by_product",
        "selected_by_variance_only", "selected_by_MSES", "oracle_best", "fixed_sieve1",
        "fixed_kernel", "nested_screen_survival_fraction", "nested_selected_fraction[MSES]",
        "nested_selected_fraction[product]", "nested_selected_fraction[variance_only]",
    ]
    for col in numeric:
        if col in candidates:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    agg_cols = [c for c in numeric if c in candidates]
    rows = []
    for keys, group in candidates.groupby(["study", "config_id", "candidate"], dropna=False, sort=True):
        row = dict(zip(["study", "config_id", "candidate"], keys))
        row["n_rep"] = len(group)
        for col in agg_cols:
            values = _finite(group[col])
            row[f"{col}_mean"] = values.mean()
            row[f"{col}_median"] = values.median()
        rows.append(row)
    out = pd.DataFrame(rows)
    _write(out, os.path.join(tables, "candidate_by_scenario.csv"))
    return out


def regime_tables(tasks: pd.DataFrame, candidates: pd.DataFrame, tables: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r = tasks[tasks.study == "R"].copy()
    pattern = re.compile(r"(?P<kind>core|stress)_(?P<family>[LQNM])_n(?P<n>\d+)_(?P<noise>low|high)(?P<p40>_p40)?")
    parsed = r.config_id.str.extract(pattern)
    for col in parsed:
        r[col] = parsed[col]
    r["n"] = pd.to_numeric(r["n"], errors="coerce")
    r["p"] = np.where(r["p40"].notna(), 40, 10)
    r["oracle_best"] = r[_selector_col("oracle_best", "MSES")] if _selector_col("oracle_best", "MSES") in r else r["selector::oracle_best"]
    # selector::oracle_best is the actual schema; the branch above keeps compatibility explicit.
    if "selector::oracle_best" in r:
        r["oracle_best"] = r["selector::oracle_best"]

    rr_candidates = candidates[candidates.study == "R"].copy()
    frac_col = "nested_selected_fraction[MSES]"
    rows = []
    for config_id, group in r.groupby("config_id", sort=True):
        winners = group["oracle_best"].dropna().astype(str)
        counts = winners.value_counts()
        cg = rr_candidates[rr_candidates.config_id == config_id]
        mses_freq = cg.groupby("candidate")[frac_col].mean().dropna() if frac_col in cg else pd.Series(dtype=float)
        if mses_freq.empty and "selected_by_MSES" in cg:
            mses_freq = cg.groupby("candidate")["selected_by_MSES"].mean().dropna()
        match = _finite(_selector_series(group, "top1", "MSES"))
        meta = group.iloc[0]
        rows.append({
            "config_id": config_id, "family": meta.family, "n": int(meta.n),
            "proxy_noise": meta.noise, "p": int(meta.p), "n_rep": len(group),
            "modal_oracle": counts.index[0], "modal_oracle_frequency": counts.iloc[0] / len(group),
            "modal_MSES": mses_freq.idxmax() if len(mses_freq) else np.nan,
            "MSES_oracle_match_rate": match.mean(),
            "distinct_oracle_winners": int(len(counts)),
        })
    regimes = pd.DataFrame(rows)
    _write(regimes, os.path.join(tables, "oracle_switching_by_regime.csv"))

    freq_rows = []
    for scope_name, group in [("pooled", r), *[(f"family_{f}", r[r.family == f]) for f in "LQNM"]]:
        counts = group.oracle_best.value_counts()
        for candidate, count in counts.items():
            freq_rows.append({"scope": scope_name, "candidate": candidate, "count": count,
                              "frequency": count / len(group)})
    frequencies = pd.DataFrame(freq_rows)
    _write(frequencies, os.path.join(tables, "oracle_winner_frequencies.csv"))

    transitions = []
    core = r[r.kind == "core"]
    for (family, noise, seed), group in core.groupby(["family", "noise", "scientific_seed"]):
        by_n = group.set_index("n").oracle_best.to_dict()
        if 1500 in by_n and 6000 in by_n:
            transitions.append({"transition": "sample_size", "family": family, "stratum": noise,
                                "scientific_seed": seed, "from": by_n[1500], "to": by_n[6000],
                                "changed": by_n[1500] != by_n[6000]})
    for (family, n, seed), group in core.groupby(["family", "n", "scientific_seed"]):
        by_noise = group.set_index("noise").oracle_best.to_dict()
        if "low" in by_noise and "high" in by_noise:
            transitions.append({"transition": "proxy_noise", "family": family, "stratum": n,
                                "scientific_seed": seed, "from": by_noise["low"], "to": by_noise["high"],
                                "changed": by_noise["low"] != by_noise["high"]})
    transitions = pd.DataFrame(transitions)
    _write(transitions, os.path.join(tables, "oracle_winner_transitions.csv"))
    return regimes, frequencies, transitions


def robustness_tables(tasks: pd.DataFrame, tables: str) -> dict[str, pd.DataFrame]:
    output = {}
    for study, filename in [("G2", "proxy_corruption.csv"), ("H2", "sample_size_scaling.csv")]:
        rows = []
        for config_id, group in tasks[tasks.study == study].groupby("config_id", sort=True):
            row = {"study": study, "config_id": config_id, "n_rep": len(group)}
            if study == "G2":
                row.update({
                    "corruption_kind": group.proxy_mod_kind.iloc[0],
                    "corruption_level": float(group.proxy_mod_level.iloc[0]),
                    "n": int(group.n.iloc[0]),
                })
            else:
                row.update({
                    "family": str(config_id).split("_", 1)[0],
                    "n": int(group.n.iloc[0]),
                })
            for selector in ["MSES", "product", "variance_only", "fixed_sieve1", "fixed_kernel"]:
                for metric in ["oracle_ratio", "regret", "catastrophic", "error", "top1"]:
                    values = _finite(_selector_series(group, metric, selector))
                    if len(values):
                        row[f"{metric}_mean[{selector}]"] = values.mean()
                        row[f"{metric}_median[{selector}]"] = values.median()
            survivor = _finite(group.get(_selector_col("survivor_count", "MSES"), pd.Series(dtype=float)))
            row["survivor_count_mean[MSES]"] = survivor.mean()
            rows.append(row)
        out = pd.DataFrame(rows)
        _write(out, os.path.join(tables, filename))
        output[study] = out

    f_rows = []
    for config_id, group in tasks[tasks.study == "F2"].groupby("config_id", sort=True):
        ign = [_json(value, {}) for value in group.get("ignorability", [])]
        ign_ate = pd.Series([d.get("ate_err", np.nan) for d in ign], dtype=float)
        row = {"config_id": config_id, "c_U": float(group.c_U.iloc[0]), "n_rep": len(group),
               "ignorability_ate_error_mean": _finite(ign_ate).mean()}
        for selector in ["MSES", "fixed_sieve1", "fixed_kernel", "oracle"]:
            error = _finite(_selector_series(group, "error", selector))
            row[f"proximal_ate_error_mean[{selector}]"] = error.mean()
        f_rows.append(row)
    f2 = pd.DataFrame(f_rows)
    _write(f2, os.path.join(tables, "confounding_strength.csv"))
    output["F2"] = f2

    i_rows = []
    for config_id, group in tasks[tasks.study == "I2"].groupby("config_id", sort=True):
        status = group.get(_selector_col("status", "MSES"), pd.Series(index=group.index, dtype=object)).astype(str)
        abstain = status.eq("ABSTAIN_LIBRARY_INADEQUATE")
        i_rows.append({
            "config_id": config_id, "library_truth": group.library_truth.iloc[0], "n_rep": len(group),
            "abstention_rate": abstain.mean(),
            "false_abstention_rate": abstain.mean() if "adequate_easy" in str(group.library_truth.iloc[0]) else np.nan,
            "true_rejection_rate": abstain.mean() if "inadequate" in str(group.library_truth.iloc[0]) else np.nan,
            "survivor_count_mean": _finite(group.get(_selector_col("survivor_count", "MSES"), pd.Series(dtype=float))).mean(),
        })
    i2 = pd.DataFrame(i_rows)
    _write(i2, os.path.join(tables, "library_inadequacy_abstention.csv"))
    output["I2"] = i2
    return output


def rhc_tables(tasks: pd.DataFrame, candidates: pd.DataFrame, tables: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    j = tasks[tasks.study == "J2"].copy()
    if j.empty:
        return pd.DataFrame(), pd.DataFrame()
    j["ate_scalar"] = j.ate.map(_first_number)
    j["ate_lo_scalar"] = j.ate_lo.map(_first_number)
    j["ate_hi_scalar"] = j.ate_hi.map(_first_number)
    for col in ["ess", "max_q", "q_balance", "survivor_count"]:
        j[col] = pd.to_numeric(j[col], errors="coerce")
    summary = pd.DataFrame([{
        "n_repeated_splits": len(j), "ate_mean": j.ate_scalar.mean(), "ate_median": j.ate_scalar.median(),
        "ate_sd": j.ate_scalar.std(ddof=1), "ate_min": j.ate_scalar.min(), "ate_max": j.ate_scalar.max(),
        "negative_effect_rate": (j.ate_scalar < 0).mean(),
        "ci_excludes_zero_rate": ((j.ate_lo_scalar > 0) | (j.ate_hi_scalar < 0)).mean(),
        "ess_mean": j.ess.mean(), "ess_min": j.ess.min(), "max_q_mean": j.max_q.mean(),
        "max_q_max": j.max_q.max(), "q_balance_mean": j.q_balance.mean(),
        "survivor_count_mean": j.survivor_count.mean(),
    }])
    _write(summary, os.path.join(tables, "rhc_stability_summary.csv"))

    cj = candidates[candidates.study == "J2"].copy()
    selection = []
    for candidate, group in cj.groupby("candidate"):
        selection.append({
            "candidate": candidate, "n_splits": len(group),
            "full_sample_deployment_screen_survival_frequency": _finite(
                group["deployment_screen_survival"]
            ).mean(),
            "full_sample_deployment_MSES_selection_frequency": _finite(
                group["deployment_selected[MSES]"]
            ).mean(),
            "outer_fold_MSES_selection_frequency": _finite(
                group["nested_selected_fraction[MSES]"]
            ).mean(),
        })
    selection = pd.DataFrame(selection)
    _write(selection, os.path.join(tables, "rhc_candidate_frequencies.csv"))
    return summary, selection


def _savefig(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_mechanism(tasks: pd.DataFrame, cells: pd.DataFrame, figures: str):
    a = tasks[tasks.study == "A2"].copy()
    pairs = [
        ("true_h_error", "D_h", "True h error", r"Held-out $D_h$"),
        ("true_q_error", "D_q", "True q error", r"Held-out $D_q$"),
        ("true_error_product", "moment_product", "True bridge-error product", r"$D_hD_q$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    for ax, (x, y, xl, yl) in zip(axes.flat[:3], pairs):
        ax.hexbin(pd.to_numeric(a[x]), pd.to_numeric(a[y]), gridsize=35, mincnt=1, cmap="Blues")
        ax.set(xlabel=xl, ylabel=yl)
    ax = axes.flat[3]
    both = cells[(cells.r_h > 0) & (cells.r_q > 0)].sort_values("predicted_bias_mean")
    ax.errorbar(both.predicted_bias_mean, both.incremental_ate_bias_mean,
                yerr=1.96 * both.incremental_ate_bias_se, fmt="o", color="#c62828", capsize=2)
    limits = [min(both.predicted_bias_mean.min(), both.incremental_ate_bias_mean.min()),
              max(both.predicted_bias_mean.max(), both.incremental_ate_bias_mean.max())]
    ax.plot(limits, limits, "k--", lw=1, label="exact prediction")
    ax.set(xlabel="Predicted population bias", ylabel="Mean incremental ATE bias")
    ax.legend(frameon=False)
    fig.suptitle("Figure 1. Aligned corruption identifies bridge error and product bias", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, os.path.join(figures, "figure1_mechanism.png"))


def figure_moment_variance(candidates: pd.DataFrame, figures: str):
    c = candidates[candidates.study.isin(["B2", "C2", "D2", "E2", "R"])].copy()
    c["moment_product"] = pd.to_numeric(c.D_h_rff_l2, errors="coerce") * pd.to_numeric(c.D_q_rff_l2, errors="coerce")
    c["po_var_mean"] = pd.to_numeric(c.po_var_mean, errors="coerce")
    keep = np.isfinite(c.moment_product) & np.isfinite(c.po_var_mean) & (c.moment_product > 0) & (c.po_var_mean > 0)
    c = c[keep]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for candidate, group in c.groupby("candidate"):
        ax.scatter(group.moment_product, group.po_var_mean, s=9, alpha=.16,
                   label=SHORT.get(candidate, candidate), rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xlabel=r"Moment-only score $D_hD_q$", ylabel="OOF pseudo-outcome variance")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_title("Figure 2. Small moment discrepancy can coexist with high uncertainty", fontweight="bold")
    _savefig(fig, os.path.join(figures, "figure2_moment_vs_variance.png"))


def figure_adequacy_uncertainty(candidates: pd.DataFrame, figures: str):
    # A fixed, non-outcome-based display case: Study B2 and its smallest registered final seed.
    eligible = candidates[candidates.study.eq("B2")].copy()
    seed = eligible.scientific_seed.min()
    c = eligible[eligible.scientific_seed == seed].copy()
    if c.empty:
        return
    c["moment_product"] = pd.to_numeric(c.D_h_rff_l2, errors="coerce") * pd.to_numeric(c.D_q_rff_l2, errors="coerce")
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for _, row in c.iterrows():
        passed = bool(row.get("screen_pass", False))
        ax.scatter(row.moment_product, row.po_var_mean, s=100 if passed else 65,
                   marker="o" if passed else "x", color="#1565c0" if passed else "#9e9e9e")
        ax.annotate(SHORT.get(row.candidate, row.candidate), (row.moment_product, row.po_var_mean),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
        if bool(row.get("oracle_best", False)):
            ax.scatter(row.moment_product, row.po_var_mean, s=260, facecolors="none",
                       edgecolors="#d32f2f", marker="*", linewidths=1.6, label="oracle best")
        selected_fraction = row.get("nested_selected_fraction[MSES]", np.nan)
        if not np.isfinite(selected_fraction):
            selected_fraction = float(bool(row.get("selected_by_MSES", False)))
        if float(selected_fraction) > 0:
            ax.scatter(row.moment_product, row.po_var_mean, s=180, facecolors="none",
                       edgecolors="#212121", marker="s", linewidths=1.4, label="MSES-selected in outer folds")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xlabel=r"Moment diagnostic $D_hD_q$", ylabel="OOF pseudo-outcome variance")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), frameon=False)
    ax.set_title(f"Figure 3. Adequacy and uncertainty (B2, first seed {int(seed)})", fontweight="bold")
    _savefig(fig, os.path.join(figures, "figure3_adequacy_uncertainty.png"))


def figure_switching(regimes: pd.DataFrame, figures: str):
    if regimes.empty:
        return
    columns = ["1500/high", "1500/low", "6000/high", "6000/low", "3000/low/p40"]
    families = list("LQNM")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, len(columns)); ax.set_ylim(0, len(families)); ax.invert_yaxis()
    for i, family in enumerate(families):
        for j, label in enumerate(columns):
            parts = label.split("/")
            n, noise = int(parts[0]), parts[1]
            p = 40 if "p40" in parts else 10
            cell = regimes[(regimes.family == family) & (regimes.n == n) &
                           (regimes.proxy_noise == noise) & (regimes.p == p)]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor="#f5f5f5", edgecolor="white"))
            if len(cell):
                row = cell.iloc[0]
                text = f"O {SHORT.get(row.modal_oracle, row.modal_oracle)}\nM {SHORT.get(row.modal_MSES, row.modal_MSES)}\nmatch {row.MSES_oracle_match_rate:.0%}"
                ax.text(j + .5, i + .5, text, ha="center", va="center", fontsize=8)
    ax.set_xticks(np.arange(len(columns)) + .5, columns)
    ax.set_yticks(np.arange(len(families)) + .5, families)
    ax.set_xlabel("n / proxy noise / dimension"); ax.set_ylabel("Structural family")
    ax.set_title("Figure 4. Oracle (O) and MSES (M) solver switching", fontweight="bold")
    for spine in ax.spines.values(): spine.set_visible(False)
    _savefig(fig, os.path.join(figures, "figure4_oracle_switching.png"))


def figure_selection(tasks: pd.DataFrame, figures: str):
    core = tasks[tasks.study.isin(["B2", "C2", "D2", "E2", "R"])]
    selectors = ["MSES", "product", "variance_only", "fixed_sieve1", "fixed_kernel", "biasvar_mult", "oracle"]
    data, labels = [], []
    for selector in selectors:
        values = _finite(_selector_series(core, "oracle_ratio", selector))
        data.append(values)
        labels.append(selector.replace("fixed_", "fixed\n").replace("variance_only", "variance"))
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True,
                    medianprops={"color": "black"})
    for patch, selector in zip(bp["boxes"], selectors):
        patch.set_facecolor(COLORS.get(selector, "#90a4ae")); patch.set_alpha(.72)
    # Plot every evaluable task so catastrophic tails remain visible rather than being hidden by
    # the conventional boxplot whiskers. Jitter is deterministic and purely presentational.
    rng = np.random.default_rng(20260813)
    for position, (values, selector) in enumerate(zip(data, selectors), start=1):
        x = position + rng.uniform(-.16, .16, size=len(values))
        ax.scatter(x, values, s=5, alpha=.10, color=COLORS.get(selector, "#546e7a"),
                   linewidths=0, rasterized=True)
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_yscale("log"); ax.set_ylabel("ATE oracle ratio (log scale; every evaluable task shown)")
    ax.set_title("Figure 5. Confirmatory selection performance and full tails", fontweight="bold")
    _savefig(fig, os.path.join(figures, "figure5_selection_performance.png"))


def figure_rhc(tasks: pd.DataFrame, selection: pd.DataFrame, figures: str):
    j = tasks[tasks.study == "J2"].copy()
    if j.empty:
        return
    j["ate_scalar"] = j.ate.map(_first_number)
    j["ate_lo_scalar"] = j.ate_lo.map(_first_number)
    j["ate_hi_scalar"] = j.ate_hi.map(_first_number)
    j["ess"] = pd.to_numeric(j.ess, errors="coerce")
    j["max_q"] = pd.to_numeric(j.max_q, errors="coerce")
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.0))
    axes = axes.flat
    x = np.arange(len(selection)); width = .25
    axes[0].bar(
        x - width,
        selection.full_sample_deployment_screen_survival_frequency,
        width,
        label="full-sample screen survives",
    )
    axes[0].bar(
        x,
        selection.full_sample_deployment_MSES_selection_frequency,
        width,
        label="full-sample deployment",
    )
    axes[0].bar(
        x + width,
        selection.outer_fold_MSES_selection_frequency,
        width,
        label="outer-fold choices",
    )
    axes[0].set_xticks(x, [SHORT.get(v, v) for v in selection.candidate], rotation=35, ha="right")
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("Frequency"); axes[0].legend(frameon=False)
    axes[1].hist(j.ate_scalar.dropna(), bins=min(10, max(4, len(j) // 3)), color="#1565c0", alpha=.78)
    axes[1].axvline(0, color="black", ls="--", lw=1); axes[1].set_xlabel("Estimated ATE")
    ordered = j.sort_values("scientific_seed").reset_index(drop=True)
    y = np.arange(len(ordered))
    ate = ordered.ate_scalar.to_numpy(float)
    lo = ordered.ate_lo_scalar.to_numpy(float)
    hi = ordered.ate_hi_scalar.to_numpy(float)
    axes[2].errorbar(ate, y, xerr=np.vstack([ate - lo, hi - ate]),
                     fmt="o", ms=3, color="#c62828", alpha=.75)
    axes[2].axvline(0, color="black", ls="--", lw=1)
    axes[2].set(xlabel="ATE with 95% CI", ylabel="Repeated split")
    axes[3].scatter(j.ess, j.max_q, color="#00897b", alpha=.78)
    axes[3].set(xlabel="Effective sample size", ylabel="Maximum q weight")
    axes[3].set_title("Weight/ESS diagnostic")
    fig.suptitle("Figure 6. RHC screening, selection, and repeated-split stability", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, os.path.join(figures, "figure6_rhc.png"))


def evidence_summary(tasks: pd.DataFrame, selectors: pd.DataFrame, cells: pd.DataFrame,
                     diagnostics: pd.DataFrame, regimes: pd.DataFrame, transitions: pd.DataFrame,
                     robustness: dict[str, pd.DataFrame], rhc: pd.DataFrame,
                     paired: pd.DataFrame) -> dict:
    def selector_row(scope, selector):
        row = selectors[(selectors.scope == scope) & (selectors.selector == selector)]
        return {} if row.empty else row.iloc[0].to_dict()
    mes = selector_row("core_plus_switching", "MSES")
    prod = selector_row("core_plus_switching", "product")
    var = selector_row("core_plus_switching", "variance_only")
    s1 = selector_row("core_plus_switching", "fixed_sieve1")
    kernel = selector_row("core_plus_switching", "fixed_kernel")
    both = cells[(cells.r_h > 0) & (cells.r_q > 0)]
    single = cells[((cells.r_h > 0) & (cells.r_q == 0)) | ((cells.r_h == 0) & (cells.r_q > 0))]
    relation = diagnostics.set_index("relation") if len(diagnostics) else pd.DataFrame()
    r_tasks = tasks[tasks.study == "R"]
    unique_oracles = sorted(r_tasks.get("selector::oracle_best", pd.Series(dtype=str)).dropna().astype(str).unique())
    non_sieve = tasks[tasks.study.isin(["B2", "C2", "D2", "E2", "R"]) &
                      tasks.get("selector::oracle_best", pd.Series(index=tasks.index)).ne("sieve1_sieve1")]
    fixed_sieve_error = pd.to_numeric(
        non_sieve.get(
            _selector_col("error", "fixed_sieve1"),
            pd.Series(index=non_sieve.index, dtype=float),
        ),
        errors="coerce",
    )
    mses_error = pd.to_numeric(
        non_sieve.get(
            _selector_col("error", "MSES"), pd.Series(index=non_sieve.index, dtype=float)
        ),
        errors="coerce",
    )
    paired_keep = np.isfinite(fixed_sieve_error) & np.isfinite(mses_error)
    paired_sieve_gain = fixed_sieve_error[paired_keep] - mses_error[paired_keep]
    out = {
        "counts": tasks.study.value_counts().sort_index().to_dict(),
        "mechanism": {
            "single_side_max_abs_incremental_cell_mean": float(single.incremental_ate_bias_mean.abs().max()),
            "both_corrupted_max_abs_incremental_cell_mean": float(both.incremental_ate_bias_mean.abs().max()),
            "predicted_vs_incremental": relation.loc["predicted_vs_incremental_ate_bias"].to_dict(),
            "D_h_vs_true_h_error": relation.loc["D_h_vs_true_h_error"].to_dict(),
            "D_q_vs_true_q_error": relation.loc["D_q_vs_true_q_error"].to_dict(),
            "moment_product_vs_true_error_product": relation.loc["moment_product_vs_true_error_product"].to_dict(),
        },
        "selection_core_plus_switching": {"MSES": mes, "product": prod, "variance_only": var,
                                           "fixed_sieve1": s1, "fixed_kernel": kernel},
        "paired_error_comparisons": paired.to_dict(orient="records"),
        "switching": {
            "distinct_oracle_candidates": unique_oracles,
            "n_distinct_oracle_candidates": len(unique_oracles),
            "regimes_with_non_sieve1_modal_oracle": int((regimes.modal_oracle != "sieve1_sieve1").sum()),
            "n_regimes": len(regimes),
            "sample_or_proxy_transition_rate": float(pd.to_numeric(transitions.changed, errors="coerce").mean()),
            "mean_MSES_oracle_match_rate": float(regimes.MSES_oracle_match_rate.mean()),
            "mean_fixed_sieve1_minus_MSES_error_when_oracle_not_sieve1": float(paired_sieve_gain.mean()),
        },
        "abstention": robustness.get("I2", pd.DataFrame()).to_dict(orient="records"),
        "rhc": {} if rhc.empty else rhc.iloc[0].to_dict(),
    }
    return _json_roundtrip(out)


def _json_roundtrip(value):
    """Convert numpy values and non-finite numbers into strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): _json_roundtrip(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_roundtrip(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["final", "development"], default="final")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    phase = os.path.join(ROUND2, args.phase)
    audit_path = os.path.join(phase, "AGGREGATION_AUDIT.json")
    audit = json.load(open(audit_path))
    if not args.allow_incomplete and (audit["missing_rows"] or audit["failed_rows"]):
        raise RuntimeError(f"refusing incomplete confirmatory analysis: {audit}")
    tasks = pd.read_csv(os.path.join(phase, "raw.csv"), low_memory=False)
    candidates = pd.read_csv(os.path.join(phase, "raw_candidates.csv"), low_memory=False)
    analysis = os.path.join(phase, "analysis")
    tables, figures = os.path.join(analysis, "tables"), os.path.join(analysis, "figures")
    os.makedirs(tables, exist_ok=True); os.makedirs(figures, exist_ok=True)

    cells, diagnostics = mechanism_tables(tasks, tables)
    selectors = selector_tables(tasks, tables)
    paired = paired_selector_tables(tasks, tables)
    screening_tables(tasks, tables)
    candidate_tables(candidates, tables)
    regimes, frequencies, transitions = regime_tables(tasks, candidates, tables)
    robustness = robustness_tables(tasks, tables)
    rhc, rhc_selection = rhc_tables(tasks, candidates, tables)

    figure_mechanism(tasks, cells, figures)
    figure_moment_variance(candidates, figures)
    figure_adequacy_uncertainty(candidates, figures)
    figure_switching(regimes, figures)
    figure_selection(tasks, figures)
    figure_rhc(tasks, rhc_selection, figures)

    summary = evidence_summary(tasks, selectors, cells, diagnostics, regimes, transitions,
                               robustness, rhc, paired)
    with open(os.path.join(analysis, "EVIDENCE_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    with open(os.path.join(analysis, "README.md"), "w") as f:
        f.write("""# Round-2 analysis artifacts

All files here are deterministic reconstructions from `final/raw.csv` and
`final/raw_candidates.csv`, which are themselves reconstructed from the immutable per-task
`result.json` records listed in the final manifest. The aggregation audit gives the exact
complete/missing/failure counts.

The unprefixed selector metrics are the primary nested outer-fold evaluations. Columns beginning
`global_oof_` and candidate flags named `selected_by_*` are descriptive full-sample OOF
diagnostics; `nested_selected_fraction[*]` records the confirmatory outer-fold choices. The
`biasvar_mult` and `biasvar_add` rows remain `EXPLORATORY_POSTHOC_FROM_ROUND1` throughout.

For the non-oracle RHC repeated-split study (J2), candidate-frequency tables explicitly separate
the full-sample OOF deployment screen/decision from the four outer-fold choices stored in
`nested_selected_by_fold`. The reported deployment ATEs are not an independent validation sample.

`tables/` contains mechanism, candidate, selector, screening, switching, robustness, abstention,
and RHC summaries. `figures/` contains the six claim-directed figures required by the Round-2
protocol. `EVIDENCE_SUMMARY.json` is a compact machine-readable basis for the final report.
""")
    print(json.dumps({"tables": len(os.listdir(tables)), "figures": len(os.listdir(figures)),
                      "evidence_summary": os.path.join(analysis, "EVIDENCE_SUMMARY.json")}, indent=2))


if __name__ == "__main__":
    main()
