"""Build immutable Round-2 task manifests under a separate result namespace."""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "mvopm_round2")
SEED_PATH = os.path.join(ROOT, "configs", "seeds", "mvopm_round2_final.json")


def _slug(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def _cfg(phase: str) -> dict:
    if phase == "development":
        return {
            "outer_folds": 2, "inner_folds": 2, "n_rff": 48, "n_boot": 499,
            "alpha": 0.05, "device": "cpu", "run_nested": False,
            "bridge": {"max_epochs": 12, "patience": 6, "eval_every": 3, "batch_size": 512},
            "head": {"max_epochs": 30, "patience": 6},
        }
    if phase != "final":
        raise ValueError("phase must be development or final")
    return {
        "outer_folds": 4, "inner_folds": 3, "n_rff": 200, "n_boot": 999,
        "alpha": 0.05, "device": "cpu", "run_nested": True,
        "bridge": {"max_epochs": 300, "patience": 45, "eval_every": 3, "batch_size": 1024},
        "head": {"max_epochs": 300, "patience": 30},
    }


def build(phase: str) -> list[dict]:
    from opm.experiments.mvopm_round2.regime_matrix import frozen_regimes

    seeds = json.load(open(SEED_PATH))
    bank = seeds["scientific_seeds"]
    development_bank = seeds["development_only_seeds"]
    cfg = _cfg(phase)
    rows = []
    config_index = 0

    def use(study, final_values, development_count=1):
        values = list(final_values)
        if phase == "final":
            return values
        # Development tasks reuse the explicitly reserved development seeds across configurations;
        # no final scientific seed is exposed before confirmatory execution.
        return list(development_bank[:development_count])

    def add(*, study, dispatch, scientific_seed, config_id, K, **kw):
        nonlocal config_index
        row = {
            "study": study, "dispatch": dispatch, "scientific_seed": int(scientific_seed),
            "split_seed": int(scientific_seed),
            "bootstrap_seed": 700000000 + int(scientific_seed) + 10000 * config_index,
            "bank_seed": 800000000 + int(scientific_seed) + 10000 * config_index,
            "config_id": config_id, "K": K, "cfg": cfg, **kw,
        }
        row["out_path"] = os.path.join(
            OUT, phase, study, _slug(config_id), f"seed{scientific_seed}", "result.json"
        )
        rows.append(row)
        config_index += 1

    # A2: 25 frozen corruption cells × 200 replications.
    grid = (0.0, 0.05, 0.10, 0.20, 0.40)
    a_seeds = use("A2", bank["A2"], 2)
    for rh in grid:
        for rq in grid:
            for seed in a_seeds:
                add(study="A2", dispatch="aligned_mechanism", scientific_seed=seed,
                    config_id=f"rh{rh}_rq{rq}_n4000", K=2, r_h=rh, r_q=rq, n=4000, c_U=1.0)

    main = [("B2", "S1", 4, 4000), ("C2", "S2", 4, 4000),
            ("D2", "nonlinear", 2, 4000), ("E2", "hamd", 4, 1500)]
    for study, scenario, K, n in main:
        for seed in use(study, bank[study], 1):
            feasible = (["kernel_kernel", "kernel_sieve1", "sieve1_kernel", "sieve1_sieve1"]
                        if scenario == "hamd" else None)
            add(study=study, dispatch="selection_v2", scientific_seed=seed,
                config_id=f"{scenario}_n{n}", K=K, scenario=scenario, n=n, c_U=1.0,
                candidate_names=feasible)

    for regime in frozen_regimes():
        for seed in use("R", bank["R"], 1):
            add(study="R", dispatch="selection_v2", scientific_seed=seed,
                config_id=regime["regime_id"], K=2, scenario="bridge_complexity", n=regime["n"],
                c_U=1.0, regime=regime, candidate_names=regime["candidate_names"])

    for c_u in (0.0, 0.5, 1.0, 2.0):
        for seed in use("F2", bank["F2"], 1):
            add(study="F2", dispatch="selection_v2", scientific_seed=seed,
                config_id=f"S1_cU{c_u}_n4000", K=4, scenario="S1", n=4000, c_U=c_u,
                evaluate_ignorability=True)

    for kind in ("additive", "missing", "shift", "heavy_tail"):
        for level in (0.0, 0.1, 0.2, 0.3):
            for seed in use("G2", bank["G2"], 1):
                add(study="G2", dispatch="selection_v2", scientific_seed=seed,
                    config_id=f"S1_{kind}{level}_n4000", K=4, scenario="S1", n=4000, c_U=1.0,
                    proxy_mod={"kind": kind, "level": level})

    for family in ("L", "N"):
        for n in (1000, 2000, 4000, 8000):
            regime = {"family": family, "proxy_noise": "low", "p": 10}
            for seed in use("H2", bank["H2"], 1):
                add(study="H2", dispatch="selection_v2", scientific_seed=seed,
                    config_id=f"{family}_n{n}_low", K=2, scenario="bridge_complexity", n=n,
                    c_U=1.0, regime=regime)

    inadequacy = [
        ("easy_L_full", {"family": "L", "proxy_noise": "low", "p": 10}, None, {}, "adequate_easy"),
        ("N_sieve1_only", {"family": "N", "proxy_noise": "low", "p": 10},
         ["sieve1_sieve1"], {}, "inadequate_constrained_library"),
        ("N_kernel_overregularized", {"family": "N", "proxy_noise": "high", "p": 10},
         ["kernel_kernel"], {"weight_decay": 1.0}, "inadequate_severe_regularization"),
    ]
    for config_id, regime, candidates, override, truth in inadequacy:
        for seed in use("I2", bank["I2"], 1):
            add(study="I2", dispatch="selection_v2", scientific_seed=seed,
                config_id=config_id, K=2, scenario="bridge_complexity", n=4000, c_U=1.0,
                regime=regime, candidate_names=candidates, bridge_override=override, library_truth=truth)

    for seed in use("J2", bank["J2"], 1):
        add(study="J2", dispatch="rhc_v2", scientific_seed=seed,
            config_id="rhc_repeated_split", K=2)
    return rows


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "development"
    rows = build(phase)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"manifest_{phase}.jsonl")
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"{phase}: {len(rows)} rows -> {path}")
    from collections import Counter
    print("per-study:", dict(Counter(r["study"] for r in rows)))


if __name__ == "__main__":
    main()
