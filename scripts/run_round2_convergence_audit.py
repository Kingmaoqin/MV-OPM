"""Run and aggregate the observed-data-only kernel convergence audit."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import concurrent.futures
import json
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "mvopm_round2", "development", "convergence_audit")


def _one(args):
    scenario, seed, n = args
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from opm.experiments.mvopm_round2.convergence_audit import audit_one
    t0 = time.time()
    rows = audit_one(scenario, seed, n=n)
    return {"scenario": scenario, "seed": seed, "n": n, "runtime_s": time.time() - t0, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    seeds = json.load(open(os.path.join(ROOT, "configs", "seeds", "mvopm_round2_final.json")))["development_only_seeds"]
    tasks = [(scenario, seed, 1500) for scenario in ("S1", "S2", "nonlinear", "hamd") for seed in seeds]
    os.makedirs(OUT, exist_ok=True)
    pending, results = [], []
    for scenario, seed, n in tasks:
        path = os.path.join(OUT, scenario, f"seed{seed}.json")
        if args.resume and os.path.exists(path):
            results.append(json.load(open(path)))
        else:
            pending.append((scenario, seed, n))
    os.environ.update({
        "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads), "NUMEXPR_NUM_THREADS": str(args.threads),
    })
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(_one, pending):
            path = os.path.join(OUT, result["scenario"], f"seed{result['seed']}.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(result, f)
            results.append(result)
            print(f"{len(results)}/{len(tasks)} {result['scenario']} seed{result['seed']} {result['runtime_s']:.1f}s", flush=True)

    rows = [row for result in results for row in result["rows"]]
    df = pd.DataFrame(rows).sort_values(["scenario", "seed", "budget"])
    df.to_csv(os.path.join(OUT, "observed_convergence.csv"), index=False)
    from opm.experiments.mvopm_round2.convergence_audit import choose_plateau_budget
    suggested = choose_plateau_budget(rows)
    summary = (df.groupby(["scenario", "budget"], as_index=False)
               .agg(median_D=("D_total_rff_l2", "median"),
                    median_h_change=("h_prediction_change", "median"),
                    median_q_change=("q_prediction_change", "median"),
                    median_objective=("optimizer_best_resid", "median")))
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    report = [
        "# Round-2 kernel convergence audit", "",
        "This audit used observed-data criteria only; no oracle PEHE or ATE error was loaded.", "",
        f"Plateau-rule suggested cap: **{suggested} epochs**.",
        "The preregistered confirmatory cap remains the conservative 300 epochs with early stopping,",
        "so the audit cannot reduce training after seeing oracle outcomes.", "",
        summary.to_markdown(index=False), "",
    ]
    with open(os.path.join(OUT, "CONVERGENCE_AUDIT.md"), "w") as f:
        f.write("\n".join(report))
    print(f"suggested observed-data plateau budget: {suggested}")


if __name__ == "__main__":
    main()
