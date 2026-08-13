"""Run ONE manifest row (like one Slurm-array task) and save a raw result.json with provenance.

    python scripts/cluster/worker.py <manifest.jsonl> <row_index>

Thread caps must be set by the launcher via env (OMP_NUM_THREADS etc.) BEFORE importing torch.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def _git_commit():
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def run_row(row: dict) -> dict:
    import numpy as np  # noqa
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if row["dispatch"] in {"aligned_mechanism", "selection_v2", "rhc_v2"}:
        from opm.experiments.mvopm_round2.core import Round2Config
        from opm.experiments.mvopm_round2.studies import (
            aligned_mechanism_row, rhc_stability_row, selection_study_row,
        )
        c = row["cfg"]
        if row["dispatch"] == "aligned_mechanism":
            return aligned_mechanism_row(
                row["scientific_seed"], row["r_h"], row["r_q"], n=row["n"],
                c_U=row.get("c_U", 1.0), n_rff=c["n_rff"], bank_seed=row["bank_seed"],
            )
        cfg = Round2Config(
            K=row["K"], n_folds=c["outer_folds"], inner_folds=c["inner_folds"],
            seed=row["scientific_seed"], split_seed=row["split_seed"],
            bootstrap_seed=row["bootstrap_seed"], bank_seed=row["bank_seed"],
            n_boot=c["n_boot"], n_rff=c["n_rff"], alpha=c["alpha"], device=c["device"],
            candidate_names=row.get("candidate_names"), bridge_kwargs={**c["bridge"], **row.get("bridge_override", {})},
            head_kwargs=c["head"], run_nested=c.get("run_nested", True),
        )
        if row["dispatch"] == "rhc_v2":
            return rhc_stability_row(row["scientific_seed"], cfg)
        return selection_study_row(
            row["study"], row["scenario"], row["n"], row["scientific_seed"], cfg,
            c_U=row.get("c_U", 1.0), proxy_mod=row.get("proxy_mod"), regime=row.get("regime"),
            evaluate_ignorability=row.get("evaluate_ignorability", False),
            library_truth=row.get("library_truth"),
        )
    from opm.experiments.mvopm.core import SelCfg
    from opm.experiments.mvopm.studies import mechanism_row, selection_row, rhc_row
    c = row["cfg"]
    disp = row["dispatch"]
    if disp == "mechanism":
        return mechanism_row(row["seed"], row["r_h"], row["r_q"], n=row["n"], c_U=row.get("c_U", 1.0),
                             n_rff=c["n_rff"])
    K = 4 if row.get("scenario") in ("S1", "S2", "nonlinear", "hamd") and row.get("scenario") != "nonlinear" else 2
    # scenario K: S1/S2/hamd=4, nonlinear/exact/rhc=2
    K = {"S1": 4, "S2": 4, "hamd": 4, "nonlinear": 2, "exact": 2}.get(row.get("scenario"), 2)
    cfg = SelCfg(K=K, n_folds=c["n_folds"], seed=row["seed"], kind=c["kind"], n_rff=c["n_rff"],
                 bridge_kwargs=c["bridge"], head_kwargs=c["head"])
    if disp == "rhc":
        cfg.K = 2
        return rhc_row(row["seed"], cfg)
    if disp == "selection":
        return selection_row(row["study"], row["scenario"], row["n"], row["seed"], cfg,
                             c_U=row.get("c_U", 1.0), mod=row.get("mod"))
    raise ValueError(f"unknown dispatch {disp}")


def main():
    manifest, idx = sys.argv[1], int(sys.argv[2])
    rows = [json.loads(l) for l in open(manifest)]
    row = rows[idx]
    out_path = row["out_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()
    scientific_seed = row.get("scientific_seed", row.get("seed"))
    prov = {"experiment": row["study"], "dispatch": row["dispatch"], "seed": scientific_seed,
            "scientific_seed": scientific_seed, "bootstrap_seed": row.get("bootstrap_seed"),
            "split_seed": row.get("split_seed"), "bank_seed": row.get("bank_seed"),
            "array_task_id": idx, "git_commit": _git_commit(), "hostname": socket.gethostname(),
            "omp_threads": os.environ.get("OMP_NUM_THREADS", "?"),
            "mkl_threads": os.environ.get("MKL_NUM_THREADS", "?"),
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "?"),
            "numexpr_threads": os.environ.get("NUMEXPR_NUM_THREADS", "?"),
            "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "config": row, "warnings": []}
    try:
        result = run_row(row)
        prov.update({"status": "ok", "exit_code": 0, "runtime_s": round(time.time() - t0, 1),
                     "end_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "result": result})
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        infrastructure_markers = ("out of memory", "cannot allocate memory", "no space left", "input/output error")
        failure_kind = "infrastructure" if isinstance(e, (MemoryError, OSError)) or any(
            marker in msg.lower() for marker in infrastructure_markers
        ) else "algorithmic"
        prov.update({"status": "error", "exit_code": 1, "runtime_s": round(time.time() - t0, 1),
                     "end_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "failure_kind": failure_kind,
                     "error": msg, "traceback": traceback.format_exc()})
    def _json_default(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "item"):
            return o.item()
        if isinstance(o, set):
            return sorted(o)
        raise TypeError(f"not JSON serializable: {type(o).__name__}")
    # Never erase a prior attempt. Explicit reruns keep the same scientific seed and archive the
    # complete previous record alongside the canonical latest result.
    if os.path.exists(out_path):
        attempts = os.path.join(os.path.dirname(out_path), "attempts")
        os.makedirs(attempts, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        shutil.copy2(out_path, os.path.join(attempts, f"result_before_{stamp}_{os.getpid()}.json"))
    with open(out_path, "w") as f:
        json.dump(prov, f, default=_json_default)
    print(f"[{idx}] {row['study']} seed{scientific_seed} {prov['status']} {prov['runtime_s']}s -> {out_path}",
          flush=True)
    sys.exit(prov["exit_code"])


if __name__ == "__main__":
    main()
