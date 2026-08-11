"""Run ONE manifest row (like one Slurm-array task) and save a raw result.json with provenance.

    python scripts/cluster/worker.py <manifest.jsonl> <row_index>

Thread caps must be set by the launcher via env (OMP_NUM_THREADS etc.) BEFORE importing torch.
"""
from __future__ import annotations

import json
import os
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
    prov = {"experiment": row["study"], "dispatch": row["dispatch"], "seed": row["seed"],
            "array_task_id": idx, "git_commit": _git_commit(), "hostname": socket.gethostname(),
            "omp_threads": os.environ.get("OMP_NUM_THREADS", "?"), "config": row}
    try:
        result = run_row(row)
        prov.update({"status": "ok", "exit_code": 0, "runtime_s": round(time.time() - t0, 1),
                     "result": result})
    except Exception as e:
        prov.update({"status": "error", "exit_code": 1, "runtime_s": round(time.time() - t0, 1),
                     "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
    with open(out_path, "w") as f:
        json.dump(prov, f, default=lambda o: None)
    print(f"[{idx}] {row['study']} seed{row['seed']} {prov['status']} {prov['runtime_s']}s -> {out_path}",
          flush=True)
    sys.exit(prov["exit_code"])


if __name__ == "__main__":
    main()
