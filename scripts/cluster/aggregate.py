"""Aggregate raw result.json files into per-study raw.csv/summary.csv with completeness checks.

    python scripts/cluster/aggregate.py <manifest.jsonl>

No number is emitted unless it is reconstructible from the raw per-run files (MV-OPM §12/§17).
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "mvopm")


def _flat(prov):
    r = prov.get("result", {})
    base = {"study": prov.get("experiment"), "seed": prov.get("seed"),
            "status": prov.get("status"), "runtime_s": prov.get("runtime_s"),
            "git": prov.get("git_commit")}
    flat = {k: v for k, v in r.items() if not k.startswith("_") and not isinstance(v, (dict, list))}
    return {**base, **flat}, r.get("_candidates")


def main():
    manifest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "manifest_pilot.jsonl")
    rows = [json.loads(l) for l in open(manifest)]
    recs, cand_recs, missing, errors = [], [], [], []
    for row in rows:
        p = row["out_path"]
        if not os.path.exists(p):
            missing.append(p); continue
        prov = json.load(open(p))
        if prov.get("status") != "ok":
            errors.append((row["study"], row["seed"], prov.get("error"))); continue
        flat, cands = _flat(prov)
        recs.append(flat)
        if cands:
            for nm, d in cands.items():
                cand_recs.append({"study": row["study"], "seed": row["seed"], "candidate": nm, **d})

    df = pd.DataFrame(recs)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "raw.csv"), index=False)
    if cand_recs:
        pd.DataFrame(cand_recs).to_csv(os.path.join(OUT, "raw_candidates.csv"), index=False)

    # completeness checks
    print("=== completeness ===")
    print("rows expected:", len(rows), " ok:", len(recs), " missing:", len(missing), " errors:", len(errors))
    if errors:
        print("ERRORS (first 5):", errors[:5])
    if len(df):
        print("per-study ok:", dict(Counter(df["study"])))
        nan_cols = [c for c in df.columns if df[c].dtype.kind == "f" and df[c].isna().any()]
        print("float cols with NaN:", nan_cols[:12])
        # duplicate (study,seed,config) guard via out_path uniqueness already; check seed dups per study
        for s, g in df.groupby("study"):
            dup = g["seed"].duplicated().sum()
            if dup:
                print(f"  WARN study {s}: {dup} duplicate seeds")
    print("wrote", os.path.join(OUT, "raw.csv"))


if __name__ == "__main__":
    main()
