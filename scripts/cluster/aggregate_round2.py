"""Reconstruct Round-2 task/candidate/selector tables from raw result.json files."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "mvopm_round2")


def _cell(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, allow_nan=True)
    return value


def aggregate(manifest: str) -> dict:
    manifest_rows = [json.loads(line) for line in open(manifest)]
    tasks, candidates, missing, failures = [], [], [], []
    for expected in manifest_rows:
        path = expected["out_path"]
        if not os.path.exists(path):
            missing.append(path)
            continue
        prov = json.load(open(path))
        if prov.get("status") != "ok":
            failures.append({
                "study": expected["study"], "config_id": expected["config_id"],
                "scientific_seed": expected["scientific_seed"], "error": prov.get("error"),
            })
            continue
        result = prov["result"]
        base = {
            "study": expected["study"], "config_id": expected["config_id"],
            "scientific_seed": expected["scientific_seed"],
            "bootstrap_seed": expected["bootstrap_seed"], "split_seed": expected["split_seed"],
            "bank_seed": expected["bank_seed"], "git_commit": prov.get("git_commit"),
            "hostname": prov.get("hostname"), "runtime_s": prov.get("runtime_s"),
            "status": prov.get("status"),
        }
        task = dict(base)
        for key, value in result.items():
            if key.startswith("_"):
                continue
            if key == "selectors" and isinstance(value, dict):
                task.update({f"selector::{k}": _cell(v) for k, v in value.items()})
            else:
                task[key] = _cell(value)
        tasks.append(task)
        for name, metrics in (result.get("_candidates") or {}).items():
            row = {**base, "candidate": name}
            row.update({k: _cell(v) for k, v in metrics.items()})
            candidates.append(row)

    manifest_name = os.path.basename(manifest)
    phase = "final" if "final" in manifest_name else "development"
    phase_dir = os.path.join(OUT, phase)
    os.makedirs(phase_dir, exist_ok=True)
    task_df, candidate_df = pd.DataFrame(tasks), pd.DataFrame(candidates)
    task_path = os.path.join(phase_dir, "raw.csv")
    candidate_path = os.path.join(phase_dir, "raw_candidates.csv")
    task_df.to_csv(task_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)

    selector_cols = [c for c in task_df if c.startswith("selector::") and task_df[c].dtype.kind in "iuf"]
    summary_rows = []
    if selector_cols:
        for (study, config_id), group in task_df.groupby(["study", "config_id"], dropna=False):
            for col in selector_cols:
                vals = pd.to_numeric(group[col], errors="coerce").dropna()
                if len(vals):
                    summary_rows.append({
                        "study": study, "config_id": config_id, "metric": col.removeprefix("selector::"),
                        "n": len(vals), "mean": vals.mean(), "median": vals.median(),
                        "sd": vals.std(ddof=1), "q25": vals.quantile(.25), "q75": vals.quantile(.75),
                    })
    summary_path = os.path.join(phase_dir, "summary_selectors.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    audit = {
        "manifest_rows": len(manifest_rows), "ok_rows": len(tasks), "candidate_rows": len(candidates),
        "missing_rows": len(missing), "failed_rows": len(failures),
        "per_study_ok": dict(Counter(r["study"] for r in tasks)),
        "missing": missing, "failures": failures,
    }
    with open(os.path.join(phase_dir, "AGGREGATION_AUDIT.json"), "w") as f:
        json.dump(audit, f, indent=2)
    return audit


def main():
    manifest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "manifest_development.jsonl")
    audit = aggregate(manifest)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
