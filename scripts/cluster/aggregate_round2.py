"""Reconstruct Round-2 task/candidate/selector tables from raw result.json files."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from opm.provenance import (
    paths_overlap,
    registered_manifest_phase,
    sha256_file,
    validate_manifest_rows,
    validate_result_checksum,
    validate_result_provenance,
)

OUT = os.path.join(ROOT, "results", "mvopm_round2")
FROZEN_FINAL_MANIFEST = os.path.join(OUT, "manifest_final.jsonl")
FROZEN_FINAL_ROOT = os.path.join(OUT, "final")
FROZEN_DEVELOPMENT_ROOT = os.path.join(OUT, "development")


def _cell(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, allow_nan=True)
    return value


def _numeric_list(value):
    """Return finite numeric list elements without changing the retained raw cell."""
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        try:
            item = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(item):
            out.append(item)
    return out


def aggregate(manifest: str, *, output_root: str | None = None) -> dict:
    manifest_hash = sha256_file(manifest)
    registered_phase = registered_manifest_phase(manifest_hash)
    if registered_phase and output_root is None:
        raise RuntimeError(
            "refusing to aggregate a registered frozen Round-2 manifest without a disjoint "
            "output_root for an audit reconstruction"
        )
    with open(manifest, encoding="utf-8") as handle:
        manifest_rows = [json.loads(line) for line in handle if line.strip()]
    validate_manifest_rows(manifest_rows)
    manifest_name = os.path.basename(manifest)
    phase = registered_phase or ("final" if "final" in manifest_name else "development")
    phase_dir = os.path.join(output_root or OUT, phase)
    protected_roots = [FROZEN_FINAL_ROOT]
    if registered_phase == "development":
        protected_roots.append(FROZEN_DEVELOPMENT_ROOT)
    if any(paths_overlap(phase_dir, protected) for protected in protected_roots):
        raise RuntimeError(
            f"refusing aggregate output {phase_dir}: target intersects a frozen Round-2 root"
        )
    tasks, candidates, missing, failures = [], [], [], []
    for array_task_id, expected in enumerate(manifest_rows):
        path = expected["out_path"]
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path, encoding="utf-8") as handle:
            prov = json.load(handle)
        validate_result_checksum(path, required=bool(expected.get("source_commit")))
        validate_result_provenance(
            prov,
            expected,
            array_task_id=array_task_id,
            manifest_sha256=manifest_hash,
        )
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
            "status": prov.get("status"), "exit_code": prov.get("exit_code"),
            "array_task_id": prov.get("array_task_id"),
            "start_time_utc": prov.get("start_time_utc"),
            "end_time_utc": prov.get("end_time_utc"),
            "warnings": _cell(prov.get("warnings", [])),
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
        nested_folds = (result.get("_nested") or {}).get("folds") or []
        for name, metrics in (result.get("_candidates") or {}).items():
            row = {**base, "candidate": name}
            row.update({k: _cell(v) for k, v in metrics.items()})
            # Keep the exact arrays above, and add analysis-ready scalars with explicit names.
            for key in ("p_h", "p_q", "p_DR", "ate_ci_width_by_contrast",
                        "ate_se_by_contrast", "ess_by_arm", "po_var_by_contrast"):
                values = _numeric_list(metrics.get(key))
                if values:
                    row[f"min_{key}"] = min(values)
                    row[f"mean_{key}"] = float(np.mean(values))
                    row[f"max_{key}"] = max(values)
            if nested_folds:
                row["nested_screen_survival_fraction"] = float(np.mean([
                    name in (fold.get("survivors") or []) for fold in nested_folds
                ]))
                selector_names = sorted({
                    selector
                    for fold in nested_folds
                    for selector in (fold.get("selected_by_selector") or {})
                })
                for selector in selector_names:
                    row[f"nested_selected_fraction[{selector}]"] = float(np.mean([
                        (fold.get("selected_by_selector") or {}).get(selector) == name
                        for fold in nested_folds
                    ]))
            elif result.get("study") == "J2":
                # J2 stores a full-sample OOF deployment decision and, separately, the four
                # outer-fold choices.  Keep both with names that cannot conflate them.
                row["deployment_screen_survival"] = float(
                    name in (result.get("survivors") or [])
                )
                row["deployment_selected[MSES]"] = float(result.get("selected") == name)
                nested_selected = result.get("nested_selected_by_fold") or []
                if nested_selected:
                    row["nested_selected_fraction[MSES]"] = float(np.mean([
                        selected == name for selected in nested_selected
                    ]))
            candidates.append(row)

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
        "manifest_sha256": manifest_hash,
        "provenance_validation": "strict_config_index_seed_source_when_registered",
        "per_study_ok": dict(Counter(r["study"] for r in tasks)),
        "missing": missing, "failures": failures,
    }
    with open(os.path.join(phase_dir, "AGGREGATION_AUDIT.json"), "w") as f:
        json.dump(audit, f, indent=2)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest", nargs="?", default=os.path.join(OUT, "manifest_development.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        help="disjoint reconstruction root; required for the frozen final manifest",
    )
    args = parser.parse_args()
    audit = aggregate(args.manifest, output_root=args.output_root)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
