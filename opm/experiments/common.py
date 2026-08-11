"""Shared experiment plumbing: config loading, results IO, REPORT.md writer."""
from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(ROOT, "configs")
RESULTS_DIR = os.path.join(ROOT, "results")
ARTIFACT_DIR = os.path.join(ROOT, "artifacts")


def parse_kv(argv: List[str]) -> Dict[str, str]:
    out = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except (ValueError, TypeError):
            pass
    if isinstance(v, str) and v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def load_config(exp: str, overrides: Dict[str, str] = None) -> dict:
    path = os.path.join(CONFIG_DIR, f"{exp}.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("experiment", exp)
    for k, v in (overrides or {}).items():
        cfg[k] = _coerce(v)
    return cfg


def results_dir(exp: str) -> str:
    d = os.path.join(RESULTS_DIR, exp)
    os.makedirs(os.path.join(d, "figures"), exist_ok=True)
    return d


def artifact_dir(exp: str, seed) -> str:
    d = os.path.join(ARTIFACT_DIR, exp, str(seed))
    os.makedirs(d, exist_ok=True)
    return d


def save_raw(exp: str, rows: List[dict]) -> str:
    d = results_dir(exp)
    path = os.path.join(d, "raw.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_report(exp: str, title: str, criteria: List[dict], body_sections: List[str]) -> str:
    """criteria: list of {name, passed(bool), detail}. Writes REPORT.md with PASS/FAIL."""
    d = results_dir(exp)
    path = os.path.join(d, "REPORT.md")
    n_pass = sum(c["passed"] for c in criteria)
    overall = "PASS" if all(c["passed"] for c in criteria) else "FAIL"
    lines = [f"# {exp} — {title}", "",
             f"**Overall: {overall}**  ({n_pass}/{len(criteria)} acceptance criteria met)", "",
             "## Acceptance criteria", ""]
    for c in criteria:
        tag = "✅ PASS" if c["passed"] else "❌ FAIL"
        lines.append(f"- {tag} — **{c['name']}**: {c['detail']}")
    lines.append("")
    lines.extend(body_sections)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
