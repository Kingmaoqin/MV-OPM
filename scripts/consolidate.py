"""Consolidate all results/{Ex}/REPORT.md into a single PASS/FAIL board.

    python scripts/consolidate.py
Writes results/SUMMARY.md.
"""
from __future__ import annotations

import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

ORDER = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"]


def parse(report_path):
    with open(report_path) as f:
        txt = f.read()
    title = txt.splitlines()[0].lstrip("# ").strip()
    m = re.search(r"\*\*Overall: (PASS|FAIL)\*\*\s*\(([^)]*)\)", txt)
    overall = m.group(1) if m else "?"
    count = m.group(2) if m else ""
    crits = re.findall(r"- (✅ PASS|❌ FAIL) — \*\*(.+?)\*\*: (.+)", txt)
    return title, overall, count, crits


def main():
    lines = ["# OPM v2 — Consolidated Results Board", "",
             "Auto-generated from each `results/{Ex}/REPORT.md`. See per-experiment reports "
             "for full tables/figures and `DECISIONS.md` for documented deviations.", "",
             "| Exp | Title | Overall | Criteria |", "|-----|-------|---------|----------|"]
    details = []
    for ex in ORDER:
        rp = os.path.join(RES, ex, "REPORT.md")
        if not os.path.exists(rp):
            lines.append(f"| {ex} | — | ⏳ not run | — |")
            continue
        title, overall, count, crits = parse(rp)
        badge = "✅ PASS" if overall == "PASS" else ("❌ FAIL" if overall == "FAIL" else "?")
        lines.append(f"| {ex} | {title} | {badge} | {count} |")
        details.append(f"### {ex} — {title}  ({badge})")
        for tag, name, det in crits:
            details.append(f"- {tag} — {name}")
    lines += ["", "## Criterion-level detail", ""] + details
    out = os.path.join(RES, "SUMMARY.md")
    os.makedirs(RES, exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
