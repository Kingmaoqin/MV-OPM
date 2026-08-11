#!/usr/bin/env bash
# Reproduce the full OPM v2 experiment suite (spec Section 8 milestones).
# Usage: bash scripts/run_all.sh   (runs from the repo root; uses conda env MDPC)
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python}"

echo "== unit tests (spec Section 7) =="
$PY -m pytest -q

echo "== M1 hard gate: E1 (must PASS before benchmarks) =="
$PY -m opm.run experiment=E1

echo "== M2: E2 =="; $PY -m opm.run experiment=E2
echo "== M3: E3 =="; $PY -m opm.run experiment=E3
echo "== M4: E4 (RHC), E5 =="; $PY -m opm.run experiment=E4; $PY -m opm.run experiment=E5
echo "== M5: E6, E7, E8 =="; $PY -m opm.run experiment=E6; $PY -m opm.run experiment=E7; $PY -m opm.run experiment=E8; $PY -m opm.run experiment=E9

echo "== consolidate =="
$PY scripts/consolidate.py
$PY scripts/make_latex.py || true
echo "Done. See results/SUMMARY.md"
