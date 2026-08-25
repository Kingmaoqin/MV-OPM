"""Fail-closed checks for the repaired-v1 development and final release gates."""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys

import pytest

from opm.provenance import sha256_file
from scripts import generate_mvopm_repaired_v1_seeds as seed_generator
from scripts import run_mvopm_repaired_v1_convergence as convergence


def _valid_rows(scenario="S1", seed=123):
    rows = []
    for budget in convergence.BUDGETS:
        first = budget == convergence.BUDGETS[0]
        rows.append({
            "scenario": scenario,
            "seed": seed,
            "budget": budget,
            "D_total_rff_l2": 1.0,
            "D_total_mmr": 1.0,
            "h_prediction_change": float("nan") if first else 0.01,
            "q_prediction_change": float("nan") if first else 0.01,
            "optimizer_best_resid": 1.0,
            "optimizer_min_training_objective": 1.0,
            "optimizer_final_training_objective": 1.0,
            "optimizer_updates": 1,
            "training_objective_scope": "minibatch_vstat_with_diagonal_regularization",
            "training_batch_size_cap": 1024,
            "parameter_l2": 1.0,
            "h_prediction_min": -1.0,
            "h_prediction_max": 1.0,
            "h_prediction_rms": 1.0,
            "q_prediction_min": 0.1,
            "q_prediction_max": 2.0,
            "q_prediction_rms": 1.0,
        })
    return rows


def _valid_shard():
    source = {
        "commit": "a" * 40,
        "dirty": False,
        "status_sha256": "b" * 64,
        "worktree_content_sha256": "c" * 64,
    }
    expected = (
        "S1", 123, 1500, "cuda:0", source, "d" * 64, convergence.HAMD_SHA256,
    )
    record = {
        "program_id": convergence.PROGRAM_ID,
        "status": "ok",
        "scenario": "S1",
        "seed": 123,
        "n": 1500,
        "device": "cuda:0",
        "source_state": source,
        "seed_registry_sha256": "d" * 64,
        "hamd_input_path": convergence.HAMD_PATH,
        "hamd_input_sha256": convergence.HAMD_SHA256,
        "runtime_s": 1.0,
        "environment": {"python": "test"},
        "rows": _valid_rows(),
    }
    return record, expected


def test_preregistered_hamd_input_is_exact():
    assert os.path.exists(convergence.HAMD_PATH)
    assert sha256_file(convergence.HAMD_PATH) == convergence.HAMD_SHA256


def test_shard_validator_accepts_complete_record_and_rejects_identity_swap():
    record, expected = _valid_shard()
    convergence._validate_shard(record, expected)
    swapped = copy.deepcopy(record)
    swapped["seed"] = 999
    with pytest.raises(RuntimeError, match="seed mismatch"):
        convergence._validate_shard(swapped, expected)


def test_shard_validator_rejects_missing_duplicate_or_nonfinite_budget_rows():
    record, expected = _valid_shard()
    missing = copy.deepcopy(record)
    missing["rows"].pop()
    with pytest.raises(RuntimeError, match="exactly four"):
        convergence._validate_shard(missing, expected)
    duplicate = copy.deepcopy(record)
    duplicate["rows"][3]["budget"] = 180
    with pytest.raises(RuntimeError, match="duplicate budget"):
        convergence._validate_shard(duplicate, expected)
    nonfinite = copy.deepcopy(record)
    nonfinite["rows"][2]["D_total_rff_l2"] = float("inf")
    with pytest.raises(RuntimeError, match="nonfinite"):
        convergence._validate_shard(nonfinite, expected)


def test_shard_checksum_is_required_and_detects_change(tmp_path):
    record, expected = _valid_shard()
    path = tmp_path / "seed123.json"
    convergence._atomic_json(str(path), record, checksum=True)
    convergence._validate_shard(record, expected, path=str(path))
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        convergence._validate_checksum(str(path))


def test_final_seed_generation_is_blocked_before_committed_decision():
    assert not os.path.exists(seed_generator.FINAL_PATH)
    with pytest.raises(RuntimeError, match="clean committed worktree|gate file is missing"):
        seed_generator._require_final_seed_gate()


@pytest.mark.parametrize("checks", [0, -1])
def test_gpu_watcher_rejects_nonpositive_checks(checks, tmp_path):
    command = [
        sys.executable,
        "scripts/run_when_gpu_idle.py",
        "--gpu", "1",
        "--checks", str(checks),
        "--interval", "1",
        "--lock", str(tmp_path / "queue.lock"),
        "--", sys.executable, "-c", "raise SystemExit(99)",
    ]
    result = subprocess.run(command, cwd=convergence.ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "checks must be at least one" in result.stderr
