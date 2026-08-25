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

from scripts import run_mvopm_repaired_v1_batch_sensitivity as batch_sensitivity

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


def test_final_seed_registry_is_disjoint_and_complete():
    assert os.path.exists(seed_generator.FINAL_PATH)
    final = json.load(open(seed_generator.FINAL_PATH, encoding="utf-8"))
    development = json.load(open(seed_generator.DEVELOPMENT_PATH, encoding="utf-8"))
    old = json.load(open(seed_generator.OLD_REGISTRY, encoding="utf-8"))
    assert sha256_file(seed_generator.DEVELOPMENT_PATH) == (
        seed_generator.EXPECTED_DEVELOPMENT_SEED_SHA256
    )
    assert final["program_id"] == seed_generator.PROGRAM_ID
    assert final["phase"] == "confirmatory_final_unexposed_before_freeze"
    assert final["selected_kernel_budget"] == 300
    assert {
        study: len(values) for study, values in final["scientific_seeds"].items()
    } == seed_generator.FINAL_COUNTS
    final_values = list(seed_generator._flatten(final["scientific_seeds"]))
    assert len(final_values) == len(set(final_values)) == sum(seed_generator.FINAL_COUNTS.values())
    assert all(seed_generator.LOW <= value < seed_generator.HIGH for value in final_values)
    assert not set(final_values) & set(seed_generator._flatten(development))
    assert not set(final_values) & set(seed_generator._flatten(old))


def test_batch_sensitivity_validator_and_parameter_guards():
    row = copy.deepcopy(_valid_rows()[3])
    row["training_batch_size_cap"] = 256
    row["h_prediction_change"] = float("nan")
    row["q_prediction_change"] = float("nan")
    batch_sensitivity._validate_row(row, scenario="S1", seed=123, batch_size=256)
    bad = copy.deepcopy(row)
    bad["D_total_mmr"] = float("inf")
    with pytest.raises(RuntimeError, match="nonfinite"):
        batch_sensitivity._validate_row(bad, scenario="S1", seed=123, batch_size=256)

    oracle = copy.deepcopy(row)
    oracle["oracle_ate_error"] = 0.0
    with pytest.raises(RuntimeError, match="unexpected fields"):
        batch_sensitivity._validate_row(oracle, scenario="S1", seed=123, batch_size=256)

    source = {
        "commit": "a" * 40,
        "dirty": False,
        "status_sha256": "b" * 64,
        "worktree_content_sha256": "c" * 64,
    }
    expected = (
        "S1",
        123,
        256,
        "cuda:0",
        source,
        batch_sensitivity.EXPECTED_DEVELOPMENT_SEED_SHA256,
    )
    record = {
        "program_id": batch_sensitivity.PROGRAM_ID,
        "status": "ok",
        "scenario": "S1",
        "seed": 123,
        "batch_size": 256,
        "n": 1500,
        "device": "cuda:0",
        "source_state": source,
        "seed_registry_sha256": batch_sensitivity.EXPECTED_DEVELOPMENT_SEED_SHA256,
        "hamd_input_path": convergence.HAMD_PATH,
        "hamd_input_sha256": convergence.HAMD_SHA256,
        "convergence_decision_sha256": batch_sensitivity.EXPECTED_DECISION_SHA256,
        "runtime_s": 1.0,
        "environment": {"python": "test"},
        "row": row,
    }
    batch_sensitivity._validate_shard(record, expected)
    record["oracle_ate_error"] = 0.0
    with pytest.raises(RuntimeError, match="unexpected fields"):
        batch_sensitivity._validate_shard(record, expected)

    start = dict(source)
    end = dict(source, status_sha256="d" * 64, worktree_content_sha256="e" * 64)
    assert batch_sensitivity._stable_source_identity(start) == (
        batch_sensitivity._stable_source_identity(end)
    )


def test_batch_sensitivity_rejects_extra_raw_shard(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_sensitivity, "OUT", str(tmp_path))
    task = ("S1", 123, 256, "cuda:0", {}, "a" * 64)
    directory = tmp_path / "S1"
    directory.mkdir()
    path = directory / "seed123_batch256.json"
    path.write_text("{}", encoding="utf-8")
    (directory / "seed123_batch256.json.sha256").write_text("x", encoding="ascii")
    digest, count = batch_sensitivity._validate_shard_inventory([task])
    assert len(digest) == 64 and count == 2
    (directory / "seed999_batch256.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory"):
        batch_sensitivity._validate_shard_inventory([task])
    from opm.experiments.mvopm_round2.convergence_audit import audit_one

    with pytest.raises(ValueError, match="budgets"):
        audit_one("S1", 123, budgets=())
    with pytest.raises(ValueError, match="batch_size"):
        audit_one("S1", 123, budgets=(300,), batch_size=0)


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
