"""Regression tests for the repaired-v1 6,700-row release and runtime gates."""
from __future__ import annotations

from collections import Counter
import json
import os
import sys

import pytest

from opm.provenance import ProvenanceMismatch, sha256_file, validate_result_provenance
from opm.release_gate import (
    EXPECTED_COUNTS,
    EXPECTED_DECISION_SHA256,
    HAMD_PATH,
    HAMD_SHA256,
    validate_independent_review_text,
    RELEASE_RELATIVE_PATH,
    validate_repaired_v1_contract,
)
from scripts import launch_mvopm_repaired_v1_final as final_controller
from scripts.cluster import launch as cluster_launch
from scripts.cluster import manifest_mvopm_repaired_v1 as manifest_builder
from scripts.cluster import worker as cluster_worker


@pytest.fixture(scope="module")
def final_rows():
    seed_payload = json.load(open(manifest_builder.SEED_PATH, encoding="utf-8"))
    return manifest_builder.build_rows(
        source_commit="a" * 40,
        seed_payload=seed_payload,
        seed_registry_sha256=sha256_file(manifest_builder.SEED_PATH),
        convergence_decision_sha256=EXPECTED_DECISION_SHA256,
        batch_sensitivity_sha256="b" * 64,
    )


def test_final_builder_exact_allocation_and_streams(final_rows):
    contract = validate_repaired_v1_contract(final_rows, root=manifest_builder.ROOT)
    assert len(final_rows) == 6700
    assert Counter(row["study"] for row in final_rows) == Counter(EXPECTED_COUNTS)
    assert contract["scientific_source_commit"] == "a" * 40
    assert not {row["study"] for row in final_rows} & {"J2"}
    assert not {row["dispatch"] for row in final_rows} & {"rhc", "rhc_v2"}

    derived = [
        value
        for row in final_rows
        for value in (row["bootstrap_seed"], row["bank_seed"])
    ]
    scientific = {row["scientific_seed"] for row in final_rows}
    assert len(derived) == len(set(derived)) == 13400
    assert not set(derived) & scientific
    assert all(row["split_seed"] == row["scientific_seed"] for row in final_rows)
    assert all(row["cfg"]["bridge"]["max_epochs"] == 300 for row in final_rows)
    assert all(row["cfg"]["bridge"]["batch_size"] == 1024 for row in final_rows)
    assert all(row["cfg"]["device"] == "cuda:0" for row in final_rows)

    per_seed_configs = {
        "A2": 25,
        "B2": 1,
        "C2": 1,
        "D2": 1,
        "E2": 1,
        "R": 20,
        "F2": 4,
        "G2": 13,
        "H2": 8,
        "I2": 3,
    }
    for study, multiplier in per_seed_configs.items():
        counts = Counter(
            row["scientific_seed"] for row in final_rows if row["study"] == study
        )
        assert set(counts.values()) == {multiplier}
    g2 = [row for row in final_rows if row["study"] == "G2"]
    assert len({row["config_id"] for row in g2}) == 13
    assert len({row["config_id"] for row in g2 if row["proxy_mod"]["level"] == 0.0}) == 1
    e2 = [row for row in final_rows if row["study"] == "E2"]
    assert all(row["hamd_input_path"] == HAMD_PATH for row in e2)
    assert all(row["hamd_input_sha256"] == HAMD_SHA256 for row in e2)


def test_final_contract_rejects_namespace_config_and_rhc_mutations(final_rows):
    row = final_rows[0]
    original_path = row["out_path"]
    try:
        row["out_path"] = os.path.join(
            manifest_builder.ROOT,
            "results",
            "mvopm_round2",
            "sentinel",
            "result.json",
        )
        with pytest.raises(ProvenanceMismatch, match="outside repaired-v1 final root"):
            validate_repaired_v1_contract(final_rows, root=manifest_builder.ROOT)
    finally:
        row["out_path"] = original_path

    original_batch = row["cfg"]["bridge"]["batch_size"]
    try:
        row["cfg"]["bridge"]["batch_size"] = 512
        with pytest.raises(ProvenanceMismatch, match="configuration mismatch"):
            validate_repaired_v1_contract(final_rows, root=manifest_builder.ROOT)
    finally:
        row["cfg"]["bridge"]["batch_size"] = original_batch

    original_dispatch = row["dispatch"]
    try:
        row["dispatch"] = "rhc_v2"
        with pytest.raises(ProvenanceMismatch, match="J2/RHC"):
            validate_repaired_v1_contract(final_rows, root=manifest_builder.ROOT)
    finally:
        row["dispatch"] = original_dispatch


def test_release_result_provenance_binds_scientific_and_authorization_commits(tmp_path):
    row = {
        "study": "B2",
        "dispatch": "selection_v2",
        "scientific_seed": 101,
        "bootstrap_seed": 201,
        "split_seed": 101,
        "bank_seed": 301,
        "config_id": "S1_n4000",
        "out_path": str(tmp_path / "result.json"),
        "source_commit": "a" * 40,
        "require_clean_source": True,
        "release_authorization_path": RELEASE_RELATIVE_PATH,
    }
    prov = {
        "experiment": "B2",
        "dispatch": "selection_v2",
        "scientific_seed": 101,
        "bootstrap_seed": 201,
        "split_seed": 101,
        "bank_seed": 301,
        "array_task_id": 0,
        "scientific_source_commit": "a" * 40,
        "authorization_commit": "c" * 40,
        "git_commit_full": "c" * 40,
        "git_dirty": False,
        "git_worktree_dirty": False,
        "release_authorization_sha256": "d" * 64,
        "manifest_sha256": "e" * 64,
        "config": row,
        "status": "ok",
        "exit_code": 0,
        "result": {"study": "B2", "seed": 101},
    }
    validate_result_provenance(
        prov,
        row,
        array_task_id=0,
        manifest_sha256="e" * 64,
        authorization_commit="c" * 40,
        release_authorization_sha256="d" * 64,
    )
    with pytest.raises(ProvenanceMismatch, match="expected authorization_commit"):
        validate_result_provenance(
            prov,
            row,
            array_task_id=0,
            manifest_sha256="e" * 64,
        )
    wrong_auth = dict(prov, authorization_commit="f" * 40)
    with pytest.raises(ProvenanceMismatch, match="recorded authorization_commit"):
        validate_result_provenance(
            wrong_auth,
            row,
            array_task_id=0,
            manifest_sha256="e" * 64,
            authorization_commit="c" * 40,
            release_authorization_sha256="d" * 64,
        )
    wrong_release = dict(prov, release_authorization_sha256="f" * 64)
    with pytest.raises(ProvenanceMismatch, match="release_authorization_sha256"):
        validate_result_provenance(
            wrong_release,
            row,
            array_task_id=0,
            manifest_sha256="e" * 64,
            authorization_commit="c" * 40,
            release_authorization_sha256="d" * 64,
        )
    missing_identity = dict(prov, result={"seed": 101})
    with pytest.raises(ProvenanceMismatch, match="missing study"):
        validate_result_provenance(
            missing_identity,
            row,
            array_task_id=0,
            manifest_sha256="e" * 64,
            authorization_commit="c" * 40,
            release_authorization_sha256="d" * 64,
        )
    prov["git_commit_full"] = "f" * 40
    with pytest.raises(ProvenanceMismatch, match="authorization commit"):
        validate_result_provenance(
            prov,
            row,
            array_task_id=0,
            manifest_sha256="e" * 64,
            authorization_commit="c" * 40,
            release_authorization_sha256="d" * 64,
        )


def test_independent_review_requires_unique_candidate_bound_pass_markers():
    manifest_sha = "e" * 64
    source_commit = "a" * 40
    review = "\n".join((
        "# Independent final review",
        "Verdict: PASS_FOR_FULL_EXECUTION",
        "P0: 0",
        "P1: 0",
        "P2: 0",
        f"Reviewed manifest SHA-256: {manifest_sha}",
        f"Reviewed scientific source commit: {source_commit}",
    ))
    validate_independent_review_text(
        review,
        manifest_sha256=manifest_sha,
        scientific_source_commit=source_commit,
    )
    for invalid in (
        review.replace("P1: 0", "P1: 1"),
        review + "\nVerdict: PASS_FOR_FULL_EXECUTION",
        review.replace(manifest_sha, "f" * 64),
        review.replace(source_commit, "b" * 40),
    ):
        with pytest.raises(ProvenanceMismatch, match="marker"):
            validate_independent_review_text(
                invalid,
                manifest_sha256=manifest_sha,
                scientific_source_commit=source_commit,
            )


def test_batch_compact_evidence_hashes_actual_files(tmp_path):
    evidence_path = tmp_path / "batch_sensitivity.csv"
    summary_path = tmp_path / "summary.csv"
    evidence_path.write_text("scenario,value\nS1,1\n", encoding="utf-8")
    summary_path.write_text("scenario,median\nS1,1\n", encoding="utf-8")
    batch = {
        "batch_sensitivity_sha256": sha256_file(str(evidence_path)),
        "summary_sha256": sha256_file(str(summary_path)),
    }
    assert manifest_builder._validate_batch_output_hashes(
        batch,
        evidence_path=str(evidence_path),
        summary_path=str(summary_path),
    ) == (batch["batch_sensitivity_sha256"], batch["summary_sha256"])
    evidence_path.write_text("scenario,value\nS1,2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="CSV differs"):
        manifest_builder._validate_batch_output_hashes(
            batch,
            evidence_path=str(evidence_path),
            summary_path=str(summary_path),
        )


def test_worker_hamd_after_hash_runs_when_scientific_code_raises(tmp_path, monkeypatch):
    hamd = tmp_path / "hamd.csv"
    hamd.write_text("stable input\n", encoding="utf-8")
    expected_sha = sha256_file(str(hamd))
    monkeypatch.setattr(cluster_worker, "HAMD_PATH", str(hamd))
    monkeypatch.setattr(cluster_worker, "HAMD_SHA256", expected_sha)

    def corrupt_then_raise(row):
        hamd.write_text("mutated input\n", encoding="utf-8")
        raise ValueError("scientific failure")

    monkeypatch.setattr(cluster_worker, "run_row", corrupt_then_raise)
    with pytest.raises(RuntimeError, match="after E2 execution"):
        cluster_worker._run_row_with_hamd_guard({"study": "E2"})


@pytest.mark.parametrize("index", [-1, 1])
def test_worker_rejects_out_of_range_before_creating_output(index, tmp_path, monkeypatch):
    row = {
        "study": "probe",
        "dispatch": "selection_v2",
        "scientific_seed": 101,
        "bootstrap_seed": 201,
        "split_seed": 101,
        "bank_seed": 301,
        "config_id": "probe",
        "out_path": str(tmp_path / "must_not_exist" / "result.json"),
        "source_commit": "a" * 40,
        "require_clean_source": True,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["worker.py", str(manifest), str(index)])
    monkeypatch.setattr(
        cluster_worker.os,
        "makedirs",
        lambda *args, **kwargs: pytest.fail("out-of-range row must not create output"),
    )
    with pytest.raises(IndexError, match="outside"):
        cluster_worker.main()


def test_launcher_parameter_guards_precede_manifest_read(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["launch.py", "/does/not/exist", "--workers", "0", "--threads", "1"],
    )
    with pytest.raises(ValueError, match="positive"):
        cluster_launch.main()


def test_program_controller_has_no_partial_execution_surface(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(sys, "argv", ["final.py", "--workers", "2", "--threads", "1"])
    with pytest.raises(RuntimeError, match="exactly one physical GPU"):
        final_controller.main()

    captured = {}

    class Completed:
        returncode = 0

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Completed()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(final_controller.subprocess, "run", fake_run)
    assert final_controller.main() == 0
    command = captured["command"]
    assert "--resume" in command and "--rerun-infrastructure" in command
    assert "--studies" not in command and "--max" not in command
    assert final_controller.MANIFEST in command
