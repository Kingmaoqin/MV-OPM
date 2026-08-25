"""Regression tests for defects found by the independent Round-2 re-audit."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import opm.data.rhc as rhc_data
from opm.dgp.nonlinear_bridge import ATE, generate as generate_nonlinear
from opm.diagnostics.diagnostics import q_sanity
from opm.data.rhc import PROXY_V, PROXY_W, _proxy_provenance
from opm.estimator.nested_mses import ATE_INTERVAL_SCOPE
from opm.estimator.tau_head import crossfit_cate_predictions
from opm.experiments.mvopm_round2.studies import _nested_rhc_ate_payload
from opm.validation.selector_v2 import screen_candidate, select_mses
from scripts.analyze_round2 import (
    _deduplicate_g2_zero_baseline,
    _regime_mean_risk_winners,
)
import scripts.analyze_round2 as analyze_round2
from opm.provenance import (
    FROZEN_ROUND2_FINAL_MANIFEST_SHA256,
    ProvenanceMismatch,
    _dirty_paths_from_porcelain,
    _paths_content_sha256,
    registered_manifest_commit,
    validate_manifest_rows,
    validate_manifest_output_namespace,
    validate_execution_manifest,
    sha256_file,
    validate_result_checksum,
    validate_result_provenance,
)
from scripts.cluster.aggregate_round2 import aggregate
from scripts.cluster.launch import _terminal
import scripts.cluster.launch as cluster_launch
from scripts.cluster.manifest_round2 import build as build_round2_manifest
from scripts.cluster.worker import _atomic_json_dump
import scripts.cluster.worker as cluster_worker


def test_nonlinear_dgp_uses_repository_wide_empirical_x_ate_target():
    ds = generate_nonlinear(257, seed=91)
    assert np.array_equal(ds.ate_true, np.mean(ds.tau_true, axis=0))
    assert np.array_equal(ds.meta["ate_population"], np.array([ATE]))
    assert ds.meta["ate_target"] == "empirical_X_mean_tau"


def test_q_sanity_reports_hard_clipping_fraction():
    q = np.array([[0.0, 1.0], [2.0, 50.0], [3.0, 50.0], [4.0, 2.0]])
    out = q_sanity(q, np.array([0, 1, 0, 1]), K=2)
    assert out[0]["fraction_at_lower_bound"] == 0.25
    assert out[1]["fraction_at_upper_bound"] == 0.5


def test_rhc_primary_payload_uses_nested_estimator_and_labels_interval():
    nested = {
        "status": "ok",
        "ate_interval_scope": ATE_INTERVAL_SCOPE,
        "ate_by_selector": {
            "MSES": {1: {"ate": -1.2, "se": 0.2, "ci_low": -1.592, "ci_high": -0.808}},
        },
    }
    payload = _nested_rhc_ate_payload(nested)
    assert payload["ate"] == [-1.2]
    assert payload["ate_source"] == "nested_outer_oof_adaptive_mses"
    assert payload["ate_interval_scope"] == ATE_INTERVAL_SCOPE


def test_regime_oracle_uses_mean_risk_not_noisy_task_winner_counts():
    rows = []
    # b wins two of three noisy tasks, while a has lower mean risk because its first win is large.
    for seed, a_err, b_err in [(1, 0.0, 1.0), (2, 0.3, 0.2), (3, 0.3, 0.2)]:
        rows.extend([
            {"study": "R", "config_id": "cfg", "scientific_seed": seed,
             "candidate": "a", "ate_err": a_err},
            {"study": "R", "config_id": "cfg", "scientific_seed": seed,
             "candidate": "b", "ate_err": b_err},
        ])
    winner = _regime_mean_risk_winners(pd.DataFrame(rows)).iloc[0]
    assert winner.mean_risk_oracle == "a"
    assert winner.mean_risk_oracle_error == pytest.approx(0.2)


def _provenance_pair(path):
    row = {
        "study": "B2",
        "dispatch": "selection_v2",
        "scientific_seed": 101,
        "bootstrap_seed": 201,
        "split_seed": 301,
        "bank_seed": 401,
        "config_id": "cfg",
        "out_path": str(path),
        "source_commit": "a" * 40,
        "require_clean_source": True,
    }
    prov = {
        "experiment": "B2",
        "dispatch": "selection_v2",
        "scientific_seed": 101,
        "bootstrap_seed": 201,
        "split_seed": 301,
        "bank_seed": 401,
        "array_task_id": 0,
        "git_commit_full": "a" * 40,
        "git_dirty": False,
        "manifest_sha256": "manifest-hash",
        "config": row,
        "status": "ok",
        "exit_code": 0,
        "result": {"study": "B2", "seed": 101},
    }
    return row, prov


def test_provenance_rejects_swapped_seed_and_config(tmp_path):
    row, prov = _provenance_pair(tmp_path / "result.json")
    validate_result_provenance(
        prov, row, array_task_id=0, manifest_sha256="manifest-hash",
    )
    bad = json.loads(json.dumps(prov))
    bad["scientific_seed"] = 999
    with pytest.raises(ProvenanceMismatch, match="scientific_seed"):
        validate_result_provenance(
            bad, row, array_task_id=0, manifest_sha256="manifest-hash",
        )


def test_frozen_round2_final_builder_and_aggregate_outputs_are_immutable(tmp_path):
    with pytest.raises(RuntimeError, match="Round-2 final is frozen"):
        build_round2_manifest("final")
    frozen_manifest = (
        "results/mvopm_round2/manifest_final.jsonl"
    )
    with pytest.raises(RuntimeError, match="registered frozen Round-2 manifest"):
        aggregate(frozen_manifest)

    copied_manifest = tmp_path / "renamed_manifest.jsonl"
    shutil.copyfile(frozen_manifest, copied_manifest)
    assert sha256_file(str(copied_manifest)) == FROZEN_ROUND2_FINAL_MANIFEST_SHA256
    with pytest.raises(RuntimeError, match="registered frozen Round-2 manifest"):
        aggregate(str(copied_manifest))
    frozen_nested_output = "results/mvopm_round2/final/audit_reconstruction"
    with pytest.raises(RuntimeError, match="intersects a frozen Round-2 root"):
        aggregate(str(copied_manifest), output_root=frozen_nested_output)


def test_worker_unconditionally_rejects_copied_frozen_manifest(tmp_path, monkeypatch):
    copied_manifest = tmp_path / "renamed_manifest.jsonl"
    shutil.copyfile("results/mvopm_round2/manifest_final.jsonl", copied_manifest)
    monkeypatch.setattr(sys, "argv", ["worker.py", str(copied_manifest), "0"])
    monkeypatch.setattr(
        cluster_worker,
        "run_row",
        lambda row: pytest.fail("frozen row must never execute"),
    )
    with pytest.raises(RuntimeError, match="refusing all execution"):
        cluster_worker.main()


def test_launcher_and_analysis_reject_frozen_content_and_output_overlap(tmp_path, monkeypatch):
    copied_manifest = tmp_path / "renamed_manifest.jsonl"
    shutil.copyfile("results/mvopm_round2/manifest_final.jsonl", copied_manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        ["launch.py", str(copied_manifest), "--workers", "1", "--max", "1"],
    )
    with pytest.raises(RuntimeError, match="refusing to launch registered frozen manifest"):
        cluster_launch.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_round2.py",
            "--phase",
            "final",
            "--output-root",
            "results/mvopm_round2/final/audit_reconstruction",
        ],
    )
    with pytest.raises(RuntimeError, match="intersects the frozen Round-2 final root"):
        analyze_round2.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_round2.py",
            "--phase",
            "development",
            "--output-root",
            "results/mvopm_round2/final/analysis",
        ],
    )
    with pytest.raises(RuntimeError, match="intersects the frozen Round-2 final root"):
        analyze_round2.main()


def test_modified_manifest_cannot_target_frozen_final_namespace(tmp_path, monkeypatch):
    protected = cluster_worker.FROZEN_FINAL_ROOT
    row = {
        "study": "B2",
        "dispatch": "selection_v2",
        "scientific_seed": 991,
        "bootstrap_seed": 992,
        "split_seed": 993,
        "bank_seed": 994,
        "config_id": "p1_namespace_probe",
        "out_path": str(
            Path(protected)
            / "P1_SENTINEL_MUST_NOT_EXIST"
            / "result.json"
        ),
        "source_commit": "c" * 40,
        "require_clean_source": True,
    }
    manifest = tmp_path / "modified_unregistered.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert registered_manifest_commit(sha256_file(str(manifest))) is None
    with pytest.raises(ProvenanceMismatch, match="protected output namespace"):
        validate_manifest_output_namespace([row], protected_roots=[protected])

    monkeypatch.setattr(sys, "argv", ["worker.py", str(manifest), "0"])
    monkeypatch.setattr(
        cluster_worker.os,
        "makedirs",
        lambda *args, **kwargs: pytest.fail("worker must reject before creating directories"),
    )
    monkeypatch.setattr(
        cluster_worker,
        "run_row",
        lambda row: pytest.fail("worker must reject before executing the row"),
    )
    with pytest.raises(ProvenanceMismatch, match="protected output namespace"):
        cluster_worker.main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["launch.py", str(manifest), "--workers", "1", "--max", "1"],
    )
    with pytest.raises(ProvenanceMismatch, match="protected output namespace"):
        cluster_launch.main()


def test_execution_manifest_rejects_all_missing_source_policy(tmp_path):
    row, _ = _provenance_pair(tmp_path / "result.json")
    row.pop("source_commit")
    row.pop("require_clean_source")
    validate_manifest_rows([row])  # registered legacy aggregation remains readable
    with pytest.raises(ProvenanceMismatch, match="full 40-character"):
        validate_execution_manifest([row])


def test_dirty_path_content_digest_changes_when_same_path_changes(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    before = _paths_content_sha256(str(tmp_path), ["source.py"])
    source.write_text("value = 2\n", encoding="utf-8")
    after = _paths_content_sha256(str(tmp_path), ["source.py"])
    assert before != after


def test_legacy_checkpoint_match_is_exact_not_prefix(tmp_path):
    row, prov = _provenance_pair(tmp_path / "result.json")
    row.pop("source_commit")
    row.pop("require_clean_source")
    prov["config"] = row
    prov.pop("git_commit_full")
    prov["git_commit"] = "aa983a2_wrong_suffix"
    with pytest.raises(ProvenanceMismatch, match="legacy checkpoint"):
        validate_result_provenance(
            prov,
            row,
            array_task_id=0,
            manifest_sha256=FROZEN_ROUND2_FINAL_MANIFEST_SHA256,
        )


def test_g2_zero_baseline_is_shared_in_pooled_inputs():
    tasks = pd.DataFrame({
        "study": ["G2"] * 5,
        "proxy_mod_kind": ["additive", "missing", "shift", "heavy_tail", "missing"],
        "proxy_mod_level": [0.0, 0.0, 0.0, 0.0, 0.1],
    })
    deduplicated = _deduplicate_g2_zero_baseline(tasks)
    assert len(deduplicated) == 2
    assert deduplicated.proxy_mod_level.eq(0.0).sum() == 1


def test_legacy_checkpoint_and_malformed_errors_fail_closed(tmp_path):
    row, prov = _provenance_pair(tmp_path / "result.json")
    legacy_row = dict(row)
    legacy_row.pop("source_commit")
    legacy_row.pop("require_clean_source")
    legacy = json.loads(json.dumps(prov))
    legacy["config"] = legacy_row
    legacy.pop("git_commit_full")
    legacy["git_commit"] = "deadbee"
    with pytest.raises(ProvenanceMismatch, match="legacy checkpoint"):
        validate_result_provenance(
            legacy,
            legacy_row,
            array_task_id=0,
            manifest_sha256=(
                "53acb6c62b0a24f1ff7dc236244f9ca6c287e767c8ed52c1ccc596cf5162820d"
            ),
        )

    malformed = json.loads(json.dumps(prov))
    malformed.update({"status": "error", "exit_code": 0})
    malformed.pop("result")
    with pytest.raises(ProvenanceMismatch, match="nonzero integer exit_code"):
        validate_result_provenance(
            malformed, row, array_task_id=0, manifest_sha256="manifest-hash",
        )


def test_manifest_rejects_mixed_missing_source_registration(tmp_path):
    row, _ = _provenance_pair(tmp_path / "one.json")
    second = dict(row)
    second.update({
        "out_path": str(tmp_path / "two.json"),
        "scientific_seed": 102,
        "bootstrap_seed": 202,
        "bank_seed": 402,
    })
    second.pop("source_commit")
    with pytest.raises(ProvenanceMismatch, match="mixes registered and missing"):
        validate_manifest_rows([row, second])


def test_git_porcelain_parser_keeps_manifest_exception_narrow():
    status = " M results/mvopm_round2/manifest_final.jsonl\n?? opm/untracked_fix.py\n"
    assert _dirty_paths_from_porcelain(status) == [
        "results/mvopm_round2/manifest_final.jsonl",
        "opm/untracked_fix.py",
    ]


def test_atomic_result_write_and_resume_checksum(tmp_path):
    path = tmp_path / "result.json"
    row, prov = _provenance_pair(path)
    digest = _atomic_json_dump(str(path), prov)
    assert digest == sha256_file(str(path))
    validate_result_checksum(str(path), required=True)
    assert _terminal(
        row, array_task_id=0, manifest_sha256="manifest-hash",
    )
    path.write_text("{}", encoding="utf-8")
    assert not _terminal(
        row, array_task_id=0, manifest_sha256="manifest-hash",
    )


def test_rhc_proxy_mapping_cannot_self_authorize(tmp_path):
    data_path = tmp_path / "rhc.csv"
    data_path.write_text("placeholder\n", encoding="utf-8")
    assert not _proxy_provenance(str(data_path))["verified"]
    sidecar = {
        "dataset_sha256": sha256_file(str(data_path)),
        "proxy_mapping": {"V": PROXY_V, "W": PROXY_W},
        "paper_citation": "Cui et al. (2023)",
        "paper_doi": "10.example/review-required",
        "paper_page_or_table": "p. 123, Table 4",
        "paper_variable_definition": "placeholder that must receive external review",
        "preparation_script_sha256": "b" * 64,
        "reviewed_by": "independent-source-review-required",
        "reviewed_on": "2026-08-24",
    }
    (tmp_path / "rhc.csv.provenance.json").write_text(
        json.dumps(sidecar), encoding="utf-8",
    )
    provenance = _proxy_provenance(str(data_path))
    assert provenance["schema_complete"]
    assert not provenance["externally_approved"]
    assert not provenance["verified"]
    assert provenance["sidecar_sha256"] not in rhc_data.APPROVED_RHC_SIDECAR_SHA256


def test_stage2_cate_evaluation_is_cross_fitted_and_complete():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(60, 3))
    phi = np.column_stack([rng.normal(size=60), rng.normal(size=60)])
    pred = crossfit_cate_predictions(
        X,
        phi,
        K=2,
        n_folds=3,
        seed=44,
        head_kwargs={"max_epochs": 3, "patience": 2, "hidden": 8, "depth": 1},
    )
    assert pred.shape == (60, 1)
    assert np.all(np.isfinite(pred))


def test_holm_and_max_variance_sensitivities_are_explicit():
    bonferroni = screen_candidate(
        [0.01, 0.02, 0.9], [0.01, 0.02, 0.9], alpha=0.05,
    )
    holm = screen_candidate(
        [0.01, 0.02, 0.9], [0.01, 0.02, 0.9], alpha=0.05, adjustment="holm",
    )
    assert bonferroni.failed_arms == (0,)
    assert holm.failed_arms == (0, 1)
    tests = {
        "a": {"p_h": [0.9, 0.9], "p_q": [0.9, 0.9]},
        "b": {"p_h": [0.9, 0.9], "p_q": [0.9, 0.9]},
    }
    uncertainty = {
        "a": {"variance_mean": 1.0, "variance_max": 10.0},
        "b": {"variance_mean": 2.0, "variance_max": 3.0},
    }
    assert select_mses(tests, uncertainty)["selected"] == "a"
    sensitivity = select_mses(tests, uncertainty, variance_key="variance_max")
    assert sensitivity["selected"] == "b"
    assert sensitivity["variance_key"] == "variance_max"
