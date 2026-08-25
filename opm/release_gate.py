"""Fail-closed release authorization for the repaired-v1 confirmatory namespace."""
from __future__ import annotations

from collections import Counter
import json
import os
import re
import subprocess

from .provenance import ProvenanceMismatch, git_state, paths_overlap, sha256_file

PROGRAM_ID = "mvopm_repaired_v1"
PROTOCOL_VERSION = "0.2"
EXPECTED_COUNTS = {
    "A2": 5000,
    "B2": 50,
    "C2": 50,
    "D2": 50,
    "E2": 50,
    "R": 600,
    "F2": 120,
    "G2": 390,
    "H2": 240,
    "I2": 150,
}
EXPECTED_DECISION_SHA256 = "b1c044e93893a0d79fa13cc7e79d1f3318be76206b00c2d4a1b3d09ab742abc9"
HAMD_PATH = "/home/xqin5/DpressionTreatmenteffect/data-merged-def-1.csv"
HAMD_SHA256 = "b40d7c17e687170c2c17c3eb892fd90a73a51e3b1e96864430ffdc2ce7264609"
REVIEW_RELATIVE_PATH = "docs/mvopm_repaired_v1/reviews/FINAL_PRELAUNCH_REVIEW.md"
PREREG_RELATIVE_PATH = "docs/mvopm_repaired_v1/PREREG.md"
MANIFEST_RELATIVE_PATH = "results/mvopm_repaired_v1/manifest_final.jsonl"
RELEASE_RELATIVE_PATH = "results/mvopm_repaired_v1/FINAL_RELEASE.json"
RELEASE_ARTIFACT_ALLOWLIST = (
    MANIFEST_RELATIVE_PATH,
    RELEASE_RELATIVE_PATH,
    PREREG_RELATIVE_PATH,
    REVIEW_RELATIVE_PATH,
)


def _path(root: str, relative: str) -> str:
    return os.path.join(root, *relative.split("/"))


def _head_bytes(root: str, relative: str) -> bytes:
    try:
        return subprocess.check_output(["git", "-C", root, "show", f"HEAD:{relative}"])
    except subprocess.CalledProcessError as exc:
        raise ProvenanceMismatch(f"release artifact is not committed at HEAD: {relative}") from exc


def targets_repaired_v1_final(rows: list[dict], *, root: str) -> bool:
    final_root = _path(root, "results/mvopm_repaired_v1/final")
    return any(
        row.get("program_id") == PROGRAM_ID
        or (
            isinstance(row.get("out_path"), str)
            and paths_overlap(row["out_path"], final_root)
        )
        for row in rows
    )


def validate_independent_review_text(
    review_text: str,
    *,
    manifest_sha256: str,
    scientific_source_commit: str,
) -> None:
    """Require one machine-readable PASS verdict bound to this exact release candidate."""
    expected = {
        "Verdict:": "Verdict: PASS_FOR_FULL_EXECUTION",
        "P0:": "P0: 0",
        "P1:": "P1: 0",
        "P2:": "P2: 0",
        "Reviewed manifest SHA-256:": f"Reviewed manifest SHA-256: {manifest_sha256}",
        "Reviewed scientific source commit:": (
            f"Reviewed scientific source commit: {scientific_source_commit}"
        ),
    }
    lines = [line.strip() for line in review_text.splitlines() if line.strip()]
    for prefix, required in expected.items():
        matching = [line for line in lines if line.startswith(prefix)]
        if matching != [required]:
            raise ProvenanceMismatch(
                f"independent review has missing, duplicate, or invalid marker: {prefix}"
            )


def _require_child_path(path: str, parent: str) -> None:
    path_real = os.path.realpath(path)
    parent_real = os.path.realpath(parent)
    try:
        common = os.path.commonpath([path_real, parent_real])
    except ValueError as exc:
        raise ProvenanceMismatch(f"output path is outside repaired-v1 final root: {path}") from exc
    if common != parent_real or path_real == parent_real:
        raise ProvenanceMismatch(f"output path is outside repaired-v1 final root: {path}")


def validate_repaired_v1_contract(rows: list[dict], *, root: str) -> dict:
    if len(rows) != 6700:
        raise ProvenanceMismatch(f"repaired-v1 final manifest must contain 6700 rows, got {len(rows)}")
    counts = dict(Counter(row.get("study") for row in rows))
    if counts != EXPECTED_COUNTS:
        raise ProvenanceMismatch(f"repaired-v1 per-study counts mismatch: {counts}")
    final_root = _path(root, "results/mvopm_repaired_v1/final")
    frozen_round2 = _path(root, "results/mvopm_round2")
    expected_cfg = {
        "outer_folds": 4,
        "inner_folds": 3,
        "n_rff": 200,
        "n_boot": 999,
        "alpha": 0.05,
        "device": "cuda:0",
        "run_nested": True,
        "bridge": {
            "max_epochs": 300,
            "patience": 45,
            "eval_every": 3,
            "batch_size": 1024,
        },
        "head": {"max_epochs": 300, "patience": 30},
    }
    source_commits = set()
    seed_hashes = set()
    decision_hashes = set()
    batch_hashes = set()
    for row in rows:
        if row.get("program_id") != PROGRAM_ID or row.get("phase") != "final":
            raise ProvenanceMismatch("repaired-v1 manifest program/phase mismatch")
        if row.get("protocol_version") != PROTOCOL_VERSION:
            raise ProvenanceMismatch("repaired-v1 manifest protocol version mismatch")
        if row.get("study") == "J2" or row.get("dispatch") in {"rhc", "rhc_v2"}:
            raise ProvenanceMismatch("J2/RHC is not authorized in repaired-v1")
        if row.get("cfg") != expected_cfg:
            raise ProvenanceMismatch("repaired-v1 final execution configuration mismatch")
        if row.get("split_seed") != row.get("scientific_seed"):
            raise ProvenanceMismatch("split seed must equal the scientific seed")
        if row.get("release_authorization_path") != RELEASE_RELATIVE_PATH:
            raise ProvenanceMismatch("repaired-v1 release authorization path mismatch")
        out_path = row.get("out_path")
        if not isinstance(out_path, str) or not os.path.isabs(out_path):
            raise ProvenanceMismatch("repaired-v1 out_path must be absolute")
        _require_child_path(out_path, final_root)
        if paths_overlap(out_path, frozen_round2):
            raise ProvenanceMismatch("repaired-v1 output intersects frozen Round-2 evidence")
        if row.get("study") == "E2":
            if (
                row.get("hamd_input_path") != HAMD_PATH
                or row.get("hamd_input_sha256") != HAMD_SHA256
            ):
                raise ProvenanceMismatch("E2 HAMD input provenance mismatch")
        source_commits.add(row.get("source_commit"))
        seed_hashes.add(row.get("seed_registry_sha256"))
        decision_hashes.add(row.get("convergence_decision_sha256"))
        batch_hashes.add(row.get("batch_sensitivity_sha256"))
    if len(source_commits) != 1:
        raise ProvenanceMismatch("repaired-v1 manifest mixes scientific source commits")
    scientific_source = next(iter(source_commits))
    if not isinstance(scientific_source, str) or not re.fullmatch(r"[0-9a-f]{40}", scientific_source):
        raise ProvenanceMismatch("repaired-v1 scientific source commit is invalid")
    for label, values in (
        ("seed registry", seed_hashes),
        ("convergence decision", decision_hashes),
        ("batch sensitivity", batch_hashes),
    ):
        if len(values) != 1 or not re.fullmatch(r"[0-9a-f]{64}", str(next(iter(values), ""))):
            raise ProvenanceMismatch(f"repaired-v1 {label} hash is missing or mixed")
    if next(iter(decision_hashes)) != EXPECTED_DECISION_SHA256:
        raise ProvenanceMismatch("repaired-v1 convergence decision hash mismatch")
    return {
        "scientific_source_commit": scientific_source,
        "seed_registry_sha256": next(iter(seed_hashes)),
        "convergence_decision_sha256": next(iter(decision_hashes)),
        "batch_sensitivity_sha256": next(iter(batch_hashes)),
        "per_study": counts,
    }


def validate_repaired_v1_release(
    rows: list[dict],
    *,
    manifest_path: str,
    manifest_sha256: str,
    root: str,
) -> dict:
    contract = validate_repaired_v1_contract(rows, root=root)
    if os.path.realpath(manifest_path) != os.path.realpath(_path(root, MANIFEST_RELATIVE_PATH)):
        raise ProvenanceMismatch("repaired-v1 final manifest must use its canonical path")
    if sha256_file(manifest_path) != manifest_sha256:
        raise ProvenanceMismatch("manifest changed between read and release validation")

    # Reconstruct the complete scientific design from the committed seed registry and frozen
    # development evidence.  Agreement among a hand-edited manifest and release JSON is not
    # sufficient: every scenario, seed allocation, configuration, stream, and output path must
    # equal the dedicated builder's deterministic output.
    from scripts.cluster import manifest_mvopm_repaired_v1 as manifest_builder

    try:
        seed_payload, actual_seed_sha = manifest_builder._load_seed_registry()
        actual_decision_sha, actual_batch_sha = (
            manifest_builder._validate_frozen_development_evidence()
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise ProvenanceMismatch(
            f"repaired-v1 committed scientific evidence is invalid: {exc}"
        ) from exc
    for label, actual, expected in (
        ("seed registry", actual_seed_sha, contract["seed_registry_sha256"]),
        ("convergence decision", actual_decision_sha, contract["convergence_decision_sha256"]),
        ("batch sensitivity", actual_batch_sha, contract["batch_sensitivity_sha256"]),
    ):
        if actual != expected:
            raise ProvenanceMismatch(f"repaired-v1 {label} evidence hash mismatch")
    expected_rows = manifest_builder.build_rows(
        source_commit=contract["scientific_source_commit"],
        seed_payload=seed_payload,
        seed_registry_sha256=actual_seed_sha,
        convergence_decision_sha256=actual_decision_sha,
        batch_sensitivity_sha256=actual_batch_sha,
    )
    if rows != expected_rows:
        raise ProvenanceMismatch(
            "repaired-v1 manifest differs from the deterministic 6,700-row scientific design"
        )
    release_path = _path(root, RELEASE_RELATIVE_PATH)
    if not os.path.exists(release_path):
        raise ProvenanceMismatch("repaired-v1 final release authorization is missing")
    source = git_state(root)
    if source["commit"] == "nogit" or source["dirty"] or source["worktree_dirty"]:
        raise ProvenanceMismatch("repaired-v1 final release requires a clean committed worktree")
    for relative in RELEASE_ARTIFACT_ALLOWLIST:
        local = open(_path(root, relative), "rb").read()
        if local != _head_bytes(root, relative):
            raise ProvenanceMismatch(f"release artifact differs from committed HEAD: {relative}")
    release = json.load(open(release_path, encoding="utf-8"))
    exact = {
        "program_id": PROGRAM_ID,
        "status": "AUTHORIZED_FOR_FULL_EXECUTION",
        "authorization_protocol": "two_commit_direct_parent_release_only_v1",
        "scientific_source_commit": contract["scientific_source_commit"],
        "manifest_sha256": manifest_sha256,
        "manifest_rows": 6700,
        "per_study": EXPECTED_COUNTS,
        "seed_registry_sha256": contract["seed_registry_sha256"],
        "convergence_decision_sha256": contract["convergence_decision_sha256"],
        "batch_sensitivity_sha256": contract["batch_sensitivity_sha256"],
        "selected_kernel_budget": 300,
        "hamd_input_path": HAMD_PATH,
        "hamd_input_sha256": HAMD_SHA256,
        "independent_review_path": REVIEW_RELATIVE_PATH,
        "independent_review_verdict": "PASS_FOR_FULL_EXECUTION",
        "release_artifact_allowlist": list(RELEASE_ARTIFACT_ALLOWLIST),
    }
    for field, value in exact.items():
        if release.get(field) != value:
            raise ProvenanceMismatch(f"final release {field} mismatch")
    optional = {
        "manifest_path",
        "preregistration_sha256",
        "independent_review_sha256",
        "batch_sensitivity_csv_sha256",
        "batch_sensitivity_summary_sha256",
    }
    if set(release) != set(exact) | optional:
        raise ProvenanceMismatch("final release contains missing or unexpected fields")
    if release.get("manifest_path") != MANIFEST_RELATIVE_PATH:
        raise ProvenanceMismatch("final release manifest path mismatch")
    if release.get("preregistration_sha256") != sha256_file(_path(root, PREREG_RELATIVE_PATH)):
        raise ProvenanceMismatch("final release preregistration hash mismatch")
    if release.get("independent_review_sha256") != sha256_file(_path(root, REVIEW_RELATIVE_PATH)):
        raise ProvenanceMismatch("final release independent review hash mismatch")
    review_text = open(_path(root, REVIEW_RELATIVE_PATH), encoding="utf-8").read()
    validate_independent_review_text(
        review_text,
        manifest_sha256=manifest_sha256,
        scientific_source_commit=contract["scientific_source_commit"],
    )
    batch_path = _path(
        root,
        "results/mvopm_repaired_v1/development/batch_sensitivity/BATCH_SENSITIVITY.json",
    )
    batch = json.load(open(batch_path, encoding="utf-8"))
    try:
        batch_csv_sha, batch_summary_sha = manifest_builder._validate_batch_output_hashes(
            batch,
            evidence_path=manifest_builder.BATCH_EVIDENCE_PATH,
            summary_path=manifest_builder.BATCH_SUMMARY_PATH,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ProvenanceMismatch(
            f"batch-sensitivity compact evidence is invalid: {exc}"
        ) from exc
    if release.get("batch_sensitivity_csv_sha256") != batch_csv_sha:
        raise ProvenanceMismatch("final release batch-sensitivity CSV hash mismatch")
    if release.get("batch_sensitivity_summary_sha256") != batch_summary_sha:
        raise ProvenanceMismatch("final release batch-sensitivity summary hash mismatch")
    if not os.path.exists(HAMD_PATH) or sha256_file(HAMD_PATH) != HAMD_SHA256:
        raise ProvenanceMismatch("HAMD input differs from the authorized digest")
    parent_line = subprocess.check_output(
        ["git", "-C", root, "rev-list", "--parents", "-n", "1", "HEAD"], text=True,
    ).split()
    if len(parent_line) != 2 or parent_line[0] != source["commit"]:
        raise ProvenanceMismatch("release authorization commit must have exactly one parent")
    if parent_line[1] != contract["scientific_source_commit"]:
        raise ProvenanceMismatch("release authorization parent is not the scientific source commit")
    changed = subprocess.check_output(
        [
            "git",
            "-C",
            root,
            "diff",
            "--name-only",
            contract["scientific_source_commit"],
            source["commit"],
        ],
        text=True,
    ).splitlines()
    if set(changed) != set(RELEASE_ARTIFACT_ALLOWLIST) or len(changed) != len(RELEASE_ARTIFACT_ALLOWLIST):
        raise ProvenanceMismatch("release-only commit changed files outside the exact allowlist")
    prereg = open(_path(root, PREREG_RELATIVE_PATH), encoding="utf-8").read()
    if "**Final execution:** authorized" not in prereg:
        raise ProvenanceMismatch("committed preregistration has not authorized final execution")
    release = dict(release)
    release["authorization_commit"] = source["commit"]
    release["release_authorization_sha256"] = sha256_file(release_path)
    return release
