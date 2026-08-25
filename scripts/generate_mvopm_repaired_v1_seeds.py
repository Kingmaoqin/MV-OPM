"""Generate disjoint cryptographic seed registries without printing seed values."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import tempfile
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROGRAM_ID = "mvopm_repaired_v1"
SEED_DIR = os.path.join(ROOT, "configs", "seeds")
OLD_REGISTRY = os.path.join(SEED_DIR, "mvopm_round2_final.json")
DEVELOPMENT_PATH = os.path.join(SEED_DIR, f"{PROGRAM_ID}_development.json")
FINAL_PATH = os.path.join(SEED_DIR, f"{PROGRAM_ID}_final.json")
PREREG_PATH = os.path.join(ROOT, "docs", PROGRAM_ID, "PREREG.md")
DECISION_PATH = os.path.join(
    ROOT, "results", PROGRAM_ID, "development", "convergence_audit",
    "CONVERGENCE_DECISION.json",
)
LOW, HIGH = 1_000_000_000, 2_000_000_000
CONVERGENCE_DIR = os.path.dirname(DECISION_PATH)
OBSERVED_PATH = os.path.join(CONVERGENCE_DIR, "observed_convergence.csv")
SUMMARY_PATH = os.path.join(CONVERGENCE_DIR, "summary.csv")
EXPECTED_DECISION_SHA256 = "b1c044e93893a0d79fa13cc7e79d1f3318be76206b00c2d4a1b3d09ab742abc9"
EXPECTED_DEVELOPMENT_SEED_SHA256 = "30dc5779c30f3912118bb01f42a93067186bd47a1f662066db941937b81795b3"
EXPECTED_HAMD_PATH = "/home/xqin5/DpressionTreatmenteffect/data-merged-def-1.csv"
EXPECTED_HAMD_SHA256 = "b40d7c17e687170c2c17c3eb892fd90a73a51e3b1e96864430ffdc2ce7264609"
EXPECTED_SOURCE_COMMIT = "13e6ab0b17480e86c739dae407681f17fe1c4082"
EXPECTED_BUDGETS = [45, 90, 180, 300]
EXPECTED_RULE = "first_adjacent_median_rff_improvement_lt_0.05_and_hq_changes_lt_0.05_else_300"
EXPECTED_OBSERVED_SHA256 = "05bf43c4dc809e3d64e61652f785c6f9db1b9cb4fa31320724316dc629af7c3f"
EXPECTED_SUMMARY_SHA256 = "2f9b0412cefe51811f9bdff98513a36df91d0dec2419f8a04cdc544257e2b644"

FINAL_COUNTS = {
    "A2": 200, "B2": 50, "C2": 50, "D2": 50, "E2": 50,
    "R": 30, "F2": 30, "G2": 30, "H2": 30, "I2": 50,
}


def _flatten(value):
    if isinstance(value, dict):
        for nested in value.values():
            yield from _flatten(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    elif isinstance(value, int):
        yield value


def _atomic_json(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".seeds.", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return digest


def _tracked_head_bytes(path: str) -> bytes:
    relative = os.path.relpath(path, ROOT).replace(os.sep, "/")
    try:
        return subprocess.check_output(["git", "-C", ROOT, "show", f"HEAD:{relative}"])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"required gate file is not committed at HEAD: {relative}") from exc


def _require_final_seed_gate() -> dict:
    status = subprocess.check_output(
        ["git", "-C", ROOT, "status", "--porcelain=v1", "--untracked-files=all"], text=True,
    )
    if status.strip():
        raise RuntimeError("final seed generation requires a clean committed worktree")
    for path in (PREREG_PATH, DECISION_PATH, OBSERVED_PATH, SUMMARY_PATH, DEVELOPMENT_PATH):
        if not os.path.exists(path):
            raise RuntimeError(f"required final-seed gate file is missing: {path}")
        current = open(path, "rb").read()
        if current != _tracked_head_bytes(path):
            raise RuntimeError(f"gate file differs from committed HEAD: {path}")
    prereg = open(PREREG_PATH, encoding="utf-8").read()
    if "**Final seed generation:** authorized" not in prereg:
        raise RuntimeError("committed preregistration has not authorized final seed generation")
    decision = json.load(open(DECISION_PATH, encoding="utf-8"))
    for marker in (
        EXPECTED_DECISION_SHA256,
        EXPECTED_OBSERVED_SHA256,
        EXPECTED_SUMMARY_SHA256,
        "rule selected the 300-epoch cap",
    ):
        if marker not in prereg:
            raise RuntimeError(f"committed preregistration is missing frozen marker: {marker}")
    if hashlib.sha256(open(DEVELOPMENT_PATH, "rb").read()).hexdigest() != EXPECTED_DEVELOPMENT_SEED_SHA256:
        raise RuntimeError("development seed registry bytes differ from the frozen SHA-256")

    source_state = decision.get("source_state")
    empty_sha = hashlib.sha256(b"").hexdigest()
    if (
        decision.get("program_id") != PROGRAM_ID
        or decision.get("status") != "observed_data_only_development_decision"
        or decision.get("n_tasks") != 32
        or decision.get("n_budget_rows") != 128
        or decision.get("oracle_metrics_loaded") is not False
        or decision.get("selected_budget") != 300
        or decision.get("budgets") != EXPECTED_BUDGETS
        or decision.get("rule") != EXPECTED_RULE
        or decision.get("device") != "cuda:0"
        or decision.get("seed_registry_sha256") != EXPECTED_DEVELOPMENT_SEED_SHA256
        or decision.get("hamd_input_path") != EXPECTED_HAMD_PATH
        or decision.get("hamd_input_sha256") != EXPECTED_HAMD_SHA256
        or not isinstance(source_state, dict)
        or source_state.get("commit") != EXPECTED_SOURCE_COMMIT
        or source_state.get("dirty") is not False
        or source_state.get("status_sha256") != empty_sha
        or source_state.get("worktree_content_sha256") != empty_sha
    ):
        raise RuntimeError("committed convergence decision is incomplete or invalid")
    if hashlib.sha256(open(DECISION_PATH, "rb").read()).hexdigest() != EXPECTED_DECISION_SHA256:
        raise RuntimeError("committed convergence decision bytes do not match the amendment")
    for field, path, expected in (
        ("observed_convergence_sha256", OBSERVED_PATH, EXPECTED_OBSERVED_SHA256),
        ("summary_sha256", SUMMARY_PATH, EXPECTED_SUMMARY_SHA256),
    ):
        actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if decision.get(field) != actual or actual != expected:
            raise RuntimeError(f"convergence decision hash mismatch: {field}")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["development", "final"])
    args = parser.parse_args()
    path = DEVELOPMENT_PATH if args.phase == "development" else FINAL_PATH
    if os.path.exists(path):
        raise RuntimeError(f"refusing to replace existing seed registry: {path}")
    decision = _require_final_seed_gate() if args.phase == "final" else None
    old = json.load(open(OLD_REGISTRY, encoding="utf-8"))
    excluded = set(_flatten(old))
    if os.path.exists(DEVELOPMENT_PATH):
        excluded.update(_flatten(json.load(open(DEVELOPMENT_PATH, encoding="utf-8"))))
    rng = secrets.SystemRandom()

    def draw(count: int) -> list[int]:
        values = []
        while len(values) < count:
            candidate = rng.randrange(LOW, HIGH)
            if candidate not in excluded:
                excluded.add(candidate)
                values.append(candidate)
        return values

    if args.phase == "development":
        payload = {
            "program_id": PROGRAM_ID,
            "phase": "development_only",
            "generation": "os_cryptographic_rng_without_replacement",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "range": [LOW, HIGH],
            "development_only_seeds": draw(8),
        }
    else:
        payload = {
            "program_id": PROGRAM_ID,
            "phase": "confirmatory_final_unexposed_before_freeze",
            "generation": "os_cryptographic_rng_without_replacement",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "range": [LOW, HIGH],
            "convergence_decision_sha256": hashlib.sha256(open(DECISION_PATH, "rb").read()).hexdigest(),
            "selected_kernel_budget": int(decision["selected_budget"]),
            "scientific_seeds": {study: draw(count) for study, count in FINAL_COUNTS.items()},
        }
    digest = _atomic_json(path, payload)
    counts = (
        {"development_only": len(payload["development_only_seeds"])}
        if args.phase == "development"
        else {key: len(value) for key, value in payload["scientific_seeds"].items()}
    )
    print(json.dumps({"path": path, "sha256": digest, "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
