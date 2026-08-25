"""Build the immutable 6,700-row MV-OPM repaired-v1 final manifest."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from opm.provenance import git_state, sha256_file

PROGRAM_ID = "mvopm_repaired_v1"
PROTOCOL_VERSION = "0.2"
RESULT_ROOT = os.path.join(ROOT, "results", PROGRAM_ID)
FINAL_ROOT = os.path.join(RESULT_ROOT, "final")
MANIFEST_PATH = os.path.join(RESULT_ROOT, "manifest_final.jsonl")
SEED_PATH = os.path.join(ROOT, "configs", "seeds", f"{PROGRAM_ID}_final.json")
DEVELOPMENT_SEED_PATH = os.path.join(
    ROOT, "configs", "seeds", f"{PROGRAM_ID}_development.json",
)
OLD_SEED_PATH = os.path.join(ROOT, "configs", "seeds", "mvopm_round2_final.json")
PREREG_PATH = os.path.join(ROOT, "docs", PROGRAM_ID, "PREREG.md")
CONVERGENCE_DECISION_PATH = os.path.join(
    RESULT_ROOT, "development", "convergence_audit", "CONVERGENCE_DECISION.json",
)
BATCH_SENSITIVITY_PATH = os.path.join(
    RESULT_ROOT, "development", "batch_sensitivity", "BATCH_SENSITIVITY.json",
)
BATCH_EVIDENCE_PATH = os.path.join(
    RESULT_ROOT, "development", "batch_sensitivity", "batch_sensitivity.csv",
)
BATCH_SUMMARY_PATH = os.path.join(RESULT_ROOT, "development", "batch_sensitivity", "summary.csv")
RELEASE_PATH = os.path.join(RESULT_ROOT, "FINAL_RELEASE.json")
RELEASE_RELATIVE_PATH = os.path.relpath(RELEASE_PATH, ROOT).replace(os.sep, "/")
HAMD_PATH = "/home/xqin5/DpressionTreatmenteffect/data-merged-def-1.csv"
HAMD_SHA256 = "b40d7c17e687170c2c17c3eb892fd90a73a51e3b1e96864430ffdc2ce7264609"
EXPECTED_DECISION_SHA256 = "b1c044e93893a0d79fa13cc7e79d1f3318be76206b00c2d4a1b3d09ab742abc9"
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
SEED_COUNTS = {
    "A2": 200,
    "B2": 50,
    "C2": 50,
    "D2": 50,
    "E2": 50,
    "R": 30,
    "F2": 30,
    "G2": 30,
    "H2": 30,
    "I2": 50,
}
LOW, HIGH = 1_000_000_000, 2_000_000_000
DERIVED_HIGH = 2_147_483_647


def _slug(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def _flatten(value):
    if isinstance(value, dict):
        for nested in value.values():
            yield from _flatten(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _flatten(nested)
    elif isinstance(value, int):
        yield value


def _load_seed_registry() -> tuple[dict, str]:
    payload = json.load(open(SEED_PATH, encoding="utf-8"))
    if (
        payload.get("program_id") != PROGRAM_ID
        or payload.get("phase") != "confirmatory_final_unexposed_before_freeze"
        or payload.get("selected_kernel_budget") != 300
        or payload.get("convergence_decision_sha256") != EXPECTED_DECISION_SHA256
        or payload.get("range") != [LOW, HIGH]
    ):
        raise RuntimeError("invalid repaired-v1 final seed registry metadata")
    bank = payload.get("scientific_seeds")
    if not isinstance(bank, dict) or set(bank) != set(SEED_COUNTS):
        raise RuntimeError("repaired-v1 final seed studies are incomplete or unexpected")
    for study, count in SEED_COUNTS.items():
        values = bank.get(study)
        if (
            not isinstance(values, list)
            or len(values) != count
            or len(set(values)) != count
            or not all(isinstance(value, int) and LOW <= value < HIGH for value in values)
        ):
            raise RuntimeError(f"invalid repaired-v1 final seeds for {study}")
    final_values = list(_flatten(bank))
    if len(final_values) != len(set(final_values)):
        raise RuntimeError("repaired-v1 final scientific seeds are not globally unique")
    old = set(_flatten(json.load(open(OLD_SEED_PATH, encoding="utf-8"))))
    development = set(_flatten(json.load(open(DEVELOPMENT_SEED_PATH, encoding="utf-8"))))
    if set(final_values) & (old | development):
        raise RuntimeError("repaired-v1 final scientific seeds overlap prior/development seeds")
    return payload, sha256_file(SEED_PATH)


def _validate_batch_output_hashes(
    batch: dict,
    *,
    evidence_path: str,
    summary_path: str,
) -> tuple[str, str]:
    evidence_sha = sha256_file(evidence_path)
    summary_sha = sha256_file(summary_path)
    if batch.get("batch_sensitivity_sha256") != evidence_sha:
        raise RuntimeError("batch-sensitivity CSV differs from its frozen digest")
    if batch.get("summary_sha256") != summary_sha:
        raise RuntimeError("batch-sensitivity summary differs from its frozen digest")
    return evidence_sha, summary_sha


def _validate_frozen_development_evidence() -> tuple[str, str]:
    decision_sha = sha256_file(CONVERGENCE_DECISION_PATH)
    if decision_sha != EXPECTED_DECISION_SHA256:
        raise RuntimeError("convergence decision differs from the pre-final amendment")
    decision = json.load(open(CONVERGENCE_DECISION_PATH, encoding="utf-8"))
    if (
        decision.get("selected_budget") != 300
        or decision.get("n_tasks") != 32
        or decision.get("n_budget_rows") != 128
        or decision.get("oracle_metrics_loaded") is not False
        or decision.get("hamd_input_path") != HAMD_PATH
        or decision.get("hamd_input_sha256") != HAMD_SHA256
    ):
        raise RuntimeError("convergence decision is incomplete or invalid")
    batch_sha = sha256_file(BATCH_SENSITIVITY_PATH)
    batch = json.load(open(BATCH_SENSITIVITY_PATH, encoding="utf-8"))
    if (
        batch.get("program_id") != PROGRAM_ID
        or batch.get("status") != "descriptive_estimator_sensitivity_complete"
        or batch.get("selected_budget_fixed") != 300
        or batch.get("batch_sizes") != [256, 512, 1024]
        or batch.get("n_new_tasks") != 64
        or batch.get("n_combined_rows") != 96
        or batch.get("oracle_metrics_loaded") is not False
        or batch.get("hamd_input_path") != HAMD_PATH
        or batch.get("hamd_input_sha256") != HAMD_SHA256
        or batch.get("convergence_decision_sha256") != EXPECTED_DECISION_SHA256
    ):
        raise RuntimeError("batch-sensitivity evidence is incomplete or invalid")
    _validate_batch_output_hashes(
        batch,
        evidence_path=BATCH_EVIDENCE_PATH,
        summary_path=BATCH_SUMMARY_PATH,
    )
    return decision_sha, batch_sha


def _final_cfg() -> dict:
    return {
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


def build_rows(
    *,
    source_commit: str,
    seed_payload: dict,
    seed_registry_sha256: str,
    convergence_decision_sha256: str,
    batch_sensitivity_sha256: str,
) -> list[dict]:
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise RuntimeError("scientific source commit must be full lowercase hexadecimal")
    from opm.experiments.mvopm_round2.regime_matrix import frozen_regimes

    bank = seed_payload["scientific_seeds"]
    cfg = _final_cfg()
    scientific_values = set(_flatten(bank))
    used_derived = set(scientific_values)
    rows: list[dict] = []

    def derived(kind: str, study: str, config_id: str, scientific_seed: int) -> int:
        nonce = 0
        while True:
            payload = "\0".join((
                PROGRAM_ID,
                kind,
                study,
                config_id,
                str(scientific_seed),
                str(nonce),
            )).encode("utf-8")
            candidate = 1 + int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % DERIVED_HIGH
            if candidate not in used_derived:
                used_derived.add(candidate)
                return candidate
            nonce += 1

    def add(*, study: str, dispatch: str, scientific_seed: int, config_id: str, K: int, **kwargs):
        row = {
            "program_id": PROGRAM_ID,
            "phase": "final",
            "protocol_version": PROTOCOL_VERSION,
            "study": study,
            "dispatch": dispatch,
            "scientific_seed": int(scientific_seed),
            "split_seed": int(scientific_seed),
            "bootstrap_seed": derived("bootstrap", study, config_id, scientific_seed),
            "bank_seed": derived("rff_bank", study, config_id, scientific_seed),
            "config_id": config_id,
            "K": K,
            "cfg": cfg,
            "source_commit": source_commit,
            "require_clean_source": True,
            "seed_registry_sha256": seed_registry_sha256,
            "convergence_decision_sha256": convergence_decision_sha256,
            "batch_sensitivity_sha256": batch_sensitivity_sha256,
            "release_authorization_path": RELEASE_RELATIVE_PATH,
            **kwargs,
        }
        row["out_path"] = os.path.join(
            FINAL_ROOT,
            study,
            _slug(config_id),
            f"seed{scientific_seed}",
            "result.json",
        )
        rows.append(row)

    grid = (0.0, 0.05, 0.10, 0.20, 0.40)
    for r_h in grid:
        for r_q in grid:
            config_id = f"rh{r_h}_rq{r_q}_n4000"
            for seed in bank["A2"]:
                add(
                    study="A2",
                    dispatch="aligned_mechanism",
                    scientific_seed=seed,
                    config_id=config_id,
                    K=2,
                    r_h=r_h,
                    r_q=r_q,
                    n=4000,
                    c_U=1.0,
                )

    main = (
        ("B2", "S1", 4, 4000),
        ("C2", "S2", 4, 4000),
        ("D2", "nonlinear", 2, 4000),
        ("E2", "hamd", 4, 1500),
    )
    for study, scenario, K, n in main:
        for seed in bank[study]:
            extra = {}
            candidates = None
            if study == "E2":
                candidates = [
                    "kernel_kernel",
                    "sieve1_sieve1",
                    "kernel_sieve1",
                    "sieve1_kernel",
                ]
                extra = {"hamd_input_path": HAMD_PATH, "hamd_input_sha256": HAMD_SHA256}
            add(
                study=study,
                dispatch="selection_v2",
                scientific_seed=seed,
                config_id=f"{scenario}_n{n}",
                K=K,
                scenario=scenario,
                n=n,
                c_U=1.0,
                candidate_names=candidates,
                **extra,
            )

    regimes = frozen_regimes()
    if len(regimes) != 20 or len({regime["regime_id"] for regime in regimes}) != 20:
        raise RuntimeError("frozen structural regime matrix must contain 20 unique regimes")
    for regime in regimes:
        for seed in bank["R"]:
            add(
                study="R",
                dispatch="selection_v2",
                scientific_seed=seed,
                config_id=regime["regime_id"],
                K=2,
                scenario="bridge_complexity",
                n=regime["n"],
                c_U=1.0,
                regime=regime,
                candidate_names=regime["candidate_names"],
            )

    for c_u in (0.0, 0.5, 1.0, 2.0):
        for seed in bank["F2"]:
            add(
                study="F2",
                dispatch="selection_v2",
                scientific_seed=seed,
                config_id=f"S1_cU{c_u}_n4000",
                K=4,
                scenario="S1",
                n=4000,
                c_U=c_u,
                evaluate_ignorability=True,
            )

    for kind in ("additive", "missing", "shift", "heavy_tail"):
        for level in (0.0, 0.1, 0.2, 0.3):
            if level == 0.0 and kind != "additive":
                continue
            for seed in bank["G2"]:
                add(
                    study="G2",
                    dispatch="selection_v2",
                    scientific_seed=seed,
                    config_id=f"S1_{kind}{level}_n4000",
                    K=4,
                    scenario="S1",
                    n=4000,
                    c_U=1.0,
                    proxy_mod={"kind": kind, "level": level},
                )

    for family in ("L", "N"):
        for n in (1000, 2000, 4000, 8000):
            regime = {"family": family, "proxy_noise": "low", "p": 10}
            for seed in bank["H2"]:
                add(
                    study="H2",
                    dispatch="selection_v2",
                    scientific_seed=seed,
                    config_id=f"{family}_n{n}_low",
                    K=2,
                    scenario="bridge_complexity",
                    n=n,
                    c_U=1.0,
                    regime=regime,
                )

    inadequacy = (
        (
            "easy_L_full",
            {"family": "L", "proxy_noise": "low", "p": 10},
            None,
            {},
            "engineered_easy_full_library",
        ),
        (
            "N_sieve1_only",
            {"family": "N", "proxy_noise": "low", "p": 10},
            ["sieve1_sieve1"],
            {},
            "engineered_nonlinear_sieve1_only",
        ),
        (
            "N_kernel_overregularized",
            {"family": "N", "proxy_noise": "high", "p": 10},
            ["kernel_kernel"],
            {"weight_decay": 1.0},
            "engineered_overregularized_kernel",
        ),
    )
    for config_id, regime, candidates, override, label in inadequacy:
        for seed in bank["I2"]:
            add(
                study="I2",
                dispatch="selection_v2",
                scientific_seed=seed,
                config_id=config_id,
                K=2,
                scenario="bridge_complexity",
                n=4000,
                c_U=1.0,
                regime=regime,
                candidate_names=candidates,
                bridge_override=override,
                library_scenario_label=label,
            )

    counts = Counter(row["study"] for row in rows)
    if len(rows) != 6700 or dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError(f"final manifest count mismatch: {len(rows)} {dict(counts)}")
    if {row["study"] for row in rows} != set(EXPECTED_COUNTS):
        raise RuntimeError("final manifest study set is incomplete or unexpected")
    if any(row["dispatch"] == "rhc_v2" or row["study"] == "J2" for row in rows):
        raise RuntimeError("J2/RHC is not authorized in repaired-v1")
    paths = [row["out_path"] for row in rows]
    task_keys = [(row["study"], row["config_id"], row["scientific_seed"]) for row in rows]
    for label, values in (
        ("out_path", paths),
        ("task key", task_keys),
        ("bootstrap seed", [row["bootstrap_seed"] for row in rows]),
        ("bank seed", [row["bank_seed"] for row in rows]),
    ):
        if len(values) != len(set(values)):
            raise RuntimeError(f"final manifest contains duplicate {label}")
    derived_values = [
        value
        for row in rows
        for value in (row["bootstrap_seed"], row["bank_seed"])
    ]
    if len(derived_values) != len(set(derived_values)):
        raise RuntimeError("bootstrap and bank seed domains overlap")
    if set(derived_values) & scientific_values:
        raise RuntimeError("derived seeds overlap scientific seeds")
    return rows


def _manifest_bytes(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def _atomic_create_or_verify(path: str, payload: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        if open(path, "rb").read() != payload:
            raise RuntimeError("refusing to replace an existing manifest with different bytes")
        return
    fd, temporary = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(os.path.dirname(path), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build() -> tuple[list[dict], str]:
    if os.path.exists(RELEASE_PATH):
        raise RuntimeError("release authorization already exists; manifest rebuilding is forbidden")
    source = git_state(ROOT)
    if source["commit"] == "nogit" or source["dirty"]:
        raise RuntimeError("final manifest requires a clean committed scientific source")
    prereg = open(PREREG_PATH, encoding="utf-8").read()
    if (
        "**Protocol version:** 0.2 (pre-final convergence amendment)" not in prereg
        or "**Final seed generation:** authorized" not in prereg
        or "**Final execution:** forbidden" not in prereg
    ):
        raise RuntimeError("pre-final amendment does not authorize manifest preparation")
    seed_payload, seed_sha = _load_seed_registry()
    decision_sha, batch_sha = _validate_frozen_development_evidence()
    rows = build_rows(
        source_commit=source["commit"],
        seed_payload=seed_payload,
        seed_registry_sha256=seed_sha,
        convergence_decision_sha256=decision_sha,
        batch_sensitivity_sha256=batch_sha,
    )
    payload = _manifest_bytes(rows)
    _atomic_create_or_verify(MANIFEST_PATH, payload)
    return rows, hashlib.sha256(payload).hexdigest()


def main() -> None:
    rows, digest = build()
    print(json.dumps({
        "path": MANIFEST_PATH,
        "sha256": digest,
        "rows": len(rows),
        "per_study": dict(Counter(row["study"] for row in rows)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
