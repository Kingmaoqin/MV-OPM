"""Run the preregistered descriptive minibatch-size sensitivity diagnostic."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from opm.provenance import canonical_json_sha256, environment_snapshot, git_state, sha256_file
from scripts import run_mvopm_repaired_v1_convergence as convergence

PROGRAM_ID = convergence.PROGRAM_ID
OUT = os.path.join(ROOT, "results", PROGRAM_ID, "development", "batch_sensitivity")
CONVERGENCE_OUT = os.path.join(ROOT, "results", PROGRAM_ID, "development", "convergence_audit")
DECISION_PATH = os.path.join(CONVERGENCE_OUT, "CONVERGENCE_DECISION.json")
OBSERVED_PATH = os.path.join(CONVERGENCE_OUT, "observed_convergence.csv")
EXPECTED_DECISION_SHA256 = "b1c044e93893a0d79fa13cc7e79d1f3318be76206b00c2d4a1b3d09ab742abc9"
EXPECTED_OBSERVED_SHA256 = "05bf43c4dc809e3d64e61652f785c6f9db1b9cb4fa31320724316dc629af7c3f"
EXPECTED_DEVELOPMENT_SEED_SHA256 = "30dc5779c30f3912118bb01f42a93067186bd47a1f662066db941937b81795b3"
BUDGET = 300
NEW_BATCHES = (256, 512)
ALL_BATCHES = (256, 512, 1024)
SUMMARY_FILES = (
    "batch_sensitivity.csv",
    "summary.csv",
    "BATCH_SENSITIVITY.json",
    "BATCH_SENSITIVITY_AUDIT.md",
)
FINITE_FIELDS = (
    "D_total_rff_l2",
    "D_total_mmr",
    "optimizer_best_resid",
    "optimizer_min_training_objective",
    "optimizer_final_training_objective",
    "optimizer_updates",
    "training_batch_size_cap",
    "parameter_l2",
    "h_prediction_min",
    "h_prediction_max",
    "h_prediction_rms",
    "q_prediction_min",
    "q_prediction_max",
    "q_prediction_rms",
)
EXPECTED_ROW_FIELDS = set(FINITE_FIELDS) | {
    "scenario",
    "seed",
    "budget",
    "h_prediction_change",
    "q_prediction_change",
    "training_objective_scope",
}
EXPECTED_SHARD_FIELDS = {
    "program_id",
    "status",
    "scenario",
    "seed",
    "batch_size",
    "n",
    "device",
    "source_state",
    "seed_registry_sha256",
    "hamd_input_path",
    "hamd_input_sha256",
    "convergence_decision_sha256",
    "runtime_s",
    "environment",
    "row",
}


def _allowed_generated_paths() -> list[str]:
    return [os.path.join(OUT, name) for name in SUMMARY_FILES]


def _source_identity(state: dict) -> dict:
    return {
        key: state.get(key)
        for key in ("commit", "dirty", "status_sha256", "worktree_content_sha256")
    }


def _stable_source_identity(state: dict) -> dict:
    return {key: state.get(key) for key in ("commit", "dirty")}

def _validate_row(
    row: object,
    *,
    scenario: str,
    seed: int,
    batch_size: int,
    reference: bool = False,
) -> dict:
    if not isinstance(row, dict):
        raise RuntimeError("batch-sensitivity row must be an object")
    if set(row) != EXPECTED_ROW_FIELDS:
        raise RuntimeError("batch-sensitivity row has missing or unexpected fields")
    exact = {
        "scenario": scenario,
        "seed": seed,
        "budget": BUDGET,
        "training_batch_size_cap": batch_size,
        "training_objective_scope": "minibatch_vstat_with_diagonal_regularization",
    }
    for field, value in exact.items():
        if row.get(field) != value:
            raise RuntimeError(f"batch-sensitivity row {field} mismatch")
    for field in FINITE_FIELDS:
        value = row.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"batch-sensitivity row has nonfinite or missing {field}")
    for field in ("h_prediction_change", "q_prediction_change"):
        value = row.get(field)
        if reference:
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(f"batch-1024 reference has nonfinite or missing {field}")
        elif not isinstance(value, float) or not math.isnan(value):
            raise RuntimeError(f"single-budget batch row must mark {field} as NaN")
    return row


def _validate_shard(record: object, expected: tuple, *, path: str | None = None) -> dict:
    scenario, seed, batch_size, device, source_state, seed_sha = expected
    if not isinstance(record, dict):
        raise RuntimeError("batch-sensitivity shard must be an object")
    if set(record) != EXPECTED_SHARD_FIELDS:
        raise RuntimeError("batch-sensitivity shard has missing or unexpected fields")
    exact = {
        "program_id": PROGRAM_ID,
        "status": "ok",
        "scenario": scenario,
        "seed": seed,
        "batch_size": batch_size,
        "n": 1500,
        "device": device,
        "source_state": source_state,
        "seed_registry_sha256": seed_sha,
        "hamd_input_path": convergence.HAMD_PATH,
        "hamd_input_sha256": convergence.HAMD_SHA256,
        "convergence_decision_sha256": EXPECTED_DECISION_SHA256,
    }
    for field, value in exact.items():
        if record.get(field) != value:
            raise RuntimeError(f"batch-sensitivity shard {field} mismatch")
    if not isinstance(record.get("runtime_s"), (int, float)) or record["runtime_s"] <= 0:
        raise RuntimeError("batch-sensitivity shard runtime is missing or invalid")
    if not isinstance(record.get("environment"), dict) or not record["environment"]:
        raise RuntimeError("batch-sensitivity shard environment is missing")
    _validate_row(record.get("row"), scenario=scenario, seed=seed, batch_size=batch_size)
    if path is not None:
        convergence._validate_checksum(path)
    return record


def _validate_frozen_inputs() -> tuple[list[int], str]:
    if not os.path.exists(convergence.HAMD_PATH):
        raise RuntimeError("HAMD input is missing")
    if sha256_file(convergence.HAMD_PATH) != convergence.HAMD_SHA256:
        raise RuntimeError("HAMD input differs from the preregistered SHA-256")
    if sha256_file(DECISION_PATH) != EXPECTED_DECISION_SHA256:
        raise RuntimeError("convergence decision bytes differ from the pre-final amendment")
    if sha256_file(OBSERVED_PATH) != EXPECTED_OBSERVED_SHA256:
        raise RuntimeError("observed convergence CSV differs from the pre-final amendment")
    seed_sha = sha256_file(convergence.SEED_PATH)
    if seed_sha != EXPECTED_DEVELOPMENT_SEED_SHA256:
        raise RuntimeError("development seed registry differs from the frozen SHA-256")
    decision = json.load(open(DECISION_PATH, encoding="utf-8"))
    if (
        decision.get("program_id") != PROGRAM_ID
        or decision.get("selected_budget") != BUDGET
        or decision.get("n_tasks") != 32
        or decision.get("n_budget_rows") != 128
        or decision.get("oracle_metrics_loaded") is not False
        or decision.get("seed_registry_sha256") != seed_sha
    ):
        raise RuntimeError("convergence decision does not authorize batch sensitivity")
    registry = json.load(open(convergence.SEED_PATH, encoding="utf-8"))
    seeds = registry.get("development_only_seeds")
    if (
        registry.get("program_id") != PROGRAM_ID
        or registry.get("phase") != "development_only"
        or not isinstance(seeds, list)
        or len(seeds) != 8
        or decision.get("seed_registry_sha256") != seed_sha
        or len(set(seeds)) != 8
        or not all(isinstance(seed, int) for seed in seeds)
    ):
        raise RuntimeError("invalid development seed registry")
    return seeds, seed_sha


def _one(args):
    scenario, seed, batch_size, device, source_state, seed_sha = args
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from opm.experiments.mvopm_round2.convergence_audit import audit_one

    if sha256_file(convergence.HAMD_PATH) != convergence.HAMD_SHA256:
        raise RuntimeError("HAMD input changed before batch-sensitivity task")
    started = time.time()
    rows = audit_one(
        scenario,
        seed,
        n=1500,
        device=device,
        budgets=(BUDGET,),
        batch_size=batch_size,
    )
    if len(rows) != 1:
        raise RuntimeError("batch-sensitivity task must return exactly one row")
    record = {
        "program_id": PROGRAM_ID,
        "status": "ok",
        "scenario": scenario,
        "seed": seed,
        "batch_size": batch_size,
        "n": 1500,
        "device": device,
        "source_state": source_state,
        "seed_registry_sha256": seed_sha,
        "hamd_input_path": convergence.HAMD_PATH,
        "hamd_input_sha256": convergence.HAMD_SHA256,
        "convergence_decision_sha256": EXPECTED_DECISION_SHA256,
        "runtime_s": time.time() - started,
        "environment": environment_snapshot(),
        "row": rows[0],
    }
    return _validate_shard(record, args)


def _reference_rows(seeds: list[int]) -> list[dict]:
    frame = pd.read_csv(OBSERVED_PATH)
    frame = frame.loc[frame["budget"] == BUDGET].copy()
    expected = {(scenario, seed) for scenario in convergence.SCENARIOS for seed in seeds}
    actual = {(row.scenario, int(row.seed)) for row in frame.itertuples(index=False)}
    if len(frame) != 32 or actual != expected:
        raise RuntimeError("batch-1024 convergence reference is incomplete")
    rows = frame.to_dict(orient="records")
    for row in rows:
        _validate_row(
            row,
            scenario=str(row["scenario"]),
            seed=int(row["seed"]),
            batch_size=1024,
            reference=True,
        )
    return rows


def _validate_shard_inventory(tasks: list[tuple]) -> tuple[str, int]:
    expected = set()
    for scenario, seed, batch_size, *_ in tasks:
        path = os.path.join(OUT, scenario, f"seed{seed}_batch{batch_size}.json")
        expected.update((path, path + ".sha256"))
    actual = set()
    for scenario in convergence.SCENARIOS:
        directory = os.path.join(OUT, scenario)
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if name.startswith("seed") and (
                name.endswith(".json") or name.endswith(".json.sha256")
            ):
                actual.add(os.path.join(directory, name))
    if actual != expected:
        raise RuntimeError("batch-sensitivity raw shard inventory is incomplete or unexpected")
    entries = [
        {
            "path": os.path.relpath(path, OUT).replace(os.sep, "/"),
            "sha256": sha256_file(path),
        }
        for path in sorted(actual)
    ]
    return canonical_json_sha256(entries), len(entries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        raise ValueError("workers and threads must be positive")
    start_source = git_state(ROOT, allowed_dirty_paths=_allowed_generated_paths())
    if start_source["commit"] == "nogit" or start_source["dirty"]:
        raise RuntimeError("batch sensitivity requires a clean committed source worktree")
    seeds, seed_sha = _validate_frozen_inputs()
    source_state = _source_identity(start_source)
    tasks = [
        (scenario, seed, batch_size, args.device, source_state, seed_sha)
        for scenario in convergence.SCENARIOS
        for seed in seeds
        for batch_size in NEW_BATCHES
    ]
    if len(tasks) != 64 or len({task[:3] for task in tasks}) != 64:
        raise RuntimeError("batch-sensitivity task registry is incomplete or duplicated")
    os.makedirs(OUT, exist_ok=True)
    pending, records = [], []
    for task in tasks:
        scenario, seed, batch_size = task[:3]
        path = os.path.join(OUT, scenario, f"seed{seed}_batch{batch_size}.json")
        if os.path.exists(path):
            if not args.resume:
                raise RuntimeError(f"batch-sensitivity shard exists; use --resume: {path}")
            record = json.load(open(path, encoding="utf-8"))
            records.append(_validate_shard(record, task, path=path))
        else:
            pending.append(task)
    os.environ.update({
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads),
        "NUMEXPR_NUM_THREADS": str(args.threads),
    })
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task, record in zip(pending, pool.map(_one, pending)):
            _validate_shard(record, task)
            path = os.path.join(
                OUT,
                record["scenario"],
                f"seed{record['seed']}_batch{record['batch_size']}.json",
            )
            convergence._atomic_json(path, record, checksum=True)
            records.append(_validate_shard(record, task, path=path))
            print(
                f"{len(records)}/{len(tasks)} {record['scenario']} "
                f"batch={record['batch_size']} {record['runtime_s']:.1f}s",
                flush=True,
            )
    record_keys = {(r["scenario"], r["seed"], r["batch_size"]) for r in records}
    expected_keys = {(task[0], task[1], task[2]) for task in tasks}
    if len(records) != 64 or record_keys != expected_keys:
        raise RuntimeError("batch-sensitivity shards are incomplete or duplicated")
    shard_digest, shard_files = _validate_shard_inventory(tasks)
    new_rows = [record["row"] for record in records]
    rows = _reference_rows(seeds) + new_rows
    keys = {(str(row["scenario"]), int(row["seed"]), int(row["training_batch_size_cap"])) for row in rows}
    expected = {
        (scenario, seed, batch_size)
        for scenario in convergence.SCENARIOS
        for seed in seeds
        for batch_size in ALL_BATCHES
    }
    if len(rows) != 96 or keys != expected:
        raise RuntimeError("combined batch-sensitivity evidence must contain exactly 96 rows")
    frame = pd.DataFrame(rows).sort_values(["scenario", "seed", "training_batch_size_cap"])
    evidence_path = os.path.join(OUT, "batch_sensitivity.csv")
    convergence._atomic_text(evidence_path, frame.to_csv(index=False))
    summary = (
        frame.groupby(["scenario", "training_batch_size_cap"], as_index=False)
        .agg(
            median_D=("D_total_rff_l2", "median"),
            median_MMR=("D_total_mmr", "median"),
            median_objective=("optimizer_best_resid", "median"),
            median_parameter_l2=("parameter_l2", "median"),
            median_h_rms=("h_prediction_rms", "median"),
            median_q_rms=("q_prediction_rms", "median"),
        )
        .sort_values(["scenario", "training_batch_size_cap"])
    )
    if len(summary) != 12:
        raise RuntimeError("batch-sensitivity summary must contain 12 scenario/batch cells")
    summary_path = os.path.join(OUT, "summary.csv")
    convergence._atomic_text(summary_path, summary.to_csv(index=False))
    evidence = {
        "program_id": PROGRAM_ID,
        "status": "descriptive_estimator_sensitivity_complete",
        "interpretation_scope": "finite_minibatch_estimator_sensitivity_not_population_equivalence",
        "source_state": source_state,
        "seed_registry_sha256": seed_sha,
        "hamd_input_path": convergence.HAMD_PATH,
        "hamd_input_sha256": convergence.HAMD_SHA256,
        "convergence_decision_sha256": EXPECTED_DECISION_SHA256,
        "selected_budget_fixed": BUDGET,
        "batch_sizes": list(ALL_BATCHES),
        "n_new_tasks": 64,
        "n_combined_rows": 96,
        "oracle_metrics_loaded": False,
        "device": args.device,
        "batch_sensitivity_sha256": sha256_file(evidence_path),
        "summary_sha256": sha256_file(summary_path),
        "shard_files": shard_files,
        "shard_path_hash_list_sha256": shard_digest,
        "driver_environment": environment_snapshot(),
        "shard_environment_sha256": sorted({
            canonical_json_sha256(record["environment"]) for record in records
        }),
    }
    evidence_json = os.path.join(OUT, "BATCH_SENSITIVITY.json")
    convergence._atomic_json(evidence_json, evidence)
    report = [
        "# MV-OPM repaired v1 batch-size sensitivity",
        "",
        "Descriptive finite-minibatch estimator sensitivity only; not population-objective equivalence.",
        "The fixed 300-epoch budget, final allocation, and candidate library cannot change here.",
        "No oracle quantity was loaded.",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    convergence._atomic_text(
        os.path.join(OUT, "BATCH_SENSITIVITY_AUDIT.md"),
        "\n".join(report),
    )
    end_source = git_state(ROOT, allowed_dirty_paths=_allowed_generated_paths())
    if _stable_source_identity(end_source) != _stable_source_identity(start_source):
        raise RuntimeError("source changed during batch-sensitivity diagnostic")
    if sha256_file(convergence.HAMD_PATH) != convergence.HAMD_SHA256:
        raise RuntimeError("HAMD input changed during batch-sensitivity diagnostic")
    print(json.dumps({
        "status": "complete",
        "n_new_tasks": len(tasks),
        "n_combined_rows": len(rows),
        "selected_budget_unchanged": BUDGET,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
