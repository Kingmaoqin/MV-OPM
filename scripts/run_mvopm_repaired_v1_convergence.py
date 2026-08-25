"""Run the preregistered observed-data convergence audit in the new evidence namespace."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from opm.provenance import canonical_json_sha256, environment_snapshot, git_state, sha256_file

PROGRAM_ID = "mvopm_repaired_v1"
OUT = os.path.join(ROOT, "results", PROGRAM_ID, "development", "convergence_audit")
SEED_PATH = os.path.join(ROOT, "configs", "seeds", f"{PROGRAM_ID}_development.json")
HAMD_PATH = "/home/xqin5/DpressionTreatmenteffect/data-merged-def-1.csv"
HAMD_SHA256 = "b40d7c17e687170c2c17c3eb892fd90a73a51e3b1e96864430ffdc2ce7264609"
SCENARIOS = ("S1", "S2", "nonlinear", "hamd")
BUDGETS = (45, 90, 180, 300)
SUMMARY_FILES = (
    "observed_convergence.csv",
    "summary.csv",
    "CONVERGENCE_DECISION.json",
    "CONVERGENCE_AUDIT.md",
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


def _allowed_generated_paths() -> list[str]:
    return [os.path.join(OUT, filename) for filename in SUMMARY_FILES]


def _atomic_bytes(path: str, payload: bytes) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".audit.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_text(path: str, payload: str) -> None:
    _atomic_bytes(path, payload.encode("utf-8"))


def _atomic_json(path: str, payload: dict, *, checksum: bool = False) -> None:
    encoded = json.dumps(payload, sort_keys=True, allow_nan=True).encode("utf-8")
    _atomic_bytes(path, encoded)
    if checksum:
        _atomic_text(path + ".sha256", sha256_file(path) + "\n")


def _validate_checksum(path: str) -> None:
    checksum_path = path + ".sha256"
    if not os.path.exists(checksum_path):
        raise RuntimeError(f"resume shard has no checksum: {path}")
    expected = open(checksum_path, encoding="ascii").read().strip()
    actual = sha256_file(path)
    if expected != actual:
        raise RuntimeError(f"resume shard checksum mismatch: {path}")


def _source_identity(state: dict) -> dict:
    return {
        key: state.get(key)
        for key in ("commit", "dirty", "status_sha256", "worktree_content_sha256")
    }


def _validate_rows(rows: object, *, scenario: str, seed: int) -> None:
    if not isinstance(rows, list) or len(rows) != len(BUDGETS):
        raise RuntimeError("audit shard must contain exactly four budget rows")
    by_budget = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("audit budget row must be an object")
        budget = row.get("budget")
        if budget in by_budget:
            raise RuntimeError("audit shard contains a duplicate budget")
        by_budget[budget] = row
    if set(by_budget) != set(BUDGETS):
        raise RuntimeError("audit shard budgets are incomplete or unexpected")
    for budget in BUDGETS:
        row = by_budget[budget]
        if row.get("scenario") != scenario or row.get("seed") != seed:
            raise RuntimeError("audit row scenario/seed mismatch")
        if row.get("training_objective_scope") != "minibatch_vstat_with_diagonal_regularization":
            raise RuntimeError("audit row objective scope mismatch")
        for field in FINITE_FIELDS:
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(f"audit row has nonfinite or missing {field}")
        for field in ("h_prediction_change", "q_prediction_change"):
            value = row.get(field)
            if budget == BUDGETS[0]:
                if not isinstance(value, float) or not math.isnan(value):
                    raise RuntimeError(f"first budget must mark {field} as NaN")
            elif not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(f"later budget has nonfinite or missing {field}")
        if int(row["training_batch_size_cap"]) != 1024:
            raise RuntimeError("audit row batch-size cap mismatch")


def _validate_shard(record: object, expected: tuple, *, path: str | None = None) -> dict:
    scenario, seed, n, device, source_identity, seed_sha, hamd_sha = expected
    if not isinstance(record, dict):
        raise RuntimeError("audit shard must be an object")
    exact = {
        "program_id": PROGRAM_ID,
        "status": "ok",
        "scenario": scenario,
        "seed": seed,
        "n": n,
        "device": device,
        "source_state": source_identity,
        "seed_registry_sha256": seed_sha,
        "hamd_input_path": HAMD_PATH,
        "hamd_input_sha256": hamd_sha,
    }
    for field, value in exact.items():
        if record.get(field) != value:
            raise RuntimeError(f"audit shard {field} mismatch")
    if not isinstance(record.get("runtime_s"), (int, float)) or record["runtime_s"] <= 0:
        raise RuntimeError("audit shard runtime is missing or invalid")
    if not isinstance(record.get("environment"), dict) or not record["environment"]:
        raise RuntimeError("audit shard environment snapshot is missing")
    _validate_rows(record.get("rows"), scenario=scenario, seed=seed)
    if path is not None:
        _validate_checksum(path)
    return record


def _one(args):
    scenario, seed, n, device, source_identity, seed_sha, hamd_sha = args
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from opm.experiments.mvopm_round2.convergence_audit import audit_one

    if sha256_file(HAMD_PATH) != hamd_sha:
        raise RuntimeError("HAMD input changed before audit task")
    started = time.time()
    rows = audit_one(scenario, seed, n=n, device=device)
    record = {
        "program_id": PROGRAM_ID,
        "status": "ok",
        "scenario": scenario,
        "seed": seed,
        "n": n,
        "device": device,
        "source_state": source_identity,
        "seed_registry_sha256": seed_sha,
        "hamd_input_path": HAMD_PATH,
        "hamd_input_sha256": hamd_sha,
        "runtime_s": time.time() - started,
        "environment": environment_snapshot(),
        "rows": rows,
    }
    return _validate_shard(record, args)


def _assert_end_state(start_source: dict, start_hamd_sha: str) -> None:
    end_source = git_state(ROOT, allowed_dirty_paths=_allowed_generated_paths())
    if _source_identity(end_source) != _source_identity(start_source):
        raise RuntimeError("source changed during convergence audit")
    if sha256_file(HAMD_PATH) != start_hamd_sha:
        raise RuntimeError("HAMD input changed during convergence audit")


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
        raise RuntimeError("convergence audit requires a clean committed source worktree")
    if not os.path.exists(HAMD_PATH) or sha256_file(HAMD_PATH) != HAMD_SHA256:
        raise RuntimeError("HAMD input is missing or differs from the preregistered SHA-256")
    registry = json.load(open(SEED_PATH, encoding="utf-8"))
    if registry.get("program_id") != PROGRAM_ID or registry.get("phase") != "development_only":
        raise RuntimeError("invalid development seed registry")
    seeds = registry.get("development_only_seeds")
    if not isinstance(seeds, list) or len(seeds) != 8 or len(set(seeds)) != 8:
        raise RuntimeError("development registry must contain eight unique seeds")
    if not all(isinstance(seed, int) for seed in seeds):
        raise RuntimeError("development seeds must be integers")
    seed_sha = sha256_file(SEED_PATH)
    source_identity = _source_identity(start_source)
    tasks = [
        (scenario, seed, 1500, args.device, source_identity, seed_sha, HAMD_SHA256)
        for scenario in SCENARIOS
        for seed in seeds
    ]
    if len(tasks) != 32 or len({(task[0], task[1]) for task in tasks}) != 32:
        raise RuntimeError("audit task registry is incomplete or duplicated")
    os.makedirs(OUT, exist_ok=True)
    pending, results = [], []
    for task in tasks:
        scenario, seed = task[0], task[1]
        path = os.path.join(OUT, scenario, f"seed{seed}.json")
        if os.path.exists(path):
            if not args.resume:
                raise RuntimeError(f"audit shard already exists; use --resume: {path}")
            record = json.load(open(path, encoding="utf-8"))
            results.append(_validate_shard(record, task, path=path))
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
            path = os.path.join(OUT, record["scenario"], f"seed{record['seed']}.json")
            _atomic_json(path, record, checksum=True)
            _validate_shard(record, task, path=path)
            results.append(record)
            print(
                f"{len(results)}/{len(tasks)} {record['scenario']} {record['runtime_s']:.1f}s",
                flush=True,
            )
    _assert_end_state(start_source, HAMD_SHA256)
    expected_keys = {(scenario, seed) for scenario in SCENARIOS for seed in seeds}
    result_keys = {(record["scenario"], record["seed"]) for record in results}
    if len(results) != 32 or result_keys != expected_keys:
        raise RuntimeError("convergence audit is incomplete or duplicated")
    rows = [row for record in results for row in record["rows"]]
    if len(rows) != 128:
        raise RuntimeError("convergence audit must contain exactly 128 budget rows")
    frame = pd.DataFrame(rows).sort_values(["scenario", "seed", "budget"])
    observed_path = os.path.join(OUT, "observed_convergence.csv")
    _atomic_text(observed_path, frame.to_csv(index=False))
    from opm.experiments.mvopm_round2.convergence_audit import choose_plateau_budget

    selected = choose_plateau_budget(rows)
    if selected not in BUDGETS:
        raise RuntimeError("plateau rule returned an unregistered budget")
    summary = (
        frame.groupby(["scenario", "budget"], as_index=False)
        .agg(
            median_D=("D_total_rff_l2", "median"),
            median_h_change=("h_prediction_change", "median"),
            median_q_change=("q_prediction_change", "median"),
            median_objective=("optimizer_best_resid", "median"),
        )
        .sort_values(["scenario", "budget"])
    )
    if len(summary) != len(SCENARIOS) * len(BUDGETS):
        raise RuntimeError("convergence summary has incomplete scenario/budget cells")
    summary_path = os.path.join(OUT, "summary.csv")
    _atomic_text(summary_path, summary.to_csv(index=False))
    decision = {
        "program_id": PROGRAM_ID,
        "status": "observed_data_only_development_decision",
        "source_state": source_identity,
        "seed_registry_sha256": seed_sha,
        "hamd_input_path": HAMD_PATH,
        "hamd_input_sha256": HAMD_SHA256,
        "n_tasks": len(tasks),
        "n_budget_rows": len(rows),
        "budgets": list(BUDGETS),
        "selected_budget": int(selected),
        "rule": "first_adjacent_median_rff_improvement_lt_0.05_and_hq_changes_lt_0.05_else_300",
        "oracle_metrics_loaded": False,
        "device": args.device,
        "observed_convergence_sha256": sha256_file(observed_path),
        "summary_sha256": sha256_file(summary_path),
        "driver_environment": environment_snapshot(),
        "shard_environment_sha256": sorted({
            canonical_json_sha256(record["environment"]) for record in results
        }),
    }
    _atomic_json(os.path.join(OUT, "CONVERGENCE_DECISION.json"), decision)
    report = [
        "# MV-OPM repaired v1 convergence audit",
        "",
        "Observed-data criteria only; no oracle ATE error, PEHE, or bridge truth was loaded.",
        "",
        f"Selected cap under the frozen rule: **{selected} epochs**.",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    _atomic_text(os.path.join(OUT, "CONVERGENCE_AUDIT.md"), "\n".join(report))
    print(json.dumps({"status": "complete", "selected_budget": selected, "n_tasks": len(tasks)}))


if __name__ == "__main__":
    main()
