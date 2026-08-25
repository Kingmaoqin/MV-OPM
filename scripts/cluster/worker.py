"""Run ONE manifest row (like one Slurm-array task) and save a raw result.json with provenance.

    python scripts/cluster/worker.py <manifest.jsonl> <row_index>

Thread caps must be set by the launcher via env (OMP_NUM_THREADS etc.) BEFORE importing torch.
"""
from __future__ import annotations

import json
import fcntl
import os
import shutil
import socket
import sys
import tempfile
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from opm.provenance import (
    environment_snapshot,
    git_state,
    registered_manifest_commit,
    sha256_file,
    validate_execution_manifest,
    validate_manifest_output_namespace,
)

FROZEN_FINAL_ROOT = os.path.join(ROOT, "results", "mvopm_round2", "final")


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _acquire_result_lock(out_path: str):
    """Hold an OS lock for the complete task; locks are released automatically after crashes."""
    handle = open(out_path + ".lock", "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another worker already owns {out_path}")
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} host={socket.gethostname()}\n")
    handle.flush()
    return handle


def _atomic_json_dump(path: str, value: dict) -> str:
    """Write a complete JSON record and checksum before atomically promoting it."""
    directory = os.path.dirname(path)
    fd, temporary = tempfile.mkstemp(prefix=".result.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, default=_json_default)
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
    digest = sha256_file(path)
    checksum_path = path + ".sha256"
    fd, checksum_tmp = tempfile.mkstemp(prefix=".checksum.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(digest + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(checksum_tmp, checksum_path)
    finally:
        if os.path.exists(checksum_tmp):
            os.unlink(checksum_tmp)
    return digest


def run_row(row: dict) -> dict:
    import numpy as np  # noqa
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if row["dispatch"] in {"aligned_mechanism", "selection_v2", "rhc_v2"}:
        from opm.experiments.mvopm_round2.core import Round2Config
        from opm.experiments.mvopm_round2.studies import (
            aligned_mechanism_row, rhc_stability_row, selection_study_row,
        )
        c = row["cfg"]
        if row["dispatch"] == "aligned_mechanism":
            return aligned_mechanism_row(
                row["scientific_seed"], row["r_h"], row["r_q"], n=row["n"],
                c_U=row.get("c_U", 1.0), n_rff=c["n_rff"], bank_seed=row["bank_seed"],
            )
        cfg = Round2Config(
            K=row["K"], n_folds=c["outer_folds"], inner_folds=c["inner_folds"],
            seed=row["scientific_seed"], split_seed=row["split_seed"],
            bootstrap_seed=row["bootstrap_seed"], bank_seed=row["bank_seed"],
            n_boot=c["n_boot"], n_rff=c["n_rff"], alpha=c["alpha"], device=c["device"],
            candidate_names=row.get("candidate_names"), bridge_kwargs={**c["bridge"], **row.get("bridge_override", {})},
            head_kwargs=c["head"], run_nested=c.get("run_nested", True),
        )
        if row["dispatch"] == "rhc_v2":
            return rhc_stability_row(row["scientific_seed"], cfg)
        return selection_study_row(
            row["study"], row["scenario"], row["n"], row["scientific_seed"], cfg,
            c_U=row.get("c_U", 1.0), proxy_mod=row.get("proxy_mod"), regime=row.get("regime"),
            evaluate_ignorability=row.get("evaluate_ignorability", False),
            library_scenario_label=row.get("library_scenario_label"),
            library_truth=row.get("library_truth"),
        )
    from opm.experiments.mvopm.core import SelCfg
    from opm.experiments.mvopm.studies import mechanism_row, selection_row, rhc_row
    c = row["cfg"]
    disp = row["dispatch"]
    if disp == "mechanism":
        return mechanism_row(row["seed"], row["r_h"], row["r_q"], n=row["n"], c_U=row.get("c_U", 1.0),
                             n_rff=c["n_rff"])
    K = 4 if row.get("scenario") in ("S1", "S2", "nonlinear", "hamd") and row.get("scenario") != "nonlinear" else 2
    # scenario K: S1/S2/hamd=4, nonlinear/exact/rhc=2
    K = {"S1": 4, "S2": 4, "hamd": 4, "nonlinear": 2, "exact": 2}.get(row.get("scenario"), 2)
    cfg = SelCfg(K=K, n_folds=c["n_folds"], seed=row["seed"], kind=c["kind"], n_rff=c["n_rff"],
                 bridge_kwargs=c["bridge"], head_kwargs=c["head"])
    if disp == "rhc":
        cfg.K = 2
        return rhc_row(row["seed"], cfg)
    if disp == "selection":
        return selection_row(row["study"], row["scenario"], row["n"], row["seed"], cfg,
                             c_U=row.get("c_U", 1.0), mod=row.get("mod"))
    raise ValueError(f"unknown dispatch {disp}")


def main():
    manifest, idx = sys.argv[1], int(sys.argv[2])
    manifest_hash = sha256_file(manifest)
    frozen_commit = registered_manifest_commit(manifest_hash)
    if frozen_commit:
        raise RuntimeError(
            f"refusing all execution through registered frozen manifest {manifest_hash} "
            f"at historical checkpoint {frozen_commit}; use a new program and namespace"
        )
    rows = [json.loads(l) for l in open(manifest)]
    validate_execution_manifest(rows)
    validate_manifest_output_namespace(rows, protected_roots=[FROZEN_FINAL_ROOT])
    row = rows[idx]
    out_path = row["out_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # The manifest is generated after the scientific source commit and can therefore be the
    # sole tracked worktree change.  Its exact bytes are independently pinned by SHA-256.
    source = git_state(ROOT, allowed_dirty_paths=[manifest])
    lock_handle = _acquire_result_lock(out_path)
    t0 = time.time()
    scientific_seed = row.get("scientific_seed", row.get("seed"))
    prov = {"experiment": row["study"], "dispatch": row["dispatch"], "seed": scientific_seed,
            "scientific_seed": scientific_seed, "bootstrap_seed": row.get("bootstrap_seed"),
            "split_seed": row.get("split_seed"), "bank_seed": row.get("bank_seed"),
            "array_task_id": idx, "git_commit": source["commit"][:7],
            "git_commit_full": source["commit"], "git_dirty": source["dirty"],
            "git_worktree_dirty": source["worktree_dirty"],
            "git_dirty_paths": source["dirty_paths"],
            "git_allowed_dirty_paths": source["allowed_dirty_paths"],
            "git_status_sha256": source["status_sha256"],
            "git_worktree_content_sha256": source["worktree_content_sha256"],
            "manifest_sha256": manifest_hash, "hostname": socket.gethostname(),
            "environment": environment_snapshot(),
            "omp_threads": os.environ.get("OMP_NUM_THREADS", "?"),
            "mkl_threads": os.environ.get("MKL_NUM_THREADS", "?"),
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "?"),
            "numexpr_threads": os.environ.get("NUMEXPR_NUM_THREADS", "?"),
            "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "config": row, "warnings": []}
    try:
        expected_commit = row.get("source_commit")
        if expected_commit and source["commit"] != expected_commit:
            raise RuntimeError(
                f"source commit {source['commit']} does not match manifest {expected_commit}"
            )
        if row.get("require_clean_source", False) and source["dirty"]:
            raise RuntimeError("manifest requires a clean source, but worker worktree is dirty")
        result = run_row(row)
        end_source = git_state(ROOT, allowed_dirty_paths=[manifest])
        if any(end_source[key] != source[key] for key in (
            "commit", "dirty", "status_sha256", "worktree_content_sha256",
        )):
            raise RuntimeError(
                "source changed during task execution; refusing to promote a mixed-source result"
            )
        prov.update({"status": "ok", "exit_code": 0, "runtime_s": round(time.time() - t0, 1),
                     "end_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "result": result})
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        infrastructure_markers = (
            "out of memory", "cannot allocate memory", "no space left", "input/output error",
        )
        provenance_markers = (
            "source commit", "source changed", "manifest requires", "registered checkpoint",
        )
        if any(marker in msg.lower() for marker in provenance_markers):
            failure_kind = "provenance"
        elif isinstance(e, (MemoryError, OSError)) or any(
            marker in msg.lower() for marker in infrastructure_markers
        ):
            failure_kind = "infrastructure"
        else:
            failure_kind = "algorithmic"
        prov.update({"status": "error", "exit_code": 1, "runtime_s": round(time.time() - t0, 1),
                     "end_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "failure_kind": failure_kind,
                     "error": msg, "traceback": traceback.format_exc()})
    # Refresh after the task so NumPy/PyTorch runtime, BLAS, CUDA, and determinism details are
    # available even though imports are intentionally delayed until after thread caps are set.
    prov["environment"] = environment_snapshot()
    try:
        # Never erase a prior attempt. Explicit reruns keep the same scientific seed and archive
        # the complete previous record alongside the canonical latest result.
        if os.path.exists(out_path):
            attempts = os.path.join(os.path.dirname(out_path), "attempts")
            os.makedirs(attempts, exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            archive = os.path.join(
                attempts, f"result_before_{stamp}_{time.time_ns()}_{os.getpid()}.json",
            )
            shutil.copy2(out_path, archive)
        result_sha256 = _atomic_json_dump(out_path, prov)
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    print(f"[{idx}] {row['study']} seed{scientific_seed} {prov['status']} {prov['runtime_s']}s -> {out_path}",
          f"sha256={result_sha256[:12]}", flush=True)
    sys.exit(prov["exit_code"])


if __name__ == "__main__":
    main()
