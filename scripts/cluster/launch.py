"""Bounded-concurrency local launcher (Slurm-array equivalent).

    python scripts/cluster/launch.py <manifest.jsonl> --workers 8 --threads 4 [--studies A,B] [--resume]

Spawns one worker SUBPROCESS per row (isolated failures), capping concurrency and per-worker BLAS/
torch threads to avoid oversubscription. Resume-aware: rows whose result.json already exists with
status 'ok' are skipped. Failed rows stay visible for resubmission (rerun the SAME seed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from opm.release_gate import (
    targets_repaired_v1_final,
    validate_repaired_v1_release,
)

from opm.provenance import (
    registered_manifest_commit,
    sha256_file,
    validate_execution_manifest,
    validate_manifest_output_namespace,
    validate_result_checksum,
    validate_result_provenance,
)

PY = os.environ.get("PY", "/home/xqin5/.conda/envs/MDPC/bin/python")
WORKER = os.path.join(ROOT, "scripts", "cluster", "worker.py")
FROZEN_FINAL_ROOT = os.path.join(ROOT, "results", "mvopm_round2", "final")
FROZEN_ROUND2_ROOT = os.path.join(ROOT, "results", "mvopm_round2")
REPAIRED_FINAL_ROOT = os.path.join(ROOT, "results", "mvopm_repaired_v1", "final")


def _terminal(
    row,
    rerun_infrastructure=False,
    *,
    array_task_id: int,
    manifest_sha256: str,
    authorization_commit: str | None = None,
    release_authorization_sha256: str | None = None,
):
    p = row["out_path"]
    if not os.path.exists(p):
        return False
    try:
        with open(p, encoding="utf-8") as handle:
            result = json.load(handle)
        validate_result_checksum(p, required=bool(row.get("source_commit")))
        validate_result_provenance(
            result,
            row,
            array_task_id=array_task_id,
            manifest_sha256=manifest_sha256,
            authorization_commit=authorization_commit,
            release_authorization_sha256=release_authorization_sha256,
        )
        if result.get("status") == "ok":
            return True
        if rerun_infrastructure and result.get("failure_kind") == "infrastructure":
            return False
        if result.get("failure_kind") == "provenance":
            return False
        # Algorithmic failures are scientific records and are terminal unless a human explicitly
        # invokes the worker for that exact row; the worker archives every previous attempt.
        return True
    except Exception as exc:
        print(f"resume rejected {p}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--studies", default=None, help="comma list to filter, e.g. A,B")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rerun-infrastructure", action="store_true",
                    help="with --resume, rerun only rows classified as infrastructure failures")
    ap.add_argument("--max", type=int, default=None, help="cap number of rows (debug)")
    args = ap.parse_args()
    if args.workers < 1 or args.threads < 1:
        raise ValueError("workers and threads must be positive")
    if args.max is not None and args.max < 1:
        raise ValueError("--max must be positive when supplied")
    if args.rerun_infrastructure and not args.resume:
        raise ValueError("--rerun-infrastructure requires --resume")

    with open(args.manifest, "rb") as handle:
        manifest_bytes = handle.read()
    rows = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines() if line.strip()]
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    frozen_commit = registered_manifest_commit(manifest_hash)
    if frozen_commit:
        raise RuntimeError(
            f"refusing to launch registered frozen manifest {manifest_hash} at historical "
            f"checkpoint {frozen_commit}; use a new program and namespace"
        )
    validate_execution_manifest(rows)
    repaired_release = None
    if targets_repaired_v1_final(rows, root=ROOT):
        repaired_release = validate_repaired_v1_release(
            rows,
            manifest_path=args.manifest,
            manifest_sha256=manifest_hash,
            root=ROOT,
        )
        if args.studies is not None or args.max is not None:
            raise RuntimeError("repaired-v1 full execution forbids study filters and row caps")
        if not args.resume or not args.rerun_infrastructure:
            raise RuntimeError(
                "repaired-v1 full execution requires --resume --rerun-infrastructure"
            )
    else:
        validate_manifest_output_namespace(
            rows,
            protected_roots=[FROZEN_ROUND2_ROOT, REPAIRED_FINAL_ROOT],
        )
    idxs = list(range(len(rows)))
    if args.studies:
        keep = set(args.studies.split(","))
        idxs = [i for i in idxs if rows[i]["study"] in keep]
    if args.resume:
        idxs = [
            i for i in idxs
            if not _terminal(
                rows[i],
                args.rerun_infrastructure,
                array_task_id=i,
                manifest_sha256=manifest_hash,
                authorization_commit=(
                    repaired_release["authorization_commit"]
                    if repaired_release is not None
                    else None
                ),
                release_authorization_sha256=(
                    repaired_release["release_authorization_sha256"]
                    if repaired_release is not None
                    else None
                ),
            )
        ]
    if args.max is not None:
        idxs = idxs[:args.max]
    print(f"launching {len(idxs)} rows, {args.workers} workers x {args.threads} threads", flush=True)

    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads), NUMEXPR_NUM_THREADS=str(args.threads),
               PYTHONUNBUFFERED="1")
    running, done, failed = {}, 0, 0
    t0 = time.time()
    pending = list(idxs)
    while pending or running:
        while pending and len(running) < args.workers:
            i = pending.pop(0)
            p = subprocess.Popen([PY, "-u", WORKER, args.manifest, str(i)], env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            running[i] = p
        time.sleep(1.0)
        for i, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[i]
                done += 1
                failed += int(rc != 0)
                if done % 10 == 0 or not pending:
                    el = time.time() - t0
                    print(f"  {done}/{len(idxs)} done ({failed} failed) {el:.0f}s "
                          f"~{el/max(done,1):.1f}s/row", flush=True)
    print(f"FINISHED {done} rows, {failed} failed, {time.time()-t0:.0f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
