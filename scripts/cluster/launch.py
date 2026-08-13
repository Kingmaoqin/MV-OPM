"""Bounded-concurrency local launcher (Slurm-array equivalent).

    python scripts/cluster/launch.py <manifest.jsonl> --workers 8 --threads 4 [--studies A,B] [--resume]

Spawns one worker SUBPROCESS per row (isolated failures), capping concurrency and per-worker BLAS/
torch threads to avoid oversubscription. Resume-aware: rows whose result.json already exists with
status 'ok' are skipped. Failed rows stay visible for resubmission (rerun the SAME seed).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.environ.get("PY", "/home/xqin5/.conda/envs/MDPC/bin/python")
WORKER = os.path.join(ROOT, "scripts", "cluster", "worker.py")


def _terminal(row, rerun_infrastructure=False):
    p = row["out_path"]
    if not os.path.exists(p):
        return False
    try:
        result = json.load(open(p))
        if result.get("status") == "ok":
            return True
        if rerun_infrastructure and result.get("failure_kind") == "infrastructure":
            return False
        # Algorithmic failures are scientific records and are terminal unless a human explicitly
        # invokes the worker for that exact row; the worker archives every previous attempt.
        return True
    except Exception:
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

    rows = [json.loads(l) for l in open(args.manifest)]
    idxs = list(range(len(rows)))
    if args.studies:
        keep = set(args.studies.split(","))
        idxs = [i for i in idxs if rows[i]["study"] in keep]
    if args.resume:
        idxs = [i for i in idxs if not _terminal(rows[i], args.rerun_infrastructure)]
    if args.max:
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


if __name__ == "__main__":
    main()
