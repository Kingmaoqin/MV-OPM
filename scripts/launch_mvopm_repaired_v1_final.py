"""Launch only the complete authorized MV-OPM repaired-v1 final program."""
from __future__ import annotations

import argparse
import os
import re
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "results", "mvopm_repaired_v1", "manifest_final.jsonl")
GENERIC_LAUNCHER = os.path.join(ROOT, "scripts", "cluster", "launch.py")
PYTHON = os.environ.get("PY", "/home/xqin5/.conda/envs/MDPC/bin/python")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all 6,700 authorized repaired-v1 rows with resume protection.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        raise ValueError("workers and threads must be positive")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"[0-9]+", visible):
        raise RuntimeError(
            "repaired-v1 final execution requires exactly one physical GPU through "
            "CUDA_VISIBLE_DEVICES"
        )
    command = [
        PYTHON,
        "-u",
        GENERIC_LAUNCHER,
        MANIFEST,
        "--workers",
        str(args.workers),
        "--threads",
        str(args.threads),
        "--resume",
        "--rerun-infrastructure",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
