"""Wait without preemption for a physical GPU to remain idle, then run one queued command."""
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import time


def _gpu_state(index: int) -> tuple[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    memory, utilization = [int(part.strip()) for part in output.split(",")]
    return memory, utilization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--memory-max-mib", type=int, default=1024)
    parser.add_argument("--util-max", type=int, default=10)
    parser.add_argument("--checks", type=int, default=3)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--lock", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.gpu < 0:
        raise ValueError("gpu index must be nonnegative")
    if args.memory_max_mib < 0:
        raise ValueError("memory threshold must be nonnegative")
    if not 0 <= args.util_max <= 100:
        raise ValueError("utilization threshold must be between 0 and 100")
    if args.checks < 1:
        raise ValueError("checks must be at least one")
    if args.interval < 1:
        raise ValueError("interval must be at least one second")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise RuntimeError("queued command is required after --")
    os.makedirs(os.path.dirname(os.path.abspath(args.lock)), exist_ok=True)
    with open(args.lock, "a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"queue lock already held: {args.lock}") from exc
        idle = 0
        while idle < args.checks:
            memory, utilization = _gpu_state(args.gpu)
            idle = idle + 1 if memory <= args.memory_max_mib and utilization <= args.util_max else 0
            print(
                f"gpu={args.gpu} memory_mib={memory} util={utilization} "
                f"idle_checks={idle}/{args.checks}",
                flush=True,
            )
            if idle < args.checks:
                time.sleep(args.interval)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
        print(f"starting queued command on physical GPU {args.gpu}", flush=True)
        completed = subprocess.run(command, env=env, check=False)
        print(f"queued command exit_code={completed.returncode}", flush=True)
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
