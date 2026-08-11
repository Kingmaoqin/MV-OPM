"""Single-command experiment runner.

    python -m opm.run experiment=E1
    python -m opm.run experiment=E3 scenario=S1 seed=0
"""
from __future__ import annotations

import importlib
import sys

from .experiments.common import load_config, parse_kv


def main(argv):
    kv = parse_kv(argv)
    exp = kv.pop("experiment", None)
    if exp is None:
        print("usage: python -m opm.run experiment=E1 [key=value ...]")
        sys.exit(1)
    cfg = load_config(exp, overrides=kv)
    mod = importlib.import_module(f"opm.experiments.{exp.lower()}")
    mod.run(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
