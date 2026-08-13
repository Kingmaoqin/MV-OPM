"""Frozen 20-cell oracle-switching regime matrix."""
from __future__ import annotations

from itertools import product

from ...dgp.bridge_complexity import FAMILIES


def frozen_regimes() -> list[dict]:
    core = [
        {
            "regime_id": f"core_{family}_n{n}_{noise}",
            "family": family,
            "n": n,
            "proxy_noise": noise,
            "p": 10,
            "c_U": 1.0,
            "candidate_names": None,
        }
        for family, n, noise in product(FAMILIES, (1500, 6000), ("low", "high"))
    ]
    stress = [
        {
            "regime_id": f"stress_{family}_n3000_low_p40",
            "family": family,
            "n": 3000,
            "proxy_noise": "low",
            "p": 40,
            "c_U": 1.0,
            # Degree-2/3 explicit polynomial matrices are computationally inapplicable at p=40.
            "candidate_names": ["kernel_kernel", "kernel_sieve1", "sieve1_kernel", "sieve1_sieve1"],
        }
        for family in FAMILIES
    ]
    rows = core + stress
    if len(rows) != 20 or len({r["regime_id"] for r in rows}) != 20:
        raise RuntimeError("frozen regime matrix invariant failed")
    return rows
