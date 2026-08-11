"""Common candidate bridge interface + library (MV-OPM §1.1).

A *solver* estimates both an h and a q (kernel/MMR, or degree-d sieve). A *candidate* pairs an
h from one solver with a q from another; mixed candidates (kernel h + sieve1 q, ...) let the
validator discriminate across solver families. For efficiency each UNIQUE solver is fit once per
data view and candidates are compositions of the cached fitted solvers.

All candidates expose the identical interface: ``predict_h(W,X) -> (n,K)``, ``predict_q(V,X) -> (n,K)``.
Candidate order is fixed (sorted by name) so candidate ordering never affects results
(enforced by test_candidate_order_invariance).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from ..bridges.kernel_moment import BridgeConfig, KernelBridges
from ..bridges.sieve import SieveBridges

# candidate name -> (h_solver, q_solver)
CANDIDATE_SPECS: Dict[str, Tuple[str, str]] = {
    "kernel_kernel": ("kernel", "kernel"),
    "sieve1_sieve1": ("sieve1", "sieve1"),
    "sieve2_sieve2": ("sieve2", "sieve2"),
    "sieve3_sieve3": ("sieve3", "sieve3"),
    "kernel_sieve1": ("kernel", "sieve1"),
    "sieve1_kernel": ("sieve1", "kernel"),
}
CANDIDATE_NAMES: List[str] = sorted(CANDIDATE_SPECS)          # FIXED order (sorted)


def _make_solver(name: str, K: int, seed: int, device: str, bridge_kwargs: dict):
    if name == "kernel":
        return KernelBridges(BridgeConfig(K=K, device=device, seed=seed, **(bridge_kwargs or {})))
    if name.startswith("sieve"):
        deg = int(name[len("sieve"):])
        return SieveBridges(K=K, degree=deg)
    raise ValueError(f"unknown solver {name}")


class _FittedCandidate:
    """A candidate = (fitted h-solver, fitted q-solver). Uniform predict interface."""

    def __init__(self, name: str, h_solver, q_solver):
        self.name = name
        self._h = h_solver
        self._q = q_solver

    def predict_h(self, W, X) -> np.ndarray:
        return self._h.predict_h(W, X)

    def predict_q(self, V, X) -> np.ndarray:
        return self._q.predict_q(V, X)


class CandidateSet:
    """Fits every unique solver ONCE on a view, then exposes the 6 composed candidates.

    ``candidate_names`` restricts the set (e.g., pilot on a subset); default = all, fixed order.
    """

    def __init__(self, K: int, seed: int = 0, device: str = "cpu", bridge_kwargs: dict = None,
                 candidate_names: List[str] = None):
        self.K = K
        self.seed = seed
        self.device = device
        self.bridge_kwargs = bridge_kwargs or {}
        self.names = list(candidate_names) if candidate_names else list(CANDIDATE_NAMES)
        self.names = sorted(self.names)                       # enforce fixed order
        self._fitted = False

    def fit(self, view) -> "CandidateSet":
        needed = sorted({s for nm in self.names for s in CANDIDATE_SPECS[nm]})
        self._solvers = {}
        for si, sname in enumerate(needed):                   # deterministic per-solver seed
            ssd = self.seed + 17 * si + 1
            torch.manual_seed(ssd)                            # make net init reproducible &
            np.random.seed(ssd)                               # candidate-order-invariant (torch
            self._solvers[sname] = _make_solver(sname, self.K, ssd,   # global RNG isolation)
                                                self.device, self.bridge_kwargs).fit(view)
        self.candidates = {nm: _FittedCandidate(nm, self._solvers[CANDIDATE_SPECS[nm][0]],
                                                self._solvers[CANDIDATE_SPECS[nm][1]])
                           for nm in self.names}
        self._fitted = True
        return self

    def __iter__(self):
        for nm in self.names:                                 # fixed (sorted) order
            yield nm, self.candidates[nm]

    def get(self, name: str) -> _FittedCandidate:
        return self.candidates[name]
