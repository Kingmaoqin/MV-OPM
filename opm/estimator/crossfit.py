"""Leakage-proof cross-fitting bookkeeping (spec 1.6; unit-tested in Section 7.5)."""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np


def make_folds(n: int, L: int, seed: int) -> List[np.ndarray]:
    """Partition [0, n) into L disjoint folds (shuffled, reproducible)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(perm, L)]


class CrossFitter:
    """Yields (train_idx, test_idx) with the guarantee test_idx ∩ train_idx = ∅.

    ``assignments`` records, for each sample, which fold predicted it — used by the
    no-leakage unit test to assert every prediction index was absent from its own
    training folds.
    """

    def __init__(self, n: int, L: int = 5, seed: int = 0):
        self.n = n
        self.L = L
        self.folds = make_folds(n, L, seed)
        self.assignments = np.full(n, -1, dtype=np.int64)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        all_idx = np.arange(self.n)
        for l, test_idx in enumerate(self.folds):
            mask = np.ones(self.n, dtype=bool)
            mask[test_idx] = False
            train_idx = all_idx[mask]
            self.assignments[test_idx] = l
            yield train_idx, test_idx

    def check_no_leakage(self) -> bool:
        """Every index predicted exactly once and never in its own training set."""
        if np.any(self.assignments < 0):
            return False
        seen = np.concatenate([f for f in self.folds])
        return len(np.unique(seen)) == self.n and len(seen) == self.n
