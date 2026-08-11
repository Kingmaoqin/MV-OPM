"""Oracle-only evaluation utilities (MV-OPM §3/§8).

Truth (tau_true/mu_true/ate_true/potential outcomes) lives HERE. The selection package
(`opm.validation.*`, `opm.candidates.*`) must NOT import this module. A unit test asserts the
selector cannot reach oracle attributes.
"""
from __future__ import annotations

import numpy as np

from ..dgp.base import CausalDataset


def has_truth(ds: CausalDataset) -> bool:
    return ds.tau_true is not None


def pehe(tau_hat: np.ndarray, ds: CausalDataset) -> float:
    t = ds.tau_true.reshape(tau_hat.shape[0], -1)
    return float(np.sqrt(np.mean((tau_hat.reshape(t.shape) - t) ** 2)))


def ate_error(ate_hat: np.ndarray, ds: CausalDataset) -> float:
    return float(np.mean(np.abs(np.asarray(ate_hat).reshape(-1) - ds.ate_true.reshape(-1))))


def candidate_causal_error(tau_hat, ate_hat, ds: CausalDataset) -> dict:
    """Oracle causal error of a fitted candidate (simulation only)."""
    return {"pehe": pehe(tau_hat, ds), "ate_err": ate_error(ate_hat, ds)}
