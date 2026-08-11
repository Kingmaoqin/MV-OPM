"""Doubly-robust pseudo-outcome (STAR) assembly (spec 1.4)."""
from __future__ import annotations

import numpy as np

from ..dgp.base import CausalDataset


def compute_phi(h_pred: np.ndarray, q_pred: np.ndarray, ds: CausalDataset) -> np.ndarray:
    """(STAR)  phi_k(O_i) = 1{T_i=k} q_k(V_i,X_i) (Y_i - h_k(W_i,X_i)) + h_k(W_i,X_i).

    h_pred, q_pred: (n, K). Returns phi: (n, K).  The h_k term is defined for ALL
    samples, not only those with T_i=k.
    """
    n, K = h_pred.shape
    Y = ds.Y.reshape(-1, 1)
    onehot = np.zeros((n, K))
    onehot[np.arange(n), ds.T] = 1.0
    residual = onehot * q_pred * (Y - h_pred)
    return residual + h_pred
