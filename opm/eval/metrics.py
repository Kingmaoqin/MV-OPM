"""Causal metrics (spec 6.2)."""
from __future__ import annotations

import numpy as np


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """sqrt( mean_i mean_{k>=1} (tau_hat_k(x_i) - tau_k(x_i))^2 )."""
    tau_hat = tau_hat.reshape(tau_true.shape)
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


def ate_error(ate_hat: np.ndarray, ate_true: np.ndarray) -> float:
    """mean_{k>=1} | ATE_hat_k - ATE_k |."""
    return float(np.mean(np.abs(np.asarray(ate_hat) - np.asarray(ate_true))))


def policy_value_synth(mu_hat: np.ndarray, mu_true: np.ndarray) -> float:
    """Value of policy argmax_k mu_hat_k(x) evaluated under TRUE potential outcomes.

    Higher is better.  mu_hat, mu_true: (n, K).
    """
    choice = np.argmax(mu_hat, axis=1)
    return float(mu_true[np.arange(mu_true.shape[0]), choice].mean())


def oracle_policy_value(mu_true: np.ndarray) -> float:
    return float(mu_true.max(axis=1).mean())
