"""Seeding, device, and small numeric helpers shared across the package."""
from __future__ import annotations

import os
import random
from typing import Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python / numpy / torch and (optionally) force deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuBLAS determinism guard (needed before use_deterministic_algorithms on CUDA).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: Optional[str]) -> torch.device:
    """Resolve a device string; default CPU (A100s here are shared and near-full)."""
    if device is None or device == "auto":
        return torch.device("cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def standardize_fit(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-feature mean/std (std floored at 1e-8) for standardization."""
    mean = a.mean(axis=0)
    std = a.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def standardize_apply(a: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (a - mean) / std


def median_bandwidth(z: np.ndarray, max_pairs: int = 4000, seed: int = 0) -> float:
    """Median-heuristic bandwidth: median pairwise Euclidean distance of ``z``.

    Uses a random subsample of pairs when n is large to keep this O(max_pairs).
    """
    n = z.shape[0]
    rng = np.random.default_rng(seed)
    if n * (n - 1) // 2 <= max_pairs:
        idx_i, idx_j = np.triu_indices(n, k=1)
    else:
        idx_i = rng.integers(0, n, size=max_pairs)
        idx_j = rng.integers(0, n, size=max_pairs)
        keep = idx_i != idx_j
        idx_i, idx_j = idx_i[keep], idx_j[keep]
    d = np.linalg.norm(z[idx_i] - z[idx_j], axis=1)
    med = float(np.median(d))
    return med if med > 1e-8 else 1.0
