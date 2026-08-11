"""Spec 7.1: U-statistic loss equals a naive double-loop reference within 1e-6."""
import numpy as np
import torch

from opm.bridges.kernel_moment import gaussian_kernel, ustat_quadratic


def test_kernel_ustat_matches_naive():
    torch.manual_seed(0)
    B, d = 32, 4
    z = torch.randn(B, d)
    R = torch.randn(B)
    bw = 1.3
    Kmat = gaussian_kernel(z, bw)

    fast = float(ustat_quadratic(R, Kmat))

    naive = 0.0
    for i in range(B):
        for j in range(B):
            if i != j:
                naive += float(R[i]) * float(R[j]) * float(Kmat[i, j])
    naive /= B * (B - 1)

    assert abs(fast - naive) < 1e-6, (fast, naive)
