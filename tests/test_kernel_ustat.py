"""Spec 7.1: U-statistic loss equals a naive double-loop reference within 1e-6."""
import numpy as np
import torch

from opm.bridges.kernel_moment import (
    gaussian_kernel,
    positive_q_link,
    ustat_quadratic,
    vstat_quadratic,
)


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


def test_kernel_training_vstat_is_nonnegative_on_adversarial_direction():
    z = torch.tensor([[0.0], [1.0]])
    Kmat = gaussian_kernel(z, 1.0)
    for scale in (1.0, 10.0, 100.0):
        residual = torch.tensor([scale, -scale])
        assert float(ustat_quadratic(residual, Kmat)) < 0.0
        assert float(vstat_quadratic(residual, Kmat)) >= 0.0


def test_positive_q_link_represents_exact_dgp_values_below_one():
    q_true = torch.tensor([
        0.149765, 1.393577, 5.450256, 5.714657, 1.766135, 0.098341,
    ], dtype=torch.float64)
    q_min, q_max = 1e-4, 50.0
    raw = torch.log(torch.expm1(q_true - q_min))
    reconstructed = positive_q_link(raw, q_min=q_min, q_max=q_max)
    assert torch.allclose(reconstructed, q_true, atol=1e-10, rtol=1e-10)
    assert float(reconstructed.min()) < 1.0
