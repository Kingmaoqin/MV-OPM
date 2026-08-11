"""Proximal baselines (spec Section 4): POR / PIPW / PDR (sieve) and NMMR (kernel-h).

PDR reuses the (STAR) pseudo-outcome with sieve-estimated bridges — this doubles as the
kernel-vs-sieve ablation (E6a).  NMMR (Kompa et al. 2022) is realized as a kernel-moment
outcome-bridge POR estimator (neural bridge + kernel MMR loss, no adversary), ATE-oriented.
"""
from __future__ import annotations

from functools import partial

from ..bridges.kernel_moment import BridgeConfig, KernelBridges
from ..bridges.sieve import SieveBridges
from .base import CrossFitProximal


def sieve_factory(K, degree, q_max):
    def make(seed):
        return SieveBridges(K=K, degree=degree, q_max=q_max)
    return make


def kernel_factory(K, q_max, device, bridge_kwargs):
    def make(seed):
        return KernelBridges(BridgeConfig(K=K, q_max=q_max, device=device, seed=seed,
                                          **(bridge_kwargs or {})))
    return make


def make_proximal_baseline(kind, K, seed=0, device="cpu", degree=1, q_max=50.0,
                           bridge_kwargs=None, head_kwargs=None):
    """kind in {POR, PIPW, PDR, POR_sieve, NMMR}. degree=1 (linear sieve) is the stable
    default and matches the E4 linear-POR milestone gate; E6a sweeps degree explicitly."""
    hk = head_kwargs or {}
    if kind in ("POR", "POR_sieve"):
        return CrossFitProximal("POR-sieve", sieve_factory(K, degree, q_max), "por",
                                K, seed=seed, device=device, head_kwargs=hk)
    if kind == "PIPW":
        return CrossFitProximal("PIPW-sieve", sieve_factory(K, degree, q_max), "pipw",
                                K, seed=seed, device=device, head_kwargs=hk)
    if kind == "PDR":
        return CrossFitProximal("PDR-sieve", sieve_factory(K, degree, q_max), "pdr",
                                K, seed=seed, device=device, head_kwargs=hk)
    if kind == "NMMR":
        return CrossFitProximal("NMMR", kernel_factory(K, q_max, device, bridge_kwargs),
                                "por", K, seed=seed, device=device, head_kwargs=hk)
    raise ValueError(kind)
