"""Scenario + proxy-modification registry for MV-OPM studies."""
from __future__ import annotations

import numpy as np

from ...dgp import finite_proxy_exact as fpe
from ...dgp import nonlinear_bridge as nlb
from ...dgp import synthetic_main as sm
from ...dgp.semisynth import generate_semisynth
from ...data.hamd import hamd_covariates

_HAMD_X = None


def _hamd(n, seed, c_U=1.0):
    global _HAMD_X
    if _HAMD_X is None:
        _HAMD_X = hamd_covariates()
    X = _HAMD_X
    if n and n < X.shape[0]:
        idx = np.random.default_rng(seed).choice(X.shape[0], n, replace=False)
        X = X[idx]
    return generate_semisynth(X, seed=seed, c_U=c_U)


SCENARIOS = {
    "exact": lambda n, seed, c_U=1.0: fpe.generate(n, seed, c_U=c_U),
    "S1": lambda n, seed, c_U=1.0: sm.generate_s1(n, seed, c_U=c_U),
    "S2": lambda n, seed, c_U=1.0: sm.generate_s2(n, seed, c_U=c_U),
    "nonlinear": lambda n, seed, c_U=1.0: nlb.generate(n, seed),
    "hamd": _hamd,
}


def generate(scenario: str, n: int, seed: int, c_U: float = 1.0):
    return SCENARIOS[scenario](n, seed, c_U=c_U)


def apply_proxy_mod(ds, mod: dict, seed: int):
    """Modify proxies in place for corruption (Study G) or breakage/abstention (Study I).
    mod = {'kind': additive|missing|shift|heavy_tail|shuffle_W|shuffle_V|noise_W|noise_V|weaken, 'level': s}
    """
    if not mod:
        return ds
    kind, s = mod.get("kind"), mod.get("level", 0.0)
    rng = np.random.default_rng(seed + 4242)
    W, V = ds.W.copy(), ds.V.copy()
    if kind in ("additive", "missing", "shift", "heavy_tail"):
        from ...dgp.synthetic_main import corrupt_proxies
        W, V = corrupt_proxies(W, V, kind, s, seed)
    elif kind == "shuffle_W":
        W = W[rng.permutation(len(W))]
    elif kind == "shuffle_V":
        V = V[rng.permutation(len(V))]
    elif kind == "noise_W":
        W = rng.normal(W.mean(0), W.std(0) + 1e-6, size=W.shape)
    elif kind == "noise_V":
        V = rng.normal(V.mean(0), V.std(0) + 1e-6, size=V.shape)
    elif kind == "weaken":                       # blend proxy toward its mean (weaken U signal)
        W = (1 - s) * W + s * W.mean(0, keepdims=True) + rng.normal(0, s * (W.std() + 1e-6), W.shape)
        V = (1 - s) * V + s * V.mean(0, keepdims=True) + rng.normal(0, s * (V.std() + 1e-6), V.shape)
    ds.W, ds.V = W, V
    return ds
