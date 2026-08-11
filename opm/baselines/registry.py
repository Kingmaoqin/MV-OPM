"""Registry mapping baseline names -> constructors, with honest 'did not run' stubs.

Every baseline exposes fit(ds) then predict_cate(X); ATE is taken as the sample mean of
predict_cate(X_eval) (or ate_vector() for OPM-style estimators).  Baselines that are not
reimplemented in this build are listed in NOT_RUN with a reason (spec Section 9: never
fabricate results for missing baselines).
"""
from __future__ import annotations

from typing import Callable, Dict

from .meta_learners import EconMLBaseline, RLearner
from .neural import DragonNet, TARNet
from .proximal import make_proximal_baseline

# Ignorability baselines (X only)
IGNORABILITY = ["S-learner", "T-learner", "X-learner", "DR-learner", "CausalForest",
                "R-learner", "TARNet", "DragonNet"]
# Proximal baselines
PROXIMAL = ["POR-sieve", "PIPW-sieve", "PDR-sieve", "NMMR"]

# Baselines intentionally not reimplemented in this build (honest bookkeeping).
NOT_RUN = {
    "DFPV": "Deep Feature Proximal Variables (Xu et al. 2021): two-stage deep-feature IV "
            "not reimplemented; the neural-proximal ATE role is covered by NMMR.",
    "P-learner": "Sverdrup & Cui 2023 proximal learner: its Neyman-orthogonal proximal-DR "
                 "objective for CATE reduces, in this framework, to regressing the (STAR) "
                 "contrast phi_1-phi_0 on X — exactly PDR-sieve/OPM. A separate module would "
                 "duplicate it, so it is represented by PDR-sieve (proximal-DR) and NMMR "
                 "(proximal-outcome, kernel).",
    "DiffPO": "Generative proximal baseline; see E7 factual-DDPM comparator instead "
              "(DiffPO public code not integrated in this build).",
}


def build(name: str, K: int, seed: int = 0, device: str = "cpu",
          bridge_kwargs=None, head_kwargs=None):
    hk = head_kwargs or {}
    s = seed
    econ = {"S-learner": "S", "T-learner": "T", "X-learner": "X",
            "DR-learner": "DR", "CausalForest": "CF"}
    if name in econ:
        return EconMLBaseline(name, econ[name], K, seed=s)
    if name == "R-learner":
        return RLearner(name, K, seed=s)
    if name == "TARNet":
        return TARNet(K=K, seed=s, device=device)
    if name == "DragonNet":
        return DragonNet(K=K, seed=s, device=device)
    if name in ("POR-sieve", "PIPW-sieve", "PDR-sieve", "NMMR"):
        kind = {"POR-sieve": "POR", "PIPW-sieve": "PIPW", "PDR-sieve": "PDR", "NMMR": "NMMR"}[name]
        return make_proximal_baseline(kind, K, seed=s, device=device,
                                      bridge_kwargs=bridge_kwargs, head_kwargs=hk)
    raise ValueError(f"unknown baseline {name}")
