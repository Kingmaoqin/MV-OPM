"""Spec 7.7: with c_U=0 (no unobserved confounding) dr_fallback and proximal agree."""
import numpy as np

from opm.dgp.synthetic_main import generate_s1
from opm.estimator.learner import OPM, OPMConfig
from opm.utils import set_seed


def test_dr_fallback_equals_proximal_when_no_confounding():
    set_seed(2)
    ds = generate_s1(n=3000, seed=2, c_U=0.0)
    prox = OPM(OPMConfig(K=4, mode="proximal", seed=2,
                         bridge_kwargs=dict(max_epochs=80, patience=24, eval_every=3),
                         head_kwargs=dict(max_epochs=150, patience=15))).fit(ds)
    fb = OPM(OPMConfig(K=4, mode="dr_fallback", seed=2,
                       head_kwargs=dict(max_epochs=150, patience=15))).fit(ds)
    for k in (1, 2, 3):
        a, b = prox.ate[k], fb.ate[k]
        pooled = np.sqrt(a["se"] ** 2 + b["se"] ** 2)
        assert abs(a["ate"] - b["ate"]) < 2 * pooled, (k, a, b)
