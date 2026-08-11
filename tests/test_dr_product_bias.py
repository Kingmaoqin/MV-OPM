"""Spec 7.4: double-robustness smoke test — h_true + corrupted q keeps ATE error < 0.15."""
import numpy as np

from opm.dgp import linear_gaussian as lg
from opm.estimator.pseudo_outcome import compute_phi
from opm.utils import set_seed


def test_dr_product_bias():
    set_seed(1)
    ds = lg.generate(n=8000, seed=1)
    # h fixed to the exact closed-form bridge for both arms
    h = np.column_stack([lg.h_true(ds.W, ds.X, np.zeros(ds.n, int)),
                         lg.h_true(ds.W, ds.X, np.ones(ds.n, int))])
    # deliberately corrupted q (wrong, but bounded); DR => ATE stays correct
    rng = np.random.default_rng(0)
    q_corr = np.clip(1.0 + rng.uniform(0, 4, size=(ds.n, 2)), 1.0, 50.0)
    phi = compute_phi(h, q_corr, ds)
    ate = float((phi[:, 1] - phi[:, 0]).mean())
    assert abs(ate - 2.0) < 0.15, ate
