"""Spec 7.2 & 7.3: closed-form bridge recovery and q moment identity (DGP 3.1)."""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from opm.dgp import linear_gaussian as lg
from opm.estimator.learner import OPM, OPMConfig
from opm.utils import set_seed


@pytest.fixture(scope="module")
def fitted_opm():
    set_seed(0)
    ds = lg.generate(n=8000, seed=0)
    opm = OPM(OPMConfig(K=2, mode="proximal", seed=0,
                        bridge_kwargs=dict(max_epochs=120, patience=30, eval_every=2),
                        head_kwargs=dict(max_epochs=200, patience=20))).fit(ds)
    return ds, opm


def test_bridge_closed_form(fitted_opm):
    ds, opm = fitted_opm
    # |ATE_hat - 2| < 0.15
    assert abs(opm.ate[1]["ate"] - 2.0) < 0.15, opm.ate[1]

    # linear probe of h_hat on (W, X_1..5, T) recovers true coefficients (R^2 > 0.95)
    W, X = ds.W[:, 0], ds.X
    feats, targets = [], []
    for k in (0, 1):
        feats.append(np.column_stack([W, X, np.full(len(W), k)]))
        targets.append(opm.h_oof[:, k])
    feats, targets = np.vstack(feats), np.concatenate(targets)
    r2 = LinearRegression().fit(feats, targets).score(feats, targets)
    assert r2 > 0.95, r2


def test_q_moment(fitted_opm):
    _, opm = fitted_opm
    for k in (0, 1):
        m = opm.q_diag[k]["En_1Tk_qk"]
        assert 0.9 <= m <= 1.1, (k, m)
