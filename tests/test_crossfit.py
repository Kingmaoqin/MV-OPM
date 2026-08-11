"""Spec 7.5: cross-fitting is leakage-proof (planted-marker bookkeeping test)."""
import numpy as np

from opm.estimator.crossfit import CrossFitter, make_folds


def test_crossfit_no_leakage():
    n, L = 137, 5
    cf = CrossFitter(n, L=L, seed=3)
    predicted = np.zeros(n, dtype=int)
    for train_idx, test_idx in cf:
        # planted marker: no test index may appear in its own training set
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert set(train_idx).isdisjoint(set(test_idx))
        predicted[test_idx] += 1
    # every index predicted exactly once, across a disjoint cover
    assert np.all(predicted == 1)
    assert cf.check_no_leakage()


def test_folds_partition():
    n, L = 200, 5
    folds = make_folds(n, L, seed=0)
    allidx = np.concatenate(folds)
    assert len(allidx) == n
    assert len(np.unique(allidx)) == n
