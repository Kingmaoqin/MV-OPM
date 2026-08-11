"""Ignorability CATE baselines (spec Section 4): S/T/X/DR-learner + Causal Forest (econml),
and an in-repo R-learner.  All adjust for X only (no proxies).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.linear_model import Ridge

from ..dgp.base import CausalDataset
from .base import Baseline


def _reg(seed):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                         random_state=seed)


def _clf(seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                          random_state=seed)


class EconMLBaseline(Baseline):
    """Wraps an econml estimator; multi-arm CATE via one-vs-reference effects."""

    def __init__(self, name, kind, K, seed=0):
        self.name = name
        self.kind = kind
        self.K = K
        self.seed = seed

    def fit(self, ds: CausalDataset) -> "EconMLBaseline":
        from econml.dr import DRLearner
        from econml.dml import CausalForestDML
        from econml.metalearners import SLearner, TLearner, XLearner
        s = self.seed
        if self.kind == "S":
            est = SLearner(overall_model=_reg(s))
        elif self.kind == "T":
            est = TLearner(models=_reg(s))
        elif self.kind == "X":
            est = XLearner(models=_reg(s), propensity_model=_clf(s), cate_models=_reg(s))
        elif self.kind == "DR":
            est = DRLearner(model_propensity=_clf(s), model_regression=_reg(s),
                            model_final=Ridge(), cv=3, random_state=s)
        elif self.kind == "CF":
            est = CausalForestDML(model_y=_reg(s), model_t=_clf(s), discrete_treatment=True,
                                  n_estimators=300, random_state=s, cv=3)
        else:
            raise ValueError(self.kind)
        est.fit(ds.Y, ds.T, X=ds.X)
        self.est = est
        return self

    def predict_cate(self, X):
        cols = []
        for k in range(1, self.K):
            eff = self.est.effect(X, T0=0, T1=k)
            cols.append(np.asarray(eff).reshape(-1))
        return np.column_stack(cols)

    def ate_vector(self):
        raise RuntimeError("call ate_from(X)")

    def ate_from(self, X):
        return self.predict_cate(X).mean(axis=0)


class RLearner(Baseline):
    """In-repo R-learner (Robinson residualization), vectorized one-vs-reference (heuristic).

    Cross-fitted nuisances m(x)=E[Y|X], pi_k(x)=P(T=k|X). For each arm k, form the
    Robinson pseudo-outcome on the (arm-k vs reference) subpopulation and regress on X.
    Documented as a heuristic multi-arm extension (spec Section 4).
    """

    def __init__(self, name, K, seed=0, n_folds=3):
        self.name = name
        self.K = K
        self.seed = seed
        self.n_folds = n_folds

    def fit(self, ds: CausalDataset) -> "RLearner":
        from sklearn.model_selection import cross_val_predict
        X, T, Y = ds.X, ds.T, ds.Y
        n = ds.n
        m_hat = cross_val_predict(_reg(self.seed), X, Y, cv=self.n_folds)
        clf = _clf(self.seed)
        pi = cross_val_predict(clf, X, T, cv=self.n_folds, method="predict_proba")
        classes = np.unique(T)
        pi_full = np.full((n, self.K), 1e-3)
        for j, k in enumerate(classes):
            pi_full[:, int(k)] = pi[:, j]

        self.models = {}
        for k in range(1, self.K):
            # binary arm-k-vs-reference residualization
            sub = (T == k) | (T == 0)
            Xk = X[sub]
            Tk = (T[sub] == k).astype(float)
            res_y = Y[sub] - m_hat[sub]
            e = pi_full[sub, k] / (pi_full[sub, 0] + pi_full[sub, k] + 1e-8)  # P(k | k or 0)
            res_t = Tk - e
            w = res_t ** 2 + 1e-6
            pseudo = res_y * res_t / w
            reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                                random_state=self.seed + k)
            reg.fit(Xk, pseudo, sample_weight=w)
            self.models[k] = reg
        return self

    def predict_cate(self, X):
        return np.column_stack([self.models[k].predict(X) for k in range(1, self.K)])

    def ate_vector(self):
        raise RuntimeError("call ate_from(X)")

    def ate_from(self, X):
        return self.predict_cate(X).mean(axis=0)
