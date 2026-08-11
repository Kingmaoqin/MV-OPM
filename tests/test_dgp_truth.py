"""Spec 7.6: oracle tau -> PEHE 0; S1 counterfactual law matches a Monte-Carlo of the DGP."""
import numpy as np
from scipy import stats as sps

from opm.dgp import synthetic_main as sm
from opm.dgp.synthetic_main import BETA_U, CF_VAR, _g0, generate_s1
from opm.eval.metrics import pehe


def test_oracle_pehe_zero():
    ds = generate_s1(n=1000, seed=0)
    assert pehe(ds.tau_true, ds.tau_true) < 1e-12


def test_s1_counterfactual_law_matches_mc():
    # Fix one covariate vector; Monte-Carlo Y(t_k) directly from the DGP constants.
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1, 10))
    g0x = float(_g0(x)[0])
    # arm k=1: tau_1(x) = 1 + 0.6 x_1
    tau1 = 1.0 + 0.6 * x[0, 0]
    N = 8000
    U = rng.normal(size=(N, 2))
    eps_y = rng.normal(size=N)
    y_cf = g0x + tau1 + U @ BETA_U + eps_y            # Y(t_1) | X=x samples

    # claimed law: N(g0 + tau_1, 1 + ||beta_U||^2)
    mean = g0x + tau1
    sd = np.sqrt(CF_VAR)
    ks = sps.kstest(y_cf, "norm", args=(mean, sd))
    assert ks.pvalue > 0.01, ks
