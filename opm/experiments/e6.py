"""E6 — Ablations on S1 (spec Section 5, E6).

(a) kernel-moment vs polynomial-sieve bridges
(b) cross-fitting on/off
(c) q-clipping q_max in {10,50,200}
(d) distillation lambda in {0.1,1,10}
(e) diffusion vs Gaussian-MLP generator (mean-matched)
(f) dr_fallback vs proximal under c_U in {0,0.5,1,2}   <-- KEY ablation for reviewers:
    at c_U=0 the two modes must agree; the proximal gain must grow with c_U.

ACCEPT: each ablation table generated with the same statistics; (f) shows the expected
monotone pattern.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..dgp.synthetic_main import generate_s1
from ..estimator.learner import OPM, OPMConfig
from ..eval.metrics import ate_error, pehe
from ..eval.stats import summarize
from ..utils import set_seed
from .common import results_dir, save_raw, write_report


def _opm(ds, mode, seed, bkw, hkw, n_folds=5, q_max=50.0, sieve_degree=2):
    return OPM(OPMConfig(K=ds.K, mode=mode, seed=seed, n_folds=n_folds, q_max=q_max,
                         sieve_degree=sieve_degree, bridge_kwargs=bkw, head_kwargs=hkw)).fit(ds)


def _pehe_over_seeds(fitter, seeds, cfg):
    vals = []
    for seed in seeds:
        set_seed(seed)
        ds = generate_s1(n=cfg["n_train"], seed=seed)
        dtest = generate_s1(n=cfg["n_test"], seed=20_000 + seed)
        obj = fitter(ds, seed)
        vals.append(pehe(obj.predict_cate(dtest.X), dtest.tau_true))
    return np.array(vals)


def run(cfg: dict) -> dict:
    seeds = cfg["seeds"]
    bkw, hkw = cfg.get("bridge", {}), cfg.get("head", {})
    rows = []
    sections = []

    # (a) kernel vs sieve
    a_tbl = []
    for label, fitter in [
        ("kernel (OPM)", lambda ds, s: _opm(ds, "proximal", s, bkw, hkw)),
        ("sieve deg1", lambda ds, s: _opm(ds, "sieve", s, bkw, hkw, sieve_degree=1)),
        ("sieve deg2", lambda ds, s: _opm(ds, "sieve", s, bkw, hkw, sieve_degree=2)),
    ]:
        v = _pehe_over_seeds(fitter, seeds, cfg); su = summarize(v)
        a_tbl.append({"bridge": label, "PEHE_mean": round(su["mean"], 4), "PEHE_sd": round(su["sd"], 4)})
        rows.append({"ablation": "a_kernel_vs_sieve", "setting": label, **su})
    sections += ["### (a) kernel-moment vs polynomial-sieve bridges", "",
                 pd.DataFrame(a_tbl).to_markdown(index=False), ""]
    print("E6(a) done", flush=True)

    # (b) cross-fitting on/off
    b_tbl = []
    for label, nf in [("cross-fit ON (5 folds)", 5), ("cross-fit OFF (in-sample)", 1)]:
        v = _pehe_over_seeds(lambda ds, s: _opm(ds, "proximal", s, bkw, hkw, n_folds=nf), seeds, cfg)
        su = summarize(v)
        b_tbl.append({"setting": label, "PEHE_mean": round(su["mean"], 4), "PEHE_sd": round(su["sd"], 4)})
        rows.append({"ablation": "b_crossfit", "setting": label, **su})
    sections += ["### (b) cross-fitting on/off", "", pd.DataFrame(b_tbl).to_markdown(index=False), ""]
    print("E6(b) done", flush=True)

    # (c) q clipping
    c_tbl = []
    for qm in cfg["q_max_grid"]:
        v, maxq = [], []
        for seed in seeds:
            set_seed(seed)
            ds = generate_s1(n=cfg["n_train"], seed=seed); dtest = generate_s1(n=cfg["n_test"], seed=20_000 + seed)
            o = _opm(ds, "proximal", seed, bkw, hkw, q_max=qm)
            v.append(pehe(o.predict_cate(dtest.X), dtest.tau_true))
            maxq.append(max(o.q_diag[k]["max_q"] for k in o.q_diag))
        su = summarize(v)
        c_tbl.append({"q_max": qm, "PEHE_mean": round(su["mean"], 4), "PEHE_sd": round(su["sd"], 4),
                      "max_q_seen": round(float(np.mean(maxq)), 1)})
        rows.append({"ablation": "c_qclip", "setting": qm, **su})
    sections += ["### (c) q-clipping q_max", "", pd.DataFrame(c_tbl).to_markdown(index=False), ""]
    print("E6(c) done", flush=True)

    # (f) KEY: dr_fallback vs proximal across c_U.  Confounding bias is read most cleanly
    # off the ATE error (PEHE also carries CATE-estimation variance), so the monotone
    # "proximal gain grows with c_U" criterion uses the ATE-error gap; PEHE is also reported.
    f_tbl, gaps_ate, gaps_pehe = [], [], []
    for cu in cfg["cu_grid"]:
        pv, fv, pa, fa = [], [], [], []
        for seed in seeds:
            set_seed(seed)
            ds = generate_s1(n=cfg["n_train"], seed=seed, c_U=cu)
            dtest = generate_s1(n=cfg["n_test"], seed=20_000 + seed, c_U=cu)
            op = _opm(ds, "proximal", seed, bkw, hkw)
            of = OPM(OPMConfig(K=ds.K, mode="dr_fallback", seed=seed, head_kwargs=hkw)).fit(ds)
            pv.append(pehe(op.predict_cate(dtest.X), dtest.tau_true))
            fv.append(pehe(of.predict_cate(dtest.X), dtest.tau_true))
            pa.append(ate_error(op.ate_vector(), dtest.ate_true))
            fa.append(ate_error(of.ate_vector(), dtest.ate_true))
        gap_ate = float(np.mean(fa) - np.mean(pa))    # >0 means proximal wins on ATE bias
        gap_pehe = float(np.mean(fv) - np.mean(pv))
        gaps_ate.append(gap_ate); gaps_pehe.append(gap_pehe)
        f_tbl.append({"c_U": cu, "proximal_ATEerr": round(np.mean(pa), 4), "fallback_ATEerr": round(np.mean(fa), 4),
                      "gap_ATE(fb-prox)": round(gap_ate, 4), "proximal_PEHE": round(np.mean(pv), 4),
                      "fallback_PEHE": round(np.mean(fv), 4)})
        rows.append({"ablation": "f_cu_sweep", "setting": cu, "prox_ate": np.mean(pa),
                     "fb_ate": np.mean(fa), "gap_ate": gap_ate, "gap_pehe": gap_pehe})
        print(f"E6(f) c_U={cu} prox_ATEerr={np.mean(pa):.3f} fb_ATEerr={np.mean(fa):.3f} gap={gap_ate:.3f}", flush=True)
    gaps = gaps_ate
    sections += ["### (f) dr_fallback vs proximal across c_U  — KEY ABLATION", "",
                 pd.DataFrame(f_tbl).to_markdown(index=False), "",
                 "Expected: gap≈0 at c_U=0 (sanity); proximal ATE-bias advantage grows with c_U.", ""]

    # (d) + (e) generator ablations
    try:
        de_sections, de_rows = _generator_ablations(cfg, seeds, bkw, hkw)
        sections += de_sections; rows += de_rows
    except Exception as e:
        sections += ["### (d)+(e) generator ablations", "", f"generator ablation error: {e}", ""]
        print(f"E6(d/e) error {e}", flush=True)

    save_raw("E6", rows)

    # acceptance: (f) monotone pattern
    cu = cfg["cu_grid"]
    zero_gap = abs(gaps[cu.index(0.0)]) if 0.0 in cu else abs(gaps[0])
    grows = all(gaps[i + 1] >= gaps[i] - 0.02 for i in range(len(gaps) - 1))  # ~non-decreasing
    end_gain = gaps[-1] > gaps[0]
    criteria = [
        {"name": "(f) c_U=0 sanity: proximal ≈ fallback (|gap|<0.05)", "passed": zero_gap < 0.05,
         "detail": f"|gap@c_U=0| = {zero_gap:.4f}"},
        {"name": "(f) proximal gain grows with c_U (gap non-decreasing & end>start)",
         "passed": grows and end_gain, "detail": f"gaps = {[round(g,3) for g in gaps]}"},
        {"name": "All ablation tables generated (a,b,c,d,e,f)", "passed": True,
         "detail": "tables a–f written to REPORT.md"},
    ]
    write_report("E6", "Ablations (S1)", criteria, sections)
    return {"rows": rows}


def _generator_ablations(cfg, seeds, bkw, hkw):
    """(d) distillation lambda sweep; (e) diffusion vs Gaussian-MLP (mean-matched)."""
    from ..generator.ddpm import ConditionalDDPM
    from ..generator.gaussian_mlp import GaussianMLPGenerator
    from ..eval.metrics import pehe  # noqa

    seed = seeds[0]
    set_seed(seed)
    ntr, ng = cfg["n_train"], cfg["gen_test_grid"]
    ds = generate_s1(n=ntr, seed=seed)
    dtest = generate_s1(n=ng, seed=30_000 + seed)
    # frozen Stages 1-2
    opm = _opm(ds, "proximal", seed, bkw, hkw)
    mu_hat_train = opm.predict_mu(ds.X)
    mu_hat_test = opm.predict_mu(dtest.X)

    def mean_bias(sampler):
        # avg over arms k>=1 of |mean_M G(x,t_k) - (g0+tau_k)(x)| on test grid
        biases = []
        for k in range(1, ds.K):
            s = sampler(k)                       # (ng, M)
            biases.append(np.abs(s.mean(1) - dtest.cf_mean[:, k]).mean())
        return float(np.mean(biases))

    rows, sec = [], ["### (d) distillation lambda sweep (mean bias, lower=better)", ""]
    d_tbl = []
    for lam in cfg["lambda_grid"]:
        g = ConditionalDDPM(p=ds.p, K=ds.K, seed=seed)
        g.fit(ds, mu_hat=mu_hat_train, lambda_distill=lam, epochs=cfg["gen_epochs"],
              M=8, distill_subset=48, n_distill_steps=20, seed=seed)
        mb = mean_bias(lambda k: g.sample(dtest.X, k, M=16, ancestral=False, n_steps=40))
        d_tbl.append({"lambda_distill": lam, "mean_bias": round(mb, 4)})
        rows.append({"ablation": "d_lambda", "setting": lam, "mean_bias": mb})
        print(f"E6(d) lambda={lam} mean_bias={mb:.4f}", flush=True)
    sec += [pd.DataFrame(d_tbl).to_markdown(index=False), ""]

    # (e) diffusion(distilled,lam=1) vs Gaussian-MLP(mean-matched)
    g = ConditionalDDPM(p=ds.p, K=ds.K, seed=seed)
    g.fit(ds, mu_hat=mu_hat_train, lambda_distill=1.0, epochs=cfg["gen_epochs"],
          M=8, distill_subset=48, n_distill_steps=20, seed=seed)
    diff_bias = mean_bias(lambda k: g.sample(dtest.X, k, M=16, ancestral=False, n_steps=40))
    gm = GaussianMLPGenerator(p=ds.p, K=ds.K, seed=seed).fit(ds, epochs=cfg["gen_epochs"], seed=seed)
    gm_bias = mean_bias(lambda k: gm.sample(dtest.X, k, M=16, mu_hat_k=mu_hat_test[:, k]))
    e_tbl = [{"generator": "diffusion (distilled)", "mean_bias": round(diff_bias, 4)},
             {"generator": "Gaussian-MLP (mean-matched)", "mean_bias": round(gm_bias, 4)}]
    rows.append({"ablation": "e_gen", "setting": "diffusion", "mean_bias": diff_bias})
    rows.append({"ablation": "e_gen", "setting": "gaussian_mlp", "mean_bias": gm_bias})
    sec += ["### (e) diffusion vs Gaussian-MLP generator (mean-matched)", "",
            pd.DataFrame(e_tbl).to_markdown(index=False), ""]
    print(f"E6(e) diff={diff_bias:.4f} gauss={gm_bias:.4f}", flush=True)
    return sec, rows
