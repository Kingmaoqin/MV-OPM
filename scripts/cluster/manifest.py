"""Build the frozen MV-OPM experiment manifest (one row = study x config x seed).

    python scripts/cluster/manifest.py pilot   # 10-20 dev seeds
    python scripts/cluster/manifest.py final    # full (or scaled) confirmatory seeds

Writes results/mvopm/manifest_<phase>.jsonl. Each row carries an EXPLICIT scientific seed (never
derived only from an array index) + all params + an out_path. Dev and final seed banks are
disjoint (dev = 0..; final = 100000.. by construction).
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "mvopm")

# scaled seed counts: (pilot, final). Final scaled to be completable on a shared host; the raw
# per-run files support extending later. Kept honest — reported scale = achieved scale.
COUNTS = {
    "A": (20, 200), "B": (12, 30), "C": (12, 30), "D": (12, 30), "E": (10, 24),
    "F": (10, 24), "G": (10, 24), "I": (12, 30), "J": (30, 30),
}


def _seeds(study, phase):
    k = COUNTS[study][0 if phase == "pilot" else 1]
    base = 0 if phase == "pilot" else 100000
    return list(range(base, base + k))


def _cfg(phase):
    # candidate-fit config (kernel epochs etc.); pilot lighter than final
    if phase == "pilot":
        return dict(n_folds=3, kind="rff_l2", n_rff=160,
                    bridge=dict(max_epochs=45, patience=15, eval_every=3, batch_size=512),
                    head=dict(max_epochs=80, patience=10))
    return dict(n_folds=4, kind="rff_l2", n_rff=200,
                bridge=dict(max_epochs=70, patience=20, eval_every=3, batch_size=1024),
                head=dict(max_epochs=120, patience=12))


def build(phase: str):
    cfg = _cfg(phase)
    rows = []

    def add(**kw):
        kw["cfg"] = cfg
        rows.append(kw)

    # Study A — exact product mechanism (cheap; corruption grid x n)
    for seed in _seeds("A", phase):
        for r_h in [0.0, 0.2, 0.5]:
            for r_q in [0.0, 0.2, 0.5]:
                for n in ([2000, 8000] if phase == "final" else [4000]):
                    add(study="A", dispatch="mechanism", seed=seed, r_h=r_h, r_q=r_q, n=n, c_U=1.0)

    n_sel = 2000 if phase == "final" else 1500
    # Study B/C/D/E — candidate selection across scenarios
    for study, scen in [("B", "S1"), ("C", "S2"), ("D", "nonlinear"), ("E", "hamd")]:
        for seed in _seeds(study, phase):
            add(study=study, dispatch="selection", scenario=scen, n=n_sel, seed=seed, c_U=1.0)
    # Study F — hidden-confounding sweep (S1)
    for seed in _seeds("F", phase):
        for cU in [0.0, 0.5, 1.0, 2.0]:
            add(study="F", dispatch="selection", scenario="S1", n=n_sel, seed=seed, c_U=cU)
    # Study G — proxy corruption (S1)
    for seed in _seeds("G", phase):
        for kind in ["additive", "missing"]:
            for lvl in [0.0, 0.1, 0.2, 0.3]:
                add(study="G", dispatch="selection", scenario="S1", n=n_sel, seed=seed, c_U=1.0,
                    mod={"kind": kind, "level": lvl})
    # Study I — broken proxy / abstention (S1)
    for seed in _seeds("I", phase):
        for kind in ["none", "shuffle_W", "noise_V", "weaken"]:
            mod = None if kind == "none" else {"kind": kind, "level": 0.7 if kind == "weaken" else 0.0}
            add(study="I", dispatch="selection", scenario="S1", n=n_sel, seed=seed, c_U=1.0, mod=mod)
    # Study J — RHC real data (repeated splits; no oracle)
    for seed in _seeds("J", phase):
        add(study="J", dispatch="rhc", seed=seed)

    for r in rows:
        m = r.get("mod") or {}
        cfgtag = f"{r.get('scenario','_')}_cU{r.get('c_U','_')}_{m.get('kind','_')}{m.get('level','')}" \
                 f"_rh{r.get('r_h','_')}rq{r.get('r_q','_')}_n{r.get('n','_')}"
        r["out_path"] = os.path.join(OUT, r["study"], cfgtag, f"seed{r['seed']}", "result.json")
    return rows


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    rows = build(phase)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"manifest_{phase}.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{phase}: {len(rows)} manifest rows -> {path}")
    from collections import Counter
    print("per-study:", dict(Counter(r["study"] for r in rows)))


if __name__ == "__main__":
    main()
