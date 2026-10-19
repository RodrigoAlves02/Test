#!/usr/bin/env python
"""Are the generation tolerances tighter than they need to be?

The labels are solved with tolerances 10-1000x tighter than ptr's defaults, because the defaults
were measured to stop the fuel descent about 0.2 % early. That measurement was made on ONE case and
never weighed against what the tightening costs. This answers both halves across a dispersion:

    how much does the ANSWER move   (fuel, tf -- the things the labels actually are)
    how much does it COST           (iterations, wall, and whether the case converges at all)

THE REJECTED SEEDS ARE THE POINT. A case that stalls at iters = 200 takes ~70 min against a healthy
7, and because a task launches 48 cases and waits for all of them, one straggler holds a whole node
for an extra hour -- about 31 % of the generation wall clock went to ~1 % of the cases. If a looser
tolerance converges them, the saving is far larger than the per-case speedup suggests.

iters is held at 200 for every level so this measures the TOLERANCE and not the budget.

Usage:  tol_test.py <nominal.npz> <out.json> <seed> [seed ...]
"""
from __future__ import annotations

import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

if hasattr(os, "sched_getaffinity"):
    _a = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {_a[int(os.environ.get("MC_SLOT", "0")) % len(_a)]})

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pinn_case as PC                                                   # noqa: E402

LEVELS = {
    "tight":   dict(iters=200, tol_obj=1e-7, tol_defect_m=1e-3, tol_defect_v=1e-4, tol_vc_m=1e-4),
    "mid":     dict(iters=200, tol_obj=1e-5, tol_defect_m=3e-3, tol_defect_v=3e-4, tol_vc_m=3e-4),
    "default": dict(iters=200, tol_obj=1e-4, tol_defect_m=5e-3, tol_defect_v=5e-4, tol_vc_m=1e-3),
}


class A:
    pos_sigma = 1000.0; vel_sigma = 1.0; mass_sigma = 2.0; point_sigma_deg = 3.0
    seed = 0


def run_one(seed, nom, level):
    A.seed = seed
    r0, v0, m0, d0 = PC.draw(A, nom)
    saved = dict(PC.TOL)
    PC.TOL.clear(); PC.TOL.update(LEVELS[level])
    t0 = time.perf_counter()
    try:
        r1, r2, sc, P, f2 = PC.solve(r0, v0, m0, 100, d_init=d0)
        out = dict(level=level, seed=int(seed), wall_s=time.perf_counter() - t0,
                   s1=str(r1.status), s2=str(r2.status),
                   iters=int(r1.iters) + int(r2.iters), iters_free=int(f2.iters),
                   fuel=float(r2.fuel_kg), tf=float(r2.tf_s),
                   D=float(r1.D_nd) * sc.L, miss=float(r2.miss_m),
                   defect=float(r2.defect_m), fuel_free=float(f2.fuel_kg))
    except Exception as e:                                    # a level that cannot solve is a result
        out = dict(level=level, seed=int(seed), wall_s=time.perf_counter() - t0,
                   s1="exception", s2=str(e)[:120], iters=-1, iters_free=-1,
                   fuel=float("nan"), tf=float("nan"), D=float("nan"), miss=float("nan"),
                   defect=float("nan"), fuel_free=float("nan"))
    PC.TOL.clear(); PC.TOL.update(saved)
    return out


def main():
    nom_path, out_path = sys.argv[1], sys.argv[2]
    seeds = [int(x) for x in sys.argv[3:]]
    nom = np.load(nom_path)
    res = []
    for s in seeds:
        for lv in LEVELS:
            r = run_one(s, nom, lv)
            res.append(r)
            print(f"seed {s:>5} {lv:<8} {r['s1']:<12}{r['s2']:<12} it {r['iters']:>4} "
                  f"fuel {r['fuel']:.5f} tf {r['tf']:8.2f} [{r['wall_s']:.0f}s]", flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
