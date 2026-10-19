#!/usr/bin/env python
"""What is the global pointing cone costing, and does dropping it rescue the infeasible draws?

POINTING_DEG = 140 is active at INTERIOR nodes on 27.5 % of the generated trajectories -- pressed
exactly against the limit -- so it is not a formality, it is shaping a third of the label set. It is
also the direct cause of 44 of the 60 rejections: the handover attitude is pinned by d_init and the
cone is imposed at every node including node 0, so a pointing draw outside it makes the problem
infeasible by construction rather than hard.

Dropping it is defensible because the landing attitude is forced elsewhere -- the terminal ramp
drives d to vertical over the last nodes and the slew limit stops it jumping there -- so the cone is
not what makes the vehicle land upright. What it does do is stop the optimiser pointing FURTHER
down, and since the engine cannot be switched off and Tmin/m exceeds gravity everywhere, pointing
down is how this vehicle descends at all. Expect shorter flights and less fuel without it.

point_deg = 180 removes it exactly rather than by special case: n.d >= cos(180) = -1 is satisfied
identically by any unit vector.

Usage:  cone_test.py <nominal.npz> <out.json> <seed> [seed ...]
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
import ptr                                                              # noqa: E402

LEVELS = {"cone140": 140.0, "cone_off": 180.0}


class A:
    pos_sigma = 1000.0; vel_sigma = 1.0; mass_sigma = 2.0; point_sigma_deg = 3.0
    seed = 0


def run_one(seed, nom, level):
    A.seed = seed
    r0, v0, m0, d0 = PC.draw(A, nom)
    nh = np.asarray(nom["n_hat"], float)
    ang0 = float(np.degrees(np.arccos(np.clip(d0 @ nh, -1, 1))))   # handover angle to vertical
    saved = PC.ALIGN.get("point_deg", None)
    PC.ALIGN["point_deg"] = LEVELS[level]
    t0 = time.perf_counter()
    try:
        r1, r2, sc, P, f2 = PC.solve(r0, v0, m0, 100, d_init=d0)
        d = r2.X[7:10, :]
        amax = float(np.degrees(np.arccos(np.clip((d.T @ nh), -1, 1))).max())
        out = dict(level=level, seed=int(seed), wall_s=time.perf_counter() - t0,
                   s1=str(r1.status), s2=str(r2.status), ang0=ang0, ang_max=amax,
                   iters=int(r1.iters) + int(r2.iters), fuel=float(r2.fuel_kg),
                   tf=float(r2.tf_s), D=float(r1.D_nd) * sc.L, miss=float(r2.miss_m))
    except Exception as e:
        out = dict(level=level, seed=int(seed), wall_s=time.perf_counter() - t0,
                   s1="exception", s2=str(e)[:100], ang0=ang0, ang_max=float("nan"),
                   iters=-1, fuel=float("nan"), tf=float("nan"), D=float("nan"),
                   miss=float("nan"))
    if saved is None:
        PC.ALIGN.pop("point_deg", None)
    else:
        PC.ALIGN["point_deg"] = saved
    return out


def main():
    nom, out_path = np.load(sys.argv[1]), sys.argv[2]
    res = []
    for s in (int(x) for x in sys.argv[3:]):
        for lv in LEVELS:
            r = run_one(s, nom, lv)
            res.append(r)
            print(f"seed {s:>5} {lv:<9} {r['s1']:<12}{r['s2']:<12} ang0 {r['ang0']:6.2f} "
                  f"amax {r['ang_max']:6.2f} it {r['iters']:>4} fuel {r['fuel']:.5f} "
                  f"tf {r['tf']:8.2f} [{r['wall_s']:.0f}s]", flush=True)
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
