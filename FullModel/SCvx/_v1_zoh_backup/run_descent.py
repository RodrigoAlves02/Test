#!/usr/bin/env python
"""Two-stage powered-descent guidance: minimum landing error, then minimum fuel to that range.

STAGE 1 (Blackmore et al. 2010, prioritised guidance) asks the only question that can be answered
unconditionally: how close to the target can this vehicle physically get from this state? It returns
a range D. STAGE 2 then minimises fuel subject to landing within D, which is feasible BY
CONSTRUCTION because D came from a solved problem.

The previous solver skipped stage 1 and posed minimum fuel with a hard terminal constraint at the
target. Near the surface that asks for a landing Tmax cannot deliver, and because the terminal and
path constraints are penalised rather than hard, it bought the shortfall with slack and reported
"converged". Stage 1 is what makes "infeasible" an answer the solver can give.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import ptr
import problem_setup as PS

HERE = os.path.dirname(os.path.abspath(__file__))


def solve(K=60, iters=35, d_margin_m=0.1, verbose=True, warm=None, **kw):
    P = PS.build(**{k: v for k, v in kw.items() if k in ("altitude_km", "cone_deg", "v0_body")})
    dyn, sc = P["dyn"], P["scales"]
    common = dict(dyn=dyn, scales=sc, x_init=P["x_init"], rf_nd=P["rf_nd"], n_hat=P["n_hat"],
                  K=K, iters=iters, verbose=verbose)

    if verbose:
        print(f"scales  L {sc.L:.1f} m   T {sc.T:.1f} s   A {sc.A:.5f} m/s^2   K {K}")
        print(f"\nSTAGE 1  minimum landing error")
    r1 = ptr.run(mode="error", warm=warm, **common)
    if r1.status.startswith("subproblem"):
        return dict(stage1=r1, stage2=None, feasible=False, setup=P)
    # a little slack so stage 2 is strictly feasible. 0.1 m, not 1 m: the margin is an upper
    # bound on the miss stage 2 is allowed to accept, and at 1 m it was measured landing up
    # to 1.15 m from the site purely because it was permitted to.
    D_nd = r1.D_nd + d_margin_m / sc.L
    if verbose:
        print(f"  -> reachable miss D = {r1.miss_m:.3f} m   tf {r1.tf_s:.1f} s   "
              f"{r1.status} in {r1.iters} it")
        print(f"\nSTAGE 2  minimum fuel, constrained to land within D + {d_margin_m:g} m")
    r2 = ptr.run(mode="fuel", D_nd=D_nd, warm=(r1.X, r1.U, r1.s), **common)
    if verbose:
        print(f"  -> fuel {r2.fuel_kg:.4f} kg   miss {r2.miss_m:.3f} m   tf {r2.tf_s:.1f} s   "
              f"defect {r2.defect_m:.3e} m   {r2.status} in {r2.iters} it")
    return dict(stage1=r1, stage2=r2, feasible=True, setup=P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=60)
    ap.add_argument("--iters", type=int, default=35)
    ap.add_argument("--altitude-km", type=float, default=C.ALTITUDE_KM)
    ap.add_argument("--cone-deg", type=float, default=C.CONE_ANGLE_DEG)
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()
    out = solve(K=a.K, iters=a.iters, altitude_km=a.altitude_km, cone_deg=a.cone_deg)
    r2 = out["stage2"]
    if r2 is None:
        print("\nINFEASIBLE: stage 1 could not be solved"); return
    os.makedirs(a.out, exist_ok=True)
    P = out["setup"]; sc = P["scales"]
    np.savez(os.path.join(a.out, "descent.npz"),
             X=r2.X, U=r2.U, s=r2.s, L=sc.L, T=sc.T, A=sc.A,
             rf=P["r_site"], n_hat=P["n_hat"], K=a.K)
    json.dump(dict(K=a.K, tf_s=r2.tf_s, fuel_kg=r2.fuel_kg, miss_m=r2.miss_m,
                   defect_m=r2.defect_m, status=r2.status, iters=r2.iters,
                   D_m=out["stage1"].miss_m, wall_s=r2.wall_s,
                   history=[out["stage1"].history, r2.history]),
              open(os.path.join(a.out, "descent.json"), "w"), indent=1)
    print(f"\nwrote {a.out}/descent.{{npz,json}}")


if __name__ == "__main__":
    main()
