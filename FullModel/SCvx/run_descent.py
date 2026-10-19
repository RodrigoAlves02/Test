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
import inspect
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import ptr
import problem_setup as PS

HERE = os.path.dirname(os.path.abspath(__file__))


def solve(K=60, iters=35, d_margin_m=0.1, verbose=True, warm=None,
          slew_max_deg_s=None, **kw):
    """Two-stage solve. slew_max_deg_s overrides config.SLEW_MAX_DEG_S for this call.

    PASS IT HERE RATHER THAN PATCHING THE CONFIG. ptr.run declares the limit as a DEFAULT ARGUMENT
    (slew_max_deg_s=C.SLEW_MAX_DEG_S), and Python evaluates default arguments once, when the def is
    executed -- i.e. when ptr is first imported. Setting config.SLEW_MAX_DEG_S afterwards changes
    nothing, and reloading this module does not help because it does not reload ptr. A sweep written
    that way ran every case at whichever value happened to be live at import time: measured, a
    1.0 deg/s case and a 2.0 deg/s case produced peaks of 1.4162 and 1.4147 deg/s, i.e. the same
    unconstrained answer twice, and the "overshoot" was against a label rather than a bound.
    Patching the config BEFORE the first import of ptr does work, which is what makes the trap
    intermittent and worth stating.
    """
    # **kw USED TO BE SILENTLY DISCARDED apart from the three setup keys, so a caller writing
    # solve(tol_abs=1e-8) got the DEFAULT tolerance and no warning -- a tolerance study ran four
    # times at the same settings and "proved" the tolerance had no effect. Anything not consumed by
    # problem_setup is now forwarded to ptr.run, and a genuinely unknown key raises instead of
    # vanishing.
    # DERIVED FROM THE SIGNATURE, not hand-listed. The hand-written tuple was
    # ("altitude_km", "cone_deg", "v0_body") -- and "v0_body" is not a parameter of
    # problem_setup.build at all (it is v0_local), so that key would have been forwarded
    # straight into a TypeError, while v0_local, m_wet, r0_override and v0_override -- all
    # real parameters -- were silently dropped on the floor. Reading the signature keeps the
    # two in step.
    _setup_keys = tuple(inspect.signature(PS.build).parameters)
    P = PS.build(**{k: v for k, v in kw.items() if k in _setup_keys})
    _fwd = {k: v for k, v in kw.items() if k not in _setup_keys}
    _ok = set(inspect.signature(ptr.run).parameters)
    _bad = set(_fwd) - _ok
    if _bad:
        raise TypeError(f"solve() got unexpected keyword argument(s) {sorted(_bad)}; "
                        f"neither problem_setup.build nor ptr.run accepts them")
    dyn, sc = P["dyn"], P["scales"]
    common = dict(dyn=dyn, scales=sc, x_init=P["x_init"], rf_nd=P["rf_nd"], n_hat=P["n_hat"],
                  K=K, iters=iters, verbose=verbose,
                  slew_max_deg_s=(C.SLEW_MAX_DEG_S if slew_max_deg_s is None
                                  else slew_max_deg_s), **_fwd)

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
    ap.add_argument("--slew-max-deg-s", dest="slew_max_deg_s", type=float,
                    default=C.SLEW_MAX_DEG_S,
                    help="thrust-direction rate limit [deg/s]; overrides config")
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()
    out = solve(K=a.K, iters=a.iters, altitude_km=a.altitude_km, cone_deg=a.cone_deg,
                slew_max_deg_s=a.slew_max_deg_s)
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
