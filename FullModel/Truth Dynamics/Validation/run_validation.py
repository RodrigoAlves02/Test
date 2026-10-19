#!/usr/bin/env python
"""Run one arc through many propagators and report accuracy, evaluations and wall clock.

Usage
    python run_validation.py --case descent
    python run_validation.py --case parking --radius-km 35 --orbits 3
    python run_validation.py --case braking --gravity poly --mesh-q 64
    python run_validation.py --case descent --backend tudat        (once tudatpy is installed)

The reference is built from the tightest adaptive solver and then CROSS-CHECKED against a second,
unrelated one. A reference validated only against itself would be circular: a self-converged run
agrees with its own truncation error, so the check has to come from a different tableau.

Nothing here aligns a step to a thrust discontinuity. The thrust is a function of time and the
propagator samples it wherever its stages fall, which is how the truth model will actually run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cases as C                                                     # noqa: E402
import truth_dynamics as TD                                           # noqa: E402


def reference(case, g, rtol=1e-13, atol=1e-14):
    g.reset()
    y_a, n_a = TD.propagate_adaptive(case.y0, case.t0, case.tf, "DOP853", rtol, atol,
                                     g, case.thrust, case.veh)
    ev_a = g.n
    g.reset()
    y_b, n_b = TD.propagate_adaptive(case.y0, case.t0, case.tf, "RK45", rtol, atol,
                                     g, case.thrust, case.veh)
    spread = float(np.linalg.norm(y_a[0:3] - y_b[0:3]))
    return y_a, ev_a, spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="descent", choices=["parking", "braking", "descent"])
    ap.add_argument("--radius-km", type=float, default=35.0)
    ap.add_argument("--orbits", type=float, default=3.0)
    ap.add_argument("--gravity", default="poly", choices=["poly", "pointmass", "pinn"])
    ap.add_argument("--mesh-q", type=int, default=64)
    ap.add_argument("--backend", default="scipy", choices=["scipy", "tudat"])
    ap.add_argument("--fixed", nargs="*", default=["rk4", "rk5", "rk6", "rk8"])
    ap.add_argument("--adaptive", nargs="*", default=["RK45", "DOP853", "Radau", "LSODA"])
    ap.add_argument("--dts", type=float, nargs="*", default=None)
    ap.add_argument("--rtols", type=float, nargs="*",
                    default=[1e-6, 1e-8, 1e-10, 1e-12])
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()

    if a.backend == "tudat" and not TD.have_tudat():
        print(TD.TUDAT_HINT)
        return 1

    kw = dict(radius_km=a.radius_km, n_orbits=a.orbits) if a.case == "parking" else {}
    case = C.build(a.case, **kw)
    g = TD.Gravity(a.gravity, mesh_q=a.mesh_q)
    span = case.tf - case.t0
    dts = a.dts or [span / n for n in (32, 64, 128, 256, 512, 1024, 2048, 4096)]

    print(f"case     : {case.name}")
    print(f"           {case.note}")
    print(f"span     : {span:.2f} s")
    print(f"gravity  : {a.gravity}" + (f", {getattr(g, 'faces', None):,} facets"
                                       if getattr(g, "faces", None) else ""))
    print(f"backend  : {a.backend}\n")

    t0 = time.perf_counter()
    y_ref, ev_ref, spread = reference(case, g)
    print(f"reference: DOP853 rtol=1e-13, {ev_ref} evaluations, {time.perf_counter()-t0:.1f} s")
    print(f"           cross-check against RK45 rtol=1e-13 differs by {spread:.3e} m")
    print(f"           -> errors below this are not resolvable\n")

    rows = []

    def record(label, setting, y, evals, steps, cpu):
        e = float(np.linalg.norm(y[0:3] - y_ref[0:3]))
        ev = float(np.linalg.norm(y[3:6] - y_ref[3:6]))
        rows.append(dict(kind=label.split()[0], label=label, setting=setting, err=e, verr=ev,
                         evals=int(evals), steps=int(steps), cpu=cpu,
                         resolved=bool(e > spread)))
        print(f"  {label:9s} {setting:>14}{e:13.3e} m {evals:9d} ev {cpu:8.2f} s"
              + ("" if e > spread else "   (at reference floor)"))

    print(f"{'scheme':>11}{'setting':>15}{'error [m]':>15}{'evals':>12}{'cpu':>11}")
    print("  --- fixed step ---")
    for sc in a.fixed:
        for dt in dts:
            g.reset(); t0 = time.perf_counter()
            try:
                y, n = TD.propagate_fixed(case.y0, case.t0, case.tf, dt, sc, g, case.thrust, case.veh)
            except Exception as exc:
                print(f"  {sc:9s} dt={dt:10.3f}  FAILED: {exc}"); continue
            record(sc, f"dt={dt:.3f}s", y, g.n, n, time.perf_counter() - t0)

    print("  --- variable step ---")
    for m in a.adaptive:
        for rt in a.rtols:
            g.reset(); t0 = time.perf_counter()
            try:
                y, n = TD.propagate_adaptive(case.y0, case.t0, case.tf, m, rt, rt * 1e-3,
                                             g, case.thrust, case.veh)
            except Exception as exc:
                print(f"  {m:9s} rtol={rt:.0e}  FAILED: {exc}"); continue
            record(m, f"rtol={rt:.0e}", y, g.n, n, time.perf_counter() - t0)

    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, f"validation_{case.name}_{a.gravity}{a.mesh_q if a.gravity=='poly' else ''}.json")
    json.dump(dict(case=case.name, note=case.note, span=span, gravity=a.gravity,
                   mesh_q=a.mesh_q, backend=a.backend, ref_evals=ev_ref,
                   ref_spread=spread, rows=rows), open(f, "w"), indent=1)
    print(f"\nwrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
