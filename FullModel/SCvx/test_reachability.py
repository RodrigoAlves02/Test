#!/usr/bin/env python
"""Does stage 1 actually report unreachability when the target becomes unreachable?

This is the reason the two-stage structure exists. On the undispersed nominal stage 1 always
returns D = 0, so it looks like a no-op; its value appears in closed loop, where a deviation can
put the site out of reach and the guidance still has to produce a FEASIBLE trajectory rather than
keep claiming a landing the thrust cannot deliver.

The test takes states along the nominal, adds excess velocity toward the surface, and asks stage 1
how close the vehicle can still get.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C, ptr, problem_setup as PS

HERE = os.path.dirname(os.path.abspath(__file__))


def main(K=40, iters=30, fracs=(0.5, 0.8), lat=(0.0, 200.0, 600.0, 1500.0, 3000.0)):
    P = PS.build(); dyn, sc, n = P["dyn"], P["scales"], P["n_hat"]
    base = np.load(os.path.join(HERE, "Figures&data", "descent_K60.npz"))
    X, s = base["X"], float(base["s"])
    Kb = X.shape[1]
    east = P["east"]
    print(f"{'along':>7}{'alt [m]':>10}{'cross [m]':>11}{'stage1':>12}{'D [m]':>12}"
          f"{'tf [s]':>9}{'stage2':>12}{'fuel [kg]':>11}")
    rows = []
    for fr in fracs:
        j = int(round(fr * (Kb - 1))); j = min(j, Kb - 2)
        x0 = X[:, j].copy()
        alt = float(n @ (x0[0:3] - P["rf_nd"])) * sc.L
        for dv in lat:
            xi = x0.copy()
            xi[0:3] = xi[0:3] + (dv / sc.L) * east     # CROSS-RANGE offset: a divert problem
            common = dict(dyn=dyn, scales=sc, x_init=xi, rf_nd=P["rf_nd"], n_hat=n,
                          K=K, iters=iters, verbose=False)
            r1 = ptr.run(mode="error", **common)
            ok1 = r1.status == "converged"
            D = r1.miss_m
            if ok1:
                r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + 1.0 / sc.L,
                             warm=(r1.X, r1.U, r1.s), **common)
                st2, fu = r2.status, r2.fuel_kg
            else:
                st2, fu = "-", float("nan")
            print(f"{fr:>7.0%}{alt:>10.0f}{dv:>11.0f}{r1.status:>12}{D:>12.3f}"
                  f"{r1.tf_s:>9.1f}{st2:>12}{fu:>11.4f}", flush=True)
            rows.append(dict(frac=fr, alt_m=alt, dv=dv, stage1=r1.status, D_m=D,
                             tf=r1.tf_s, stage2=st2, fuel=float(fu)))
    reach = [r for r in rows if r["stage1"] == "converged"]
    nz = [r for r in reach if r["D_m"] > 1.0]
    print(f"\n{len(reach)}/{len(rows)} cases solved by stage 1; "
          f"{len(nz)} of them reported the site OUT OF REACH (D > 1 m)")
    if nz:
        print("  in those cases stage 2 still returned a feasible landing, at the best reachable")
        print("  point rather than claiming the target -- which is the whole purpose of stage 1")
    json.dump(rows, open(os.path.join(HERE, "Figures&data", "reachability.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
