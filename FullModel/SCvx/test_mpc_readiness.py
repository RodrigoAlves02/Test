#!/usr/bin/env python
"""Three things that decide whether this solver can close a loop, measured rather than assumed.

  1. WALL TIME per guidance call. A 30 s cycle is the budget.
  2. MID-FLIGHT RESTART from a shifted warm start -- the actual MPC call.
  3. BEHAVIOUR WHEN THE STATE IS OUTSIDE THE GLIDESLOPE CONE. The nominal rides the cone at
     29.99 deg of a 30 deg limit, and node 0 is a hard equality, so any dispersion that pushes the
     vehicle outside makes the subproblem infeasible at the very first node.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C, ptr, problem_setup as PS

HERE = os.path.dirname(os.path.abspath(__file__))


def shift(X, U, s, K, frac):
    """The part of a plan still ahead, resampled onto K nodes -- what an MPC cycle hands over."""
    Kb = X.shape[1]
    tau_old = np.linspace(0.0, 1.0, Kb)
    t0 = frac
    tq = np.clip(t0 + np.linspace(0.0, 1.0, K) * (1.0 - t0), 0.0, 1.0)
    Xs = np.array([np.interp(tq, tau_old, X[i]) for i in range(X.shape[0])])
    Us = np.array([np.interp(tq, tau_old, U[i]) for i in range(U.shape[0])])
    return Xs, Us, s * (1.0 - t0)


def main(K=60, iters=35):
    P = PS.build(); dyn, sc, n = P["dyn"], P["scales"], P["n_hat"]
    base = np.load(os.path.join(HERE, "Figures&data", "descent_K60.npz"))
    Xb, Ub, sb = base["X"], base["U"], float(base["s"])
    Pl = np.eye(3) - np.outer(n, n)
    rows = []

    print("1. WALL TIME AND MID-FLIGHT RESTART  (warm start = the remaining plan, as in MPC)")
    print(f"{'along':>7}{'t_go [s]':>10}{'K':>4}{'stage1 [s]':>12}{'stage2 [s]':>12}"
          f"{'total [s]':>11}{'it1':>5}{'it2':>5}{'status':>13}{'fuel [kg]':>11}")
    for KK in (30, 60):
        for fr in (0.0, 0.25, 0.5, 0.75, 0.9):
            j = int(round(fr * (Xb.shape[1] - 1))); j = min(j, Xb.shape[1] - 2)
            xi = Xb[:, j].copy()
            Xs, Us, ss = shift(Xb, Ub, sb, KK, fr)
            common = dict(dyn=dyn, scales=sc, x_init=xi, rf_nd=P["rf_nd"], n_hat=n,
                          K=KK, iters=iters, verbose=False)
            r1 = ptr.run(mode="error", warm=(Xs, Us, ss), **common)
            r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + 0.1 / sc.L,
                         warm=(r1.X, r1.U, r1.s), **common)
            tot = r1.wall_s + r2.wall_s
            print(f"{fr:>7.0%}{sc.t_dim(ss):>10.1f}{KK:>4}{r1.wall_s:>12.2f}{r2.wall_s:>12.2f}"
                  f"{tot:>11.2f}{r1.iters:>5}{r2.iters:>5}{r2.status:>13}{r2.fuel_kg:>11.4f}",
                  flush=True)
            rows.append(dict(kind="mpc", frac=fr, K=KK, wall=tot, it1=r1.iters, it2=r2.iters,
                             status=r2.status, fuel=r2.fuel_kg, tgo=sc.t_dim(ss)))

    print("\n2. STATE PUSHED OUTSIDE THE GLIDESLOPE CONE  (node 0 is a hard constraint)")
    print(f"{'along':>7}{'cone angle at x0':>19}{'stage1':>13}{'stage2':>13}{'fuel [kg]':>11}")
    for fr in (0.5, 0.8):
        j = int(round(fr * (Xb.shape[1] - 1)))
        x0 = Xb[:, j].copy()
        alt = float(n @ (x0[0:3] - P["rf_nd"]))
        for extra in (0.0, 0.5, 1.02, 1.15):
            lat_dir = Pl @ (x0[0:3] - P["rf_nd"])
            ln = np.linalg.norm(lat_dir)
            lat_dir = lat_dir / ln if ln > 1e-12 else P["east"]
            target = np.tan(np.radians(C.GLIDESLOPE_DEG)) * alt * extra if extra > 0 else ln
            xi = x0.copy()
            xi[0:3] = P["rf_nd"] + n * alt + lat_dir * (target if extra > 0 else ln)
            ang = np.degrees(np.arctan2(np.linalg.norm(Pl @ (xi[0:3] - P["rf_nd"])),
                                        n @ (xi[0:3] - P["rf_nd"])))
            Xs, Us, ss = shift(Xb, Ub, sb, K, fr)
            common = dict(dyn=dyn, scales=sc, x_init=xi, rf_nd=P["rf_nd"], n_hat=n,
                          K=K, iters=iters, verbose=False)
            r1 = ptr.run(mode="error", warm=(Xs, Us, ss), **common)
            if r1.status in ("converged", "maxiter"):
                r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + 0.1 / sc.L,
                             warm=(r1.X, r1.U, r1.s), **common)
                st2, fu = r2.status, r2.fuel_kg
            else:
                st2, fu = "-", float("nan")
            print(f"{fr:>7.0%}{ang:>18.2f}$^\\circ$"[:26].replace("$^\\circ$", " deg")
                  + f"{r1.status:>13}{st2:>13}{fu:>11.4f}", flush=True)
            rows.append(dict(kind="cone", frac=fr, angle=float(ang), stage1=r1.status,
                             stage2=st2, fuel=float(fu)))
    json.dump(rows, open(os.path.join(HERE, "Figures&data", "mpc_readiness.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
