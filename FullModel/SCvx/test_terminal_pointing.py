#!/usr/bin/env python
"""What does an upright touchdown cost?

Sweeps the terminal thrust-pointing cone from free down to exactly vertical and reports the
propellant, so the choice is made on a number rather than on preference.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C, ptr, problem_setup as PS, subproblem as SP, verify as V
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))


def main(K=60, iters=35, angles=(None, 20.0, 10.0, 5.0, 0.0), nodes=(1, 3)):
    P = PS.build(); dyn, sc, n = P["dyn"], P["scales"], P["n_hat"]
    print(f"{'terminal cone':>14}{'nodes':>7}{'status':>12}{'it':>4}{'tf [s]':>9}"
          f"{'fuel [kg]':>11}{'d fuel':>9}{'end angle':>11}{'miss flown':>12}")
    base = None; rows = []
    cases = [(None, 0)] + [(a, nd) for nd in nodes for a in angles if a is not None]
    for a, nd in cases:
        common = dict(dyn=dyn, scales=sc, x_init=P["x_init"], rf_nd=P["rf_nd"], n_hat=n,
                      K=K, iters=iters, verbose=False,
                      term_point_deg=a, term_point_nodes=nd)
        r1 = ptr.run(mode="error", **common)
        r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + 1.0 / sc.L, warm=(r1.X, r1.U, r1.s), **common)
        Xf, _ = V.fly(dyn, sc, r2.X, r2.U, r2.s)
        Pl = np.eye(3) - np.outer(n, n)
        miss = float(np.linalg.norm(Pl @ (Xf[0:3, -1] - P["rf_nd"]))) * sc.L
        u = r2.U[0:3, K - 2]
        ang = np.degrees(np.arccos(np.clip(u @ n / np.linalg.norm(u), -1, 1)))
        if base is None:
            base = r2.fuel_kg
        d = (r2.fuel_kg - base) / base * 100
        lbl = "free" if a is None else f"{a:g} deg"
        print(f"{lbl:>14}{nd:>7}{r2.status:>12}{r2.iters:>4}{r2.tf_s:>9.1f}"
              f"{r2.fuel_kg:>11.4f}{d:>8.2f}%{ang:>11.3f}{miss:>12.4f}", flush=True)
        rows.append(dict(cone=a, nodes=nd, status=r2.status, tf=r2.tf_s, fuel=r2.fuel_kg,
                         pct=d, end_angle=float(ang), miss=miss))
    json.dump(rows, open(os.path.join(HERE, "Figures&data", "terminal_pointing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
