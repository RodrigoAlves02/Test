#!/usr/bin/env python
"""Ground truth: fuel as a function of flight time, from fixed-time solves.

With tf pinned the flight time cannot be got wrong for solver reasons, so this curve is the
reference the free-time solver's answer must lie on. It also settles whether the optimum is sharp
or flat -- which decides whether a spread in the returned tf is a defect or just the shape of the
objective.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C, ptr, problem_setup as PS

HERE = os.path.dirname(os.path.abspath(__file__))


def main(K=30, iters=25, tfs=None):
    P = PS.build(); dyn, sc = P["dyn"], P["scales"]
    v0n = float(np.linalg.norm(P["x_init"][3:6])) * sc.v
    rng = float(np.linalg.norm(P["x_init"][0:3] - P["rf_nd"])) * sc.L
    t_lo, t_hi = C.tf_bounds(v0n, rng)
    if tfs is None:
        tfs = np.arange(900.0, 3001.0, 150.0)
    common = dict(dyn=dyn, scales=sc, x_init=P["x_init"], rf_nd=P["rf_nd"],
                  n_hat=P["n_hat"], K=K, iters=iters, verbose=False)
    print(f"{'tf [s]':>9}{'status':>12}{'it':>4}{'fuel [kg]':>11}{'miss [m]':>10}"
          f"{'defect [m]':>12}{'|nu|':>10}")
    rows = []
    for tf in tfs:
        if not (t_lo <= tf <= t_hi):
            continue
        r1 = ptr.run(mode="error", fix_tf_s=tf, **common)
        D = r1.D_nd + 1.0 / sc.L
        r2 = ptr.run(mode="fuel", D_nd=D, warm=(r1.X, r1.U, r1.s), fix_tf_s=tf, **common)
        ok = (r2.miss_m < 1.0) and (r2.defect_m < 1.0) and (r2.vc < 1e-5)
        rows.append(dict(tf=float(tf), fuel=r2.fuel_kg, miss=r2.miss_m, defect=r2.defect_m,
                         vc=r2.vc, status=r2.status, iters=r2.iters, ok=bool(ok)))
        print(f"{tf:>9.0f}{r2.status:>12}{r2.iters:>4}{r2.fuel_kg:>11.4f}{r2.miss_m:>10.4f}"
              f"{r2.defect_m:>12.2e}{r2.vc:>10.1e}{'' if ok else '   <- rejected'}", flush=True)
    good = [r for r in rows if r["ok"]]
    if good:
        f = np.array([r["fuel"] for r in good]); t = np.array([r["tf"] for r in good])
        i = int(np.argmin(f))
        print(f"\nminimum on the swept grid: tf {t[i]:.0f} s, fuel {f[i]:.4f} kg")
        band = t[f < f[i] * 1.001]
        print(f"within 0.1% of that minimum: tf {band.min():.0f} .. {band.max():.0f} s "
              f"(width {band.max()-band.min():.0f} s)")
    os.makedirs(os.path.join(HERE, "Figures&data"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "Figures&data", "tf_sweep.json"), "w"), indent=1)
    return rows


if __name__ == "__main__":
    main()
