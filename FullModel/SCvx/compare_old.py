#!/usr/bin/env python
"""Like-for-like comparison with the previous solver.

The two setups are NOT identical -- the ray-cast site here puts the start 12000 m out against the
old 11771 m -- so comparing the headline fuel numbers would be comparing two different problems.
This solves the NEW solver from the OLD solver's exact initial condition and landing site, and flies
both control histories through the same propagator, open loop from node 0.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C, gravity as GR, ptr, verify as V
from dynamics import Dynamics, defect

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(C.THESIS, "scvx_landing", "descent_nominal.npz")


def load_old():
    z = np.load(OLD)
    X, U, sig, rf = z["X"], z["U"], float(z["sigma"]), np.asarray(z["rf"], float)
    return X, U, sig, rf


def main(K_new=60, iters=35):
    Xo, Uo, so, rf = load_old()
    r0 = Xo[0:3, 0].copy(); v0 = Xo[3:6, 0].copy(); m0 = float(np.exp(Xo[6, 0]))
    sc = C.Scales.build(r0, rf, m_wet=m0)
    dyn = Dynamics(sc)
    fld = GR.field()
    g_eff = fld.g(rf + 1.0 * rf / np.linalg.norm(rf)) - np.cross(C.W_BODY, np.cross(C.W_BODY, rf))
    n = -g_eff / np.linalg.norm(g_eff)
    rf_nd = rf / sc.L
    x_init = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(m0)]])
    print(f"OLD initial condition: range {sc.L:.1f} m, |v0| {np.linalg.norm(v0):.4f} m/s, "
          f"m0 {m0:.2f} kg, K_old {Xo.shape[1]}")

    # --- fly the OLD solution, open loop, through the same propagator ----------------------------
    Xo_nd = np.vstack([Xo[0:3] / sc.L, Xo[3:6] * sc.T / sc.L, Xo[6:7]])
    Uo_nd = np.vstack([Uo[0:3] / sc.A, Uo[3:4] / sc.A])
    so_nd = so / sc.T
    Xf_old, _ = V.fly(dyn, sc, Xo_nd, Uo_nd, so_nd)
    Pl = np.eye(3) - np.outer(n, n)
    old = dict(tf=so, fuel_rep=float(np.exp(Xo[6, 0]) - np.exp(Xo[6, -1])),
               fuel_flown=float(np.exp(Xf_old[6, 0]) - np.exp(Xf_old[6, -1])),
               miss_flown=float(np.linalg.norm(Pl @ (Xf_old[0:3, -1] - rf_nd))) * sc.L,
               alt_flown=float(n @ (Xf_old[0:3, -1] - rf_nd)) * sc.L,
               v_flown=float(np.linalg.norm(Xf_old[3:6, -1])) * sc.v,
               div=float(np.abs(Xf_old[0:3] - Xo_nd[0:3]).max()) * sc.L, K=Xo.shape[1])

    # --- solve the SAME problem with the new solver ----------------------------------------------
    common = dict(dyn=dyn, scales=sc, x_init=x_init, rf_nd=rf_nd, n_hat=n, K=K_new,
                  iters=iters, verbose=False)
    r1 = ptr.run(mode="error", **common)
    r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + 1.0 / sc.L, warm=(r1.X, r1.U, r1.s), **common)
    Xf_new, _ = V.fly(dyn, sc, r2.X, r2.U, r2.s)
    new = dict(tf=r2.tf_s, fuel_rep=r2.fuel_kg,
               fuel_flown=float(np.exp(Xf_new[6, 0]) - np.exp(Xf_new[6, -1])),
               miss_flown=float(np.linalg.norm(Pl @ (Xf_new[0:3, -1] - rf_nd))) * sc.L,
               alt_flown=float(n @ (Xf_new[0:3, -1] - rf_nd)) * sc.L,
               v_flown=float(np.linalg.norm(Xf_new[3:6, -1])) * sc.v,
               div=float(np.abs(Xf_new[0:3] - r2.X[0:3]).max()) * sc.L, K=K_new,
               status=r2.status, iters=r2.iters)

    print(f"\n{'':32}{'OLD solver':>16}{'NEW solver':>16}")
    print(f"{'nodes K':32}{old['K']:>16d}{new['K']:>16d}")
    print(f"{'flight time [s]':32}{old['tf']:>16.1f}{new['tf']:>16.1f}")
    print(f"{'propellant, reported [kg]':32}{old['fuel_rep']:>16.4f}{new['fuel_rep']:>16.4f}")
    print(f"{'propellant, FLOWN [kg]':32}{old['fuel_flown']:>16.4f}{new['fuel_flown']:>16.4f}")
    print(f"{'lateral miss, FLOWN [m]':32}{old['miss_flown']:>16.4f}{new['miss_flown']:>16.4f}")
    print(f"{'terminal altitude, FLOWN [m]':32}{old['alt_flown']:>16.4f}{new['alt_flown']:>16.4f}")
    print(f"{'terminal speed, FLOWN [m/s]':32}{old['v_flown']:>16.4f}{new['v_flown']:>16.4f}")
    print(f"{'open-loop divergence [m]':32}{old['div']:>16.4f}{new['div']:>16.4f}")
    d = (new['fuel_flown'] - old['fuel_flown']) / old['fuel_flown'] * 100
    print(f"\npropellant: {d:+.2f}% "
          f"({'NEW is cheaper' if d < 0 else 'OLD is cheaper'}), on the SAME initial condition")
    json.dump(dict(old=old, new=new), open(os.path.join(HERE, "Figures&data", "compare_old.json"),
                                           "w"), indent=1)


if __name__ == "__main__":
    main()
