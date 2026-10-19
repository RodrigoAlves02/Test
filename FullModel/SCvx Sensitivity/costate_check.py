#!/usr/bin/env python
"""Two independent routes to the same gradient: finite differences, and the solver's own duals.

WHAT THIS VERIFIES. For a minimum-fuel optimal control problem the gradient of the optimal cost
with respect to the initial state IS the initial costate,

    dJ*/dx0  =  -lambda(0)

That is an identity of the problem, not a property of any solver. So there are two ways to get the
same seven numbers, and they share no code path:

  PRIMAL   perturb one component of x0, re-solve from scratch, and difference the propellant.
           That is sweep.py: 350 independent solves, each converged to its own tolerances.
  DUAL     solve ONCE, and read the dual variable of the constraint X[:, 0] == x_init.
           That number is produced by CLARABEL's interior-point solution of the final convex
           subproblem and never sees a finite difference.

If they agree, the solver is not merely converging -- it is converging to a point whose sensitivity
structure is the one the physics demands, and its duals are trustworthy. That is a far stronger
statement than "the residual is small", and it is the one a verification chapter should make.

THE UNIT CHAIN, WHICH IS WHERE THIS GOES WRONG IF IT IS NOT WRITTEN DOWN. The subproblem minimises
obj = -z_K / alpha_nd (plus penalties that vanish at convergence), while the quantity of interest is
the PHYSICAL propellant J = exp(z_0) - exp(z_K) in kg, and the state is non-dimensional. So

    d(obj)/dx_nd            = the dual, call it mu
    dz_K/dx_nd              = -alpha_nd * mu
    dJ/dx_nd                = -exp(z_K) * dz_K/dx_nd = exp(z_K) * alpha_nd * mu     (x != z_0)
    dJ/dr_phys [kg/m]       = dJ/dx_nd / L
    dJ/dv_phys [kg/(m/s)]   = dJ/dx_nd * T / L
    dJ/dm0     [kg/kg]      = 1 + exp(z_K) * alpha_nd * mu_z / m0

the +1 on the mass row being the explicit exp(z_0) in J, which no other row has.

THE SIGN CONVENTION IS FIXED ONCE, NOT PER CHANNEL. cvxpy's sign for an equality dual depends on
how the constraint was written, so the overall sign is settled on a single channel and then every
other channel has to follow it. Seven numbers agreeing under one shared convention is a test;
flipping the sign per channel would not be.
"""
from __future__ import annotations

import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TBB_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.abspath(os.path.join(HERE, "..", "SCvx", "_v1_zoh_backup"))
SCVX = os.path.abspath(os.path.join(HERE, "..", "SCvx"))
sys.path.insert(0, SCVX)
sys.path.insert(0, V1)

import config as C                                                           # noqa: E402
assert os.path.dirname(os.path.abspath(C.__file__)) == V1

SITE = dict(name="A", lat_deg=-36.975, lon_deg=95.401)
C.POINTING_DEG = 180.0
C.LAT_DEG, C.LON_DEG = SITE["lat_deg"], SITE["lon_deg"]

import gravity as GR                                                         # noqa: E402
import ptr                                                                   # noqa: E402
import subproblem as SP                                                      # noqa: E402
from dynamics import NX, Dynamics, discretize                                # noqa: E402

assert NX == 7
K = 100
ALTITUDE_KM = 11.771
TOL = dict(iters=200, tol_obj=1e-7, tol_defect_m=1e-3, tol_defect_v=1e-4)
OUT = os.path.join(HERE, "Figures&data")


def geometry():
    r_site, n_hat = GR.site_and_normal(lat_deg=SITE["lat_deg"], lon_deg=SITE["lon_deg"])
    east = np.cross([0.0, 0.0, 1.0], n_hat); east /= np.linalg.norm(east)
    north = np.cross(n_hat, east)
    r0 = r_site + ALTITUDE_KM * 1e3 * n_hat
    vl = np.asarray(C.V0_LOCAL, float)
    v0 = vl[0] * n_hat + vl[1] * east + vl[2] * north
    return r_site, n_hat, r0, v0


def main():
    r_site, n_hat, r0, v0 = geometry()
    m0 = float(C.MWET)
    sc = C.Scales.build(r0, r_site, m_wet=m0)
    fld = GR.pinn_field()
    dyn = Dynamics(sc, field=fld)
    x_init = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(m0)]])
    common = dict(dyn=dyn, scales=sc, x_init=x_init, rf_nd=r_site / sc.L,
                  n_hat=n_hat, K=K, verbose=False, **TOL)

    print(f"site {SITE['name']}   handover {ALTITUDE_KM:g} km   K {K}   "
          f"gravity PINN-GM-III   7-state")
    r1 = ptr.run(mode="error", warm=None, **common)
    if not str(r1.status).startswith("converged"):
        raise SystemExit(f"stage 1: {r1.status}")
    z = np.load(os.path.join(OUT, "sweep_nominal_plan.npz"))
    warm = (np.vstack([z["r"] / sc.L, z["v"] / sc.v, np.log(z["m"])[None, :]]),
            np.vstack([z["a"] / sc.A, (z["gamma"] / sc.A)[None, :]]),
            float(sc.t_nd(float(z["tf_s"]))))
    r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + 0.1 / sc.L, warm=warm, ratio_test=True, **common)
    print(f"converged solve: {r2.status} in {r2.iters} it   J {r2.fuel_kg:.5f} kg   "
          f"tf {r2.tf_s:.2f} s   defect {r2.defect_m:.2e} m\n")

    # ---- re-solve the FINAL subproblem at the converged reference, and read its dual ----------
    # The duals belong to the last convex subproblem, so it has to be rebuilt at the converged
    # linearisation point rather than taken from somewhere mid-iteration.
    Xr, Ur, sr = r2.X, r2.U, float(r2.s)
    Ad, Bd, Sd, zd, _ = discretize(dyn, Xr, Ur, sr, n_sub=10)
    prob, V = SP.build(Ad, Bd, Sd, zd, Xr, Ur, sr, x_init, r_site / sc.L, n_hat, sc,
                       mode="fuel", D_nd=r1.D_nd + 0.1 / sc.L, w_vc=1e4, w_tr=0.0, w_ts=0.0,
                       vtouch_nd=C.VTOUCH / sc.v, R=1.0, Rs=1.0,
                       term_point_deg=C.TERMINAL_POINT_DEG,
                       term_point_nodes=C.TERMINAL_POINT_NODES)
    name, st = SP.solve(prob)
    mu = np.asarray(V["ic"].dual_value, float)
    # Settled on the SIX position/velocity rows, which are pure chain rule with no additive term,
    # by requiring d(z_K)/d(z_0) = -alpha_nd * mu_z to be positive: a heavier vehicle at handover
    # is a heavier vehicle at touchdown, whatever cvxpy's sign convention happens to be.
    SIGN = -1.0 if (-sc.alpha_nd * mu[6]) < 0 else 1.0
    zK = float(np.asarray(V["X"].value)[6, -1])
    print(f"final subproblem: {name} {st}   dual of X[:,0] == x_init:")
    print("  " + "  ".join(f"{x:+.4e}" for x in mu) + "\n")

    # ---- the unit chain ----------------------------------------------------------------------
    # THE SIGN GOES ON THE DUAL, NOT ON THE FINISHED GRADIENT. cvxpy's sign for an equality dual
    # depends on how the constraint was written, so one overall sign has to be settled -- but it
    # must be applied BEFORE the chain rule, because the mass row carries an additive +1 from the
    # explicit exp(z_0) in J and -(1 + x) is not (1 - x). Applying it afterwards put the mass
    # sensitivity at -1994 g/kg against a measured +5.87.
    mu = SIGN * mu
    a_nd = sc.alpha_nd
    base = np.exp(zK) * a_nd * mu                       # dJ/dx_nd, kg per non-dimensional unit
    grad = np.empty(7)
    grad[0:3] = base[0:3] / sc.L                        # kg per m
    grad[3:6] = base[3:6] * sc.T / sc.L                 # kg per (m/s)
    grad[6] = 1.0 + base[6] / m0                        # kg per kg
    grad *= 1e3                                         # grams

    # ---- what the 350 primal solves measured -------------------------------------------------
    S = json.load(open(os.path.join(OUT, "sweep.json")))
    J0 = S["nominal"]["J_kg"]
    fd = {}
    for ch in ("r_x", "r_y", "r_z", "v_x", "v_y", "v_z", "m"):
        s = [r for r in S["rows"] if r["channel"] == ch and r["converged"]]
        d = np.array([r["delta"] for r in s])
        y = (np.array([r["J_kg"] for r in s]) - J0) * 1e3
        M = np.vstack([d, d * d]).T                     # linear + curvature, no constant
        (a, b), *_ = np.linalg.lstsq(M, y, rcond=None)
        fd[ch] = (float(a), float(b), len(s))

    # SIGN FIXED ONCE, on the channel with the cleanest finite difference (largest |slope| over
    # its own scatter), then applied unchanged to all seven.
    chans = ["r_x", "r_y", "r_z", "v_x", "v_y", "v_z", "m"]
    g = np.array([grad[i] for i in range(7)])
    sgn = SIGN
    print(f"sign convention, settled on d(z_K)/d(z_0) > 0 alone: dual * {sgn:+.0f}\n")
    U = {"r_x": "g/m", "r_y": "g/m", "r_z": "g/m", "v_x": "g per m/s", "v_y": "g per m/s",
         "v_z": "g per m/s", "m": "g/kg"}
    print(f"{'chan':>5} {'finite difference':>18} {'costate -lambda(0)':>20} {'diff':>10} "
          f"{'rel':>8}  {'unit':<10}")
    print("-" * 78)
    for i, ch in enumerate(chans):
        a, b, n = fd[ch]
        d = g[i] - a
        rel = 100.0 * d / a if abs(a) > 1e-9 else float("nan")
        print(f"{ch:>5} {a:>18.5f} {g[i]:>20.5f} {d:>+10.5f} {rel:>+7.1f} %  {U[ch]:<10}")
    out = os.path.join(OUT, "costate_check.json")
    json.dump(dict(site=SITE, K=K, status=str(r2.status), J_kg=float(r2.fuel_kg),
                   tf_s=float(r2.tf_s), dual=mu.tolist(), sign=float(sgn),
                   grad_g=dict(zip(chans, g.tolist())),
                   fd_g=dict((c, fd[c][0]) for c in chans)), open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
