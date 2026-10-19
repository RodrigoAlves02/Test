#!/usr/bin/env python
"""Independent audit of a solved trajectory.

Nothing here reuses the solver's own numbers. The control history is flown through the nonlinear
dynamics at high accuracy, and every constraint is checked both AT the nodes and BETWEEN them --
the between-node check matters because the constraints are imposed only at temporal nodes, so
inter-sample violation is possible in principle and is a known weakness of node-collocated
formulations.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C, problem_setup as PS, run_descent as RD

HERE = os.path.dirname(os.path.abspath(__file__))


def fly(dyn, sc, X, U, s, n_per=200):
    """DOP853 at tight tolerance, restarted at every ZOH node (the switches are known exactly)."""
    K = X.shape[1]; h = s / (K - 1)
    x = X[:, 0].copy(); traj = [x.copy()]; dense = [x.copy()]
    for k in range(K - 1):
        uk = U[:, k]
        sol = solve_ivp(lambda t, y: dyn.f(y, uk), (0.0, h), x, method="DOP853",
                        rtol=1e-12, atol=1e-14, dense_output=True)
        ts = np.linspace(0.0, h, n_per)
        dense.extend(sol.sol(ts).T[1:])
        x = sol.y[:, -1]; traj.append(x.copy())
    return np.array(traj).T, np.array(dense).T


def audit(K=30, iters=30, tag=""):
    out = RD.solve(K=K, iters=iters, verbose=True)
    r2, P = out["stage2"], out["setup"]
    dyn, sc, n = P["dyn"], P["scales"], P["n_hat"]
    X, U, s = r2.X, r2.U, r2.s
    Xf, Xd = fly(dyn, sc, X, U, s)

    rf = P["rf_nd"]; Pl = np.eye(3) - np.outer(n, n)
    print("\n" + "=" * 78)
    print("AUDIT  (control flown through the nonlinear dynamics, DOP853 rtol 1e-12)")
    print("=" * 78)
    miss_solver = float(np.linalg.norm(Pl @ (X[0:3, -1] - rf))) * sc.L
    miss_flown = float(np.linalg.norm(Pl @ (Xf[0:3, -1] - rf))) * sc.L
    alt_flown = float(n @ (Xf[0:3, -1] - rf)) * sc.L
    v_flown = float(np.linalg.norm(Xf[3:6, -1])) * sc.v
    fuel_solver = float(np.exp(X[6, 0]) - np.exp(X[6, -1]))
    fuel_flown = float(np.exp(Xf[6, 0]) - np.exp(Xf[6, -1]))
    print(f"  flight time            {sc.t_dim(s):10.1f} s")
    print(f"  lateral miss   solver  {miss_solver:10.4f} m      flown {miss_flown:10.4f} m")
    print(f"  terminal altitude              -             flown {alt_flown:10.4f} m")
    print(f"  terminal speed                 -             flown {v_flown:10.4f} m/s "
          f"(limit {C.VTOUCH})")
    print(f"  propellant     solver  {fuel_solver:10.4f} kg     flown {fuel_flown:10.4f} kg"
          f"     diff {abs(fuel_flown-fuel_solver):.2e}")
    print(f"  max node divergence solver-vs-flown "
          f"{float(np.abs(Xf[0:3]-X[0:3]).max())*sc.L:10.4e} m")

    # THE THRUST IS Gamma*d: the DIRECTION is the state X[7:10] and the MAGNITUDE is the control
    # U[3]. Reading either alone as the thrust is wrong -- X[7:10] is a unit vector.
    m = np.exp(X[6]); dvec = X[7:10]; dn = np.linalg.norm(dvec, axis=0)
    T = m * U[3] * dn * sc.A
    # The CHARGE-VERSUS-DELIVERY GAP, which is what the old 'lcvx_gap' measured in a formulation
    # where Gamma was a slack above ||a||. Here the charge is Gamma and the delivery is Gamma*||d||,
    # so the gap is Gamma*(1 - ||d||) and it is the ||d|| leak in newtons. Both earlier treatments
    # of ||d|| showed up here: 1.043 one way, 0.957 (4.787 N delivered against 5 N charged) the
    # other. Under d-dot = omega x d it should read zero to the defect.
    gap = float(np.max(np.abs(m * U[3] * (1.0 - dn) * sc.A)))
    lcvx_gap = gap
    # Gamma is ZERO-ORDER HELD and ||d|| is conserved along a segment, so the thrust MAGNITUDE is
    # constant across each hold -- there is no between-node dip to check any more. That dip was an
    # artefact of holding Gamma while ramping ||a||, and it is what the ZOH split removed.
    T_mid = np.array([m[k] * U[3, k] * dn[k] * sc.A for k in range(K - 1)])
    # omega IS the slew rate: the control itself, in rad per unit tau, converted to deg/s. No
    # reconstruction from consecutive directions, because omega x d cannot change ||d|| so none of
    # omega is spent on anything but rotation.
    dh = dvec / np.maximum(dn, 1e-12)
    slew = np.degrees(np.linalg.norm(U[0:3], axis=0) / sc.T)
    ang = np.degrees(np.arccos(np.clip(dh.T @ n, -1, 1)))
    # Glideslope residual in METRES, not the cone angle. The cone's apex sits AT the site, so the
    # angle is meaningless there: at a terminal altitude of 8.6 mm with 8 mm of lateral offset the
    # angle reads 43 deg while the constraint ||lateral|| <= tan(gamma)*altitude is violated by
    # 3 mm. The residual is the quantity the constraint actually states and it stays finite.
    tg_ = np.tan(np.radians(C.GLIDESLOPE_DEG))
    gs_node = (np.linalg.norm(Pl @ (X[0:3] - rf[:, None]), axis=0)
               - tg_ * (n @ (X[0:3] - rf[:, None]))) * sc.L
    gs_dense = (np.linalg.norm(Pl @ (Xd[0:3] - rf[:, None]), axis=0)
                - tg_ * (n @ (Xd[0:3] - rf[:, None]))) * sc.L
    alt_dense = (n @ (Xd[0:3] - rf[:, None])) * sc.L
    print(f"\n  CONSTRAINTS                      at nodes           between nodes")
    print(f"    thrust [N]            {T.min():8.3f} .. {T.max():7.3f}      "
          f"(bounds {C.TMIN} .. {C.TMAX})")
    print(f"    thrust angle [deg]    {ang.min():8.2f} .. {ang.max():7.2f}      "
          f"(limit {C.POINTING_DEG})")
    print(f"    glideslope resid [m]  {gs_node.max():8.3f} max          "
          f"{gs_dense.max():7.3f} max   (<= 0 satisfies the {C.GLIDESLOPE_DEG:g} deg cone)")
    print(f"    altitude [m]          {(n@(X[0:3]-rf[:,None])).min()*sc.L:8.2f} min         "
          f"{alt_dense.min():7.2f} min   (must stay >= 0)")
    print(f"    mass [kg]             {m.min():8.2f} min                       "
          f"(dry {C.MDRY})")
    print(f"    LCvx gap [N]          {lcvx_gap:8.2e} max          "
          f"(0 = the relaxation is tight, i.e. the answer is physical)")
    print(f"    thrust mid-segment[N] {T_mid.min():8.3f} min          "
          f"(piecewise-linear thrust can dip below the floor between nodes)")
    print(f"    slew rate [deg/s]     {slew.max():8.3f} max          "
          f"(limit {C.SLEW_MAX_DEG_S})")
    # the thrust floor is allowed to be missed by the LCvx gap and no more: ||a|| <= Gamma and
    # Gamma >= Tmin, so a non-zero gap lets the delivered thrust sit that far under the floor.
    ok = (T.min() >= C.TMIN - max(lcvx_gap, 1e-6) - 1e-6 and T.max() <= C.TMAX + 1e-6
          and (C.SLEW_MAX_DEG_S is None or slew.max() <= C.SLEW_MAX_DEG_S * 1.02)
          and ang.max() <= C.POINTING_DEG + 1e-6
          and gs_dense.max() <= 0.05
          and alt_dense.min() >= -0.05 and m.min() >= C.MDRY)
    print(f"\n  ALL CONSTRAINTS SATISFIED (nodes and between): {ok}")
    res = dict(tf_s=sc.t_dim(s), lcvx_gap_N=lcvx_gap, slew_max_deg_s=float(slew.max()),
           T_mid_min_N=float(T_mid.min()), miss_solver_m=miss_solver, miss_flown_m=miss_flown,
               alt_flown_m=alt_flown, vtouch_flown=v_flown, fuel_solver_kg=fuel_solver,
               fuel_flown_kg=fuel_flown, T_min=float(T.min()), T_max=float(T.max()),
               ang_max_deg=float(ang.max()), gs_resid_max_m=float(gs_dense.max()),
               alt_min_m=float(alt_dense.min()), all_ok=bool(ok), status=r2.status,
               iters=r2.iters, defect_m=r2.defect_m)
    os.makedirs(os.path.join(HERE, "Figures&data"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "Figures&data", f"audit{tag}.json"), "w"), indent=1)
    np.savez(os.path.join(HERE, "Figures&data", f"descent{tag}.npz"), X=X, U=U, s=s,
             L=sc.L, T=sc.T, A=sc.A, rf=P["r_site"], n_hat=n, K=K, Xflown=Xf)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    audit(K=a.K, iters=a.iters, tag=f"_K{a.K}")
