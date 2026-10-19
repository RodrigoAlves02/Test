#!/usr/bin/env python
"""Run the published benchmark through OUR solver and report the discrepancy. See benchmark.py.

Usage:  run_verification.py [--case min-max] [--K 60] [--no-segment-lcvx]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.join(HERE, "..", "SCvx")
sys.path.insert(0, SCVX)
sys.path.insert(0, HERE)

import benchmark as B                                                        # noqa: E402

# ---- CONFIGURE BEFORE IMPORTING THE SOLVER --------------------------------------------------
# subproblem.build binds gs_deg, point_deg and the tf_bounds defaults at DEF time, so the Eros
# values would be baked in if config were patched after the import. Patching first is what lets
# the solver be imported unmodified.
import config as C                                                           # noqa: E402

C.TMIN, C.TMAX = B.TMIN, B.TMAX
C.MWET, C.MDRY = B.M_WET, B.M_DRY
C.ISP, C.G0, C.ALPHA = B.ISP, B.G0, B.ALPHA
C.W_BODY = np.zeros(3)              # no rotating frame: kills Coriolis and centrifugal
C.VTOUCH = 0.0                      # exact soft landing, not our 0.3 m/s touchdown box
C.GLIDESLOPE_DEG = 89.9             # >= 89 disables the cone
C.POINTING_DEG = 180.0              # no thrust-pointing constraint in the benchmark
C.TERMINAL_POINT_DEG = 180.0        # no terminal attitude constraint either
C.TERMINAL_POINT_NODES = 0
# The benchmark is free-final-time. tf_bounds binds Eros's mass, thrust and gravity as DEFAULT
# arguments, so it is re-pointed at the same formula evaluated with the Mars vehicle -- the bracket
# is not under test, the optimum inside it is, and inventing a loose one is not free: a stub
# returning (1, 300) s put the initial guess at 2 s and the solve never escaped it.
_tf_bounds_orig = C.tf_bounds
C.tf_bounds = lambda v0n, rng, **kw: _tf_bounds_orig(
    v0n, rng, m_wet=B.M_WET, m_dry=B.M_DRY, t_max=B.TMAX, t_min=B.TMIN,
    g_max=abs(B.G_MARS[2]), v_touch=0.0)

import ptr                                                                   # noqa: E402
import subproblem as SP                                                      # noqa: E402
from dynamics import Dynamics                                                # noqa: E402


def run(case="min-max", K=60, segment_lcvx=True, verbose=False):
    _ = segment_lcvx   # retired: with ZOH, a and Gamma are held alike and the node bound suffices
    cs = B.CASES[case]
    r0, v0 = cs["r0"], cs["v0"]
    rf = np.zeros(3)
    n_hat = np.array([0.0, 0.0, 1.0])            # "local vertical" = the landing-site normal
    sc = C.Scales.build(r0, rf, m_wet=B.M_WET, t_max=B.TMAX)
    dyn = Dynamics(sc, field=B.ConstantField())
    # 10 states now: [r, v, z, d]. The initial direction is free, so seed it along the vertical;
    # the solver moves it. a_init=None leaves both the handover direction and magnitude free.
    x_init = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(B.M_WET)], n_hat])
    r = ptr.run(dyn=dyn, scales=sc, x_init=x_init, rf_nd=rf / sc.L, n_hat=n_hat,
                K=K, mode="fuel", D_nd=0.0, iters=60, verbose=verbose,
                slew_max_deg_s=None, a_init=None,
                term_point_deg=180.0, term_point_nodes=0,
                # tol_vc_m is NOT loosened here any more. It used to be 1e-2, ten times the
                # solver default, because the old convergence test multiplied the SUM over all ten
                # state rows by L -- a conversion correct only for the three position rows -- and
                # these cases could not pass the real 1e-3 m bar against a number that was mostly a
                # dimensionless direction residual wearing metres. With the test split per channel
                # (ptr.run), all four benchmark meshes converge at the DEFAULT tolerances and return
                # propellant identical to four decimals, so the workaround is gone. Do not put it
                # back without checking vc_worst first: it hides real position defects.
                tol_obj=1e-6, tol_defect_m=1e-2)
    return r, sc, cs


def thrust_N(r, sc):
    """Thrust magnitude in NEWTONS at every node. The thrust is Gamma*d, so its magnitude is
    Gamma*||d||, and Gamma is a non-dimensional ACCELERATION -- multiply by sc.A for m/s^2 and by
    the mass for force. ||d|| is carried explicitly rather than assumed to be 1: that is the whole
    quantity under test here.
    """
    return r.U[3, :] * np.linalg.norm(r.X[7:10, :], axis=0) * sc.A * np.exp(r.X[6, :])


def true_cost(r, sc):
    """Propellant the thrust profile ACTUALLY costs: integrate the solver's own mass ODE with
    the DELIVERED acceleration ||Gamma*d|| = Gamma*||d|| instead of the CHARGED Gamma.

    The two differ by exactly the ||d|| leak, which is why this is the number reported rather than
    the solver's own. dynamics.f charges mass flow on Gamma; if ||d|| < 1 the vehicle delivers less
    thrust than it pays for and the charge is an overestimate, and if ||d|| > 1 it delivers more
    than it pays for and the charge is free propellant. Under the cross-product kinematics
    d-dot = omega x d the norm is conserved, so these should now agree to the defect -- and when
    they do, that agreement is the result, not an assumption.

    Gamma is ZERO-ORDER HELD and ||d|| is conserved across a segment, so the delivered magnitude is
    CONSTANT on each hold and the integral over a segment is exact without quadrature.
    """
    K = r.X.shape[1]
    dtau, al = 1.0 / (K - 1), C.ALPHA * sc.A * sc.T
    G = r.U[3, :]
    dn = np.linalg.norm(r.X[7:10, :], axis=0)
    z = float(r.X[6, 0])
    for k in range(K - 1):
        z -= al * G[k] * dn[k] * r.s * dtau
    return float(np.exp(r.X[6, 0]) - np.exp(z))


def switch_times(r, sc, frac=0.5):
    """Times at which ||T|| crosses the midpoint of its range -- the bang-bang switches.

    The thrust magnitude is ZERO-ORDER HELD, so it does not cross anything continuously: it JUMPS
    at a node. The switch time is therefore the node boundary itself, t_{k+1}, and the resolution
    of the answer is one node spacing -- no interpolation, because interpolating a discontinuous
    signal invents a ramp the solution does not contain. Reporting it this way is what makes the
    comparison with the published switch honest about its own resolution.
    """
    a = thrust_N(r, sc)
    t = np.linspace(0.0, sc.t_dim(r.s), a.size)
    mid = 0.5 * (B.TMIN + B.TMAX)
    return [t[k + 1] for k in range(a.size - 1) if (a[k] - mid) * (a[k + 1] - mid) < 0]


def report(r, sc, cs, label):
    m0 = B.M_WET
    used = m0 - float(np.exp(r.X[6, -1]))
    tf = sc.t_dim(r.s)
    rf_err = float(np.linalg.norm(r.X[0:3, -1])) * sc.L
    vf_err = float(np.linalg.norm(r.X[3:6, -1])) * sc.v
    tc = true_cost(r, sc)
    sw = switch_times(r, sc)
    Tn = thrust_N(r, sc)
    print(f"  {label:<18} {r.status:<11} charged {used:9.4f}  TRUE {tc:9.4f} kg "
          f"({100*(tc-cs['m_used'])/cs['m_used']:+6.2f}%)  tf {tf:8.4f} ({100*(tf-cs['tf'])/cs['tf']:+5.2f}%)  "
          f"sw {[round(x,3) for x in sw]}  |r_f| {rf_err:.1e}")
    print(f"  {'':<18} |d| [{r.dnorm_lo:.6f}, {r.dnorm_hi:.6f}]   T [{Tn.min():.1f}, {Tn.max():.1f}] N "
          f"(floor {B.TMIN:.1f}, cap {B.TMAX:.1f})   charged-vs-true {used - tc:+.4f} kg   "
          f"{r.iters} it, {r.wall_s:.0f} s")
    return used, tc, tf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="min-max", choices=list(B.CASES))
    ap.add_argument("--Ks", type=int, nargs="+", default=[30, 60, 120])
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    cs = B.CASES[a.case]
    print(f"CASE '{a.case}'   published: m_used {cs['m_used']} kg, tf {cs['tf']} s   [{cs['note']}]")
    print(f"  r0 {cs['r0']}  v0 {cs['v0']}  m0 {B.M_WET} kg")
    print(f"  published switches: {cs['switches']} s\n")
    for K in a.Ks:
        print(f"  --- K = {K} ---")
        for seg in (True, False):
            r, sc, cs = run(a.case, K, segment_lcvx=seg, verbose=a.verbose)
            report(r, sc, cs, "segment-LCvx" if seg else "node-only")
