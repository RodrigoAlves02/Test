#!/usr/bin/env python
"""The constrained benchmark: Malyuta et al. arXiv:2106.09125, Problem (95), parameters (108).

Adds to the TFC cases the three things they did not exercise -- a rotating frame, a glideslope cone
on the position, and a tilt cone on the thrust. See benchmark.MAL for the parameters and for the
two recorded differences (their 4-facet glideslope circumscribes the exact cone; we omit their
||v|| <= vmax, which the paper states is never active here).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "SCvx"))
sys.path.insert(0, HERE)
import benchmark as B                                                        # noqa: E402

M = B.MAL
import config as C                                                           # noqa: E402

C.TMIN, C.TMAX = M["rho_min"], M["rho_max"]
C.MWET, C.MDRY = M["m_wet"], M["m_dry"]
C.ISP, C.G0, C.ALPHA = M["isp"], M["ge"], M["alpha"]
C.W_BODY = M["omega"]                       # the ROTATING FRAME is active in this case
C.VTOUCH = 0.0                              # r(tf) = v(tf) = 0 exactly
C.GLIDESLOPE_DEG = M["gamma_gs"]
C.POINTING_DEG = M["gamma_p"]
C.TERMINAL_POINT_DEG = 180.0
C.TERMINAL_POINT_NODES = 0
_tf0 = C.tf_bounds
C.tf_bounds = lambda v0n, rng, **kw: _tf0(v0n, rng, m_wet=M["m_wet"], m_dry=M["m_dry"],
                                          t_max=M["rho_max"], t_min=M["rho_min"],
                                          g_max=abs(M["g"][2]), v_touch=0.0)

import ptr                                                                   # noqa: E402
from dynamics import Dynamics                                                # noqa: E402


def run(K=76, fix_tf=None, verbose=False, iters=400, warm=None, slew_max_deg_s=None):
    """Solve the Malyuta case.

    iters DEFAULTS TO 400, NOT 60. From a COLD start the flight time has to climb from the seed's
    25.7 s to about 75 s with the state trust region saturated the whole way, and K=38 was measured
    converging only at iteration 207. Cut off at 60 it stops with the mass pinned at the DRY limit
    and tf 9.4% low -- a non-solution that still plots, which is how a stale figure gets made. Check
    the status, not just the picture.

    slew_max_deg_s DEFAULTS TO None, which removes the constraint entirely rather than making it
    slack -- Malyuta imposes no rate limit, so adding one would be solving a different problem and
    the comparison with the published tf would not be a verification. The argument exists so the
    slew machinery can be exercised ON a literature-verified case: set it far above the trajectory's
    natural turn rate and the answer must not move. That is the only external check available for
    the rate limit, since no published case in this folder constrains it.

    PASS A WARM START FOR THE FINER MESHES; see solve_ladder below. The cold start does not scale
    with K, and that is the whole story of this case:

        K     start   result                      iterations
        38    cold    converged, tf +0.82%        207
        76    cold    maxiter,   tf +0.89%        600   -> warm: converged, +0.81%, 25 it
        151   cold    subproblem_unbounded        29    -> warm: converged, +0.67%, 18 it

    Cold at K=151 the flight time creeps 0.07 s per iteration -- it needs 300+ more to reach 75 s --
    while the state trust region sits at 100% every iteration and the flight-time region at 3%. The
    radius then shrinks on the defect rule, the mass equation absorbs 150 kg of virtual control with
    the mass pinned at the dry limit, velocity defects reach 38 m/s, and the SOCP degenerates
    (CLARABEL user_limit, SCS unbounded). None of that is the mesh being too fine: a converged coarse
    solution already has tf at ~75 s, so the fine mesh never makes the climb. ptr.run resamples a
    warm start onto whatever K is asked for, so no extra machinery is needed.
    """
    r0, v0, rf = M["r0"], M["v0"], np.zeros(3)
    n_hat = np.array([0.0, 0.0, 1.0])
    sc = C.Scales.build(r0, rf, m_wet=M["m_wet"], t_max=M["rho_max"])
    dyn = Dynamics(sc, field=B.ConstantField(M["g"]))
    x_init = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(M["m_wet"])], n_hat])  # +d
    r = ptr.run(dyn=dyn, scales=sc, x_init=x_init, rf_nd=rf / sc.L, n_hat=n_hat,
                K=K, mode="fuel", D_nd=0.0, iters=iters, verbose=verbose,
                slew_max_deg_s=slew_max_deg_s, a_init=None,
                term_point_deg=180.0, term_point_nodes=0,
                gs_deg=M["gamma_gs"], point_deg=M["gamma_p"],
                # tol_vc_m is NOT loosened here any more. It used to be 1e-2, ten times the
                # solver default, because the old convergence test multiplied the SUM over all ten
                # state rows by L -- a conversion correct only for the three position rows -- and
                # these cases could not pass the real 1e-3 m bar against a number that was mostly a
                # dimensionless direction residual wearing metres. With the test split per channel
                # (ptr.run), all four benchmark meshes converge at the DEFAULT tolerances and return
                # propellant identical to four decimals, so the workaround is gone. Do not put it
                # back without checking vc_worst first: it hides real position defects.
                warm=warm, fix_tf_s=fix_tf, tol_obj=1e-6, tol_defect_m=1e-2)
    return r, sc



def solve_ladder(Ks=(38, 76, 151), verbose=False, slew_max_deg_s=None):
    """Solve the coarsest mesh cold, then warm-start each finer one from the previous solution.

    This is how this case should be run. Solved cold at every mesh the flight-time error does not
    refine (+0.82 / +0.89 / fails); on the ladder it does: +0.82% -> +0.81% -> +0.67%, which is the
    mesh convergence a verification case is supposed to show.

    Yields (K, Result, scales) so a caller can report or plot any rung. A rung that fails to converge
    is still carried forward -- it is strictly better information than the cold straight line -- but
    the status is returned untouched so the caller can see it.
    """
    warm = None
    for K in Ks:
        r, sc = run(K, verbose=verbose, warm=warm, slew_max_deg_s=slew_max_deg_s)
        warm = (r.X, r.U, r.s)
        yield K, r, sc


def audit(r, sc):
    m = np.exp(r.X[6, :]); K = r.X.shape[1]
    t = np.linspace(0.0, sc.t_dim(r.s), K)
    pos = r.X[0:3, :] * sc.L
    vel = r.X[3:6, :] * sc.v
    # thrust is Gamma * d now: magnitude from the control, direction from the state
    dvec = r.X[7:10, :]
    Tc = r.U[3, :] * dvec * sc.A * m
    sig = r.U[3, :] * sc.A * m                      # the magnitude variable itself
    nT = np.linalg.norm(Tc, axis=0)
    e3 = np.array([0.0, 0.0, 1.0])
    P = np.eye(3) - np.outer(e3, e3)
    tilt = np.degrees(np.arccos(np.clip(Tc[2] / np.maximum(nT, 1e-12), -1, 1)))
    # GLIDESLOPE, MEASURED THE WAY THE CONSTRAINT IS WRITTEN: a cone with its apex at the ACHIEVED
    # landing point r_K, and imposed on k = 0..K-2 only. Two things were wrong with reading it as an
    # angle from the ORIGIN over all nodes, and together they manufactured a violation that does not
    # exist: this solution landed 8.3e-11 m from the apex, so arccos(z/|r|) at the terminal node is
    # 0/0 and returned 90.39 deg against an 86 deg limit, while the constraint residual there is
    # exactly zero. Over the nodes the cone actually binds, the residual is -3.877e-3 m -- satisfied,
    # and tight: the apex-relative angle reads 86.0000 deg.
    #
    # So the residual IN METRES is what is reported, which is the quantity the constraint states and
    # which stays finite at the apex. The angle is kept alongside but computed apex-relative and only
    # over the constrained nodes, with the degenerate terminal node excluded rather than clipped.
    dr = pos - pos[:, K - 1:K]
    tg = np.tan(np.radians(M["gamma_gs"]))
    gs_res = np.linalg.norm(P @ dr, axis=0) - tg * (e3 @ dr)          # <= 0 means satisfied
    gs = np.degrees(np.arctan2(np.linalg.norm(P @ dr, axis=0), np.maximum(e3 @ dr, 1e-30)))
    return dict(t=t, pos=pos, vel=vel, m=m, Tc=Tc, sig=sig, nT=nT, tilt=tilt,
                gs=gs[:K - 1], gs_res_m=gs_res[:K - 1],
                sp=np.linalg.norm(vel, axis=0), tf=sc.t_dim(r.s),
                used=M["m_wet"] - m[-1], lcvx=np.abs(sig - nT).max())


if __name__ == "__main__":
    print(f"MALYUTA case  published tf* = {M['tf']} s, ZOH dt = 1 s   [MESH LADDER: "
          f"38 cold, then each mesh warm-started from the previous -- see run()]")
    print(f"  gamma_gs {M['gamma_gs']} deg, gamma_p {M['gamma_p']} deg, omega {M['omega_deg_s']} deg/s\n")
    print(f"{'K':>5} {'status':>12} {'tf':>9} {'err':>8} {'m_used':>10} {'max tilt':>9} "
          f"{'max gs':>8} {'max |v|':>9} {'LCvx gap':>10} {'|r_f|':>9} {'gs res[m]':>10}")
    for K, r, sc in solve_ladder():
        a = audit(r, sc)
        rf = float(np.linalg.norm(r.X[0:3, -1])) * sc.L
        print(f"{K:5d} {r.status:>12} {a['tf']:9.4f} {100*(a['tf']-M['tf'])/M['tf']:+7.2f}% "
              f"{a['used']:10.4f} {a['tilt'].max():8.2f} {a['gs'].max():7.2f} {a['sp'].max():8.2f} "
              f"{a['lcvx']:10.3e} {rf:9.1e} {a['gs_res_m'].max():+10.2e}")
