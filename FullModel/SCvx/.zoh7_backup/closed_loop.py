#!/usr/bin/env python
"""Closed-loop descent: re-plan every cycle, fly the plan through the actuator, propagate truth.

The guidance cycle here is not a real-time budget -- the online solve is to be replaced by a
network -- so the cycle length is chosen for fidelity, not speed.

Three things are carried over from what the previous closed-loop test measured, because each was a
real failure there:

  * THE ACTUATOR IS A ZOH DEVICE, NOT AN ODE STATE. Writing a slew rate limiter as a differential
    equation makes it a switch: the direction overshoots the command and reverses, which is a
    sliding mode, and an adaptive integrator cannot resolve it -- one 30 s arc took 225 s and
    518 639 gravity calls against 0.3 s for its neighbour. Here the thrust is held constant over a
    short sub-step and the actuator is advanced analytically at the boundaries.

  * FOLLOW THE PLAN ACROSS THE CYCLE, DO NOT HOLD u(tau=0). Minimum-fuel control is bang-bang, so
    the optimal switch falls inside a cycle; holding the first node defers it by up to a full cycle,
    and the minimum-fuel trajectory has no margin to absorb that.

  * START THE ACTUATOR WHERE THE VEHICLE ACTUALLY IS. The engine is already lit at handover.
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
sys.path.insert(0, os.path.join(HERE, "..", "Guidance and Control"))

import config as C
import ptr
from dynamics import Dynamics
import problem_setup as PS
from actuator_model import Actuator                                      # noqa: E402


def plan_thrust(X, U, s, dt_ahead, scales):
    """Thrust the plan calls for, dt_ahead (non-dimensional) after it was made. Newtons.

    The thrust is a STATE and the rate is held, so a is piecewise LINEAR between nodes: interpolate
    it rather than holding the node value. That continuity is the point of the formulation -- a
    held thrust steps at every node boundary, and an actuator with a 2 deg/s slew limit needs
    seconds to follow a step, which it spends pointing the wrong way.
    """
    K = X.shape[1]
    h = s / (K - 1)
    kf = float(np.clip(dt_ahead / h, 0.0, K - 1 - 1e-12))
    k = int(np.floor(kf)); lam = kf - k
    # ZERO-ORDER HOLD: the plan commands U[0:3, k] for the whole of segment k. Interpolating
    # between nodes would be flying a ramp the plan never asked for -- and the plan's own
    # discretisation, its mass flow and its LCvx bound all assume the hold.
    a = U[0:3, k]
    m = float(np.exp((1.0 - lam) * X[6, k] + lam * X[6, k + 1]))
    return m * a * scales.A, k


def rescale_plan(X, U, s, sc_old, sc_new):
    """Move a plan from one non-dimensionalisation to another.

    The scales are chosen so the problem's own variables are O(1). Held fixed at the handover values
    (L = 11771 m, T = 600 s) they stop doing that as the vehicle closes in: at 87 m and 72 s to go,
    positions run at 0.007 and sigma at 0.12, i.e. under 1% of the magnitude the scaling was built
    for, while the trust radius still reads 0.35 -- larger than the whole remaining trajectory.
    CLARABEL reported "solution may be inaccurate" at exactly that cycle.

    States convert through their physical values. The rate control is then REBUILT from the
    converted thrust state rather than converted directly: w = da/dtau carries both the scale change
    and the dilation, and rebuilding it by finite difference is exact and impossible to get wrong.
    """
    K = X.shape[1]
    Xn = np.array(X, float)
    Xn[0:3] = X[0:3] * sc_old.L / sc_new.L
    Xn[3:6] = X[3:6] * (sc_old.L / sc_old.T) * (sc_new.T / sc_new.L)
    s_new = s * sc_old.T / sc_new.T
    Un = np.array(U, float)
    Un[0:3] = U[0:3] * sc_old.A / sc_new.A                 # thrust is the control under ZOH
    Un[3] = np.linalg.norm(Un[0:3], axis=0)                # LCvx-tight: Gamma = |a|
    return Xn, Un, s_new


def nominal_warm(nom, sc, x_plan):
    """Seed CYCLE 0 from the nominal trajectory instead of a straight line.

    Every other cycle inherits shift_plan(previous solution) and converges in single digits.
    Cycle 0 has nothing to inherit, so it cold-starts from ptr.initial_guess -- a cubic Hermite
    through the boundary conditions with the thrust backed out of it. That guess is dynamically
    consistent but it is not a TRAJECTORY, and cycle 0 is the one cycle that shows it: 30
    iterations, maxiter, and the only defect in the whole descent (9.68e-3 m) above tol_defect_m.
    It is also the hardest cycle -- 7 states pinned, a(0) free, tf free, 12 km to fly.

    A real vehicle does not cold-start an optimiser at handover. It carries a reference trajectory
    computed on the ground, which is exactly what this is: the converged open-loop nominal, moved
    onto this cycle's scales and shifted onto the state the filter is reporting.

    THE SHIFT DECAYS. A rigid translation would drag the guess's terminal point off the surface by
    the full offset, and the terminal constraint n.(r_K - rf) == 0 wants it on. The weight is the
    Hermite basis h00(tau) = 2tau^3 - 3tau^2 + 1: one at the start, zero at the end, and FLAT at
    both, so neither endpoint's velocity is contaminated by the decay. The velocity correction
    dr*h00'/s is what keeps r-dot = v holding through the middle of the guess rather than only at
    its ends -- ptr.run then overwrites node 0 with the exact IC anyway.

    The landing point needs no adjustment. Stage 1 IS the minimum-landing-error problem, so its
    terminal position is a free variable; the nominal's is already the right guess whether or not
    the site turns out to be reachable, and the D > 0 case resolves itself in the solve.
    """
    Xn, Un, sn = rescale_plan(nom["X"], nom["U"], nom["s"], nom["scales"], sc)
    K = Xn.shape[1]
    tau = np.linspace(0.0, 1.0, K)
    h00 = 2.0 * tau ** 3 - 3.0 * tau ** 2 + 1.0          # 1 -> 0, flat at both ends
    d00 = 6.0 * tau ** 2 - 6.0 * tau                     # its derivative
    dr = np.asarray(x_plan[0:3], float) - Xn[0:3, 0]
    dv = np.asarray(x_plan[3:6], float) - Xn[3:6, 0]
    Xn[0:3] = Xn[0:3] + np.outer(dr, h00)
    Xn[3:6] = Xn[3:6] + np.outer(dv, h00) + np.outer(dr, d00) / sn
    Xn[6] = Xn[6] + (float(x_plan[6]) - float(Xn[6, 0]))  # mass offset persists down the arc
    return Xn, Un, sn


def shift_plan(X, U, s, frac):
    """The remaining plan, resampled onto the same node count -- the next cycle's warm start."""
    K = X.shape[1]
    tau = np.linspace(0.0, 1.0, K)
    tq = np.clip(frac + np.linspace(0.0, 1.0, K) * (1.0 - frac), 0.0, 1.0)
    Xs = np.array([np.interp(tq, tau, X[i]) for i in range(X.shape[0])])
    Us = np.array([np.interp(tq, tau, U[i]) for i in range(U.shape[0])])
    return Xs, Us, s * (1.0 - frac)


def rhs(dyn, x7, T_vec_N, scales):
    """TRUTH dynamics: 7 states [r, v, z] with the thrust supplied EXTERNALLY, in newtons.

    The plan carries thrust as an 8th-10th state because that is what makes the slew rate a convex
    constraint and the thrust continuous. The vehicle does not: its thrust is whatever the actuator
    is producing, which is an input here, not a state to be integrated. Keeping the two separate is
    what stops the plan's intended thrust from being mistaken for the delivered one.
    """
    r, v = x7[0:3], x7[3:6]
    m = float(np.exp(x7[6]))
    a = np.asarray(T_vec_N, float) / m / scales.A
    vdot = a + dyn.Cco @ v + dyn.Ccf @ r + dyn.g_nd(r)
    return np.concatenate([v, vdot, [-dyn.alpha * float(np.linalg.norm(a))]])


def rk4(dyn, x7, T_vec_N, h, scales):
    k1 = rhs(dyn, x7, T_vec_N, scales)
    k2 = rhs(dyn, x7 + 0.5 * h * k1, T_vec_N, scales)
    k3 = rhs(dyn, x7 + 0.5 * h * k2, T_vec_N, scales)
    k4 = rhs(dyn, x7 + h * k3, T_vec_N, scales)
    return x7 + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--iters", type=int, default=30)
    # Stage 1 only has to produce D and a seed for stage 2, and when the site is
    # reachable its optimum is DEGENERATE -- every feasible landing scores zero miss --
    # so it has no preference and will wander until it runs out of iterations. Measured
    # hitting maxiter on every cycle at 30 iterations and a 5 mm defect tolerance.
    # Give it a short budget and a loose tolerance; stage 2 keeps the tight ones.
    ap.add_argument("--iters1", type=int, default=10)
    ap.add_argument("--tol1-m", type=float, default=1.0)
    ap.add_argument("--dt-guid", type=float, default=60.0, help="guidance cycle [s]")
    ap.add_argument("--dt-act", type=float, default=0.5, help="actuator sub-step [s]")
    ap.add_argument("--dt-prop", type=float, default=0.25, help="truth RK4 step [s]")
    ap.add_argument("--tau", type=float, default=0.5, help="thrust lag [s]")
    ap.add_argument("--slew", type=float, default=2.0, help="slew limit [deg/s]")
    ap.add_argument("--max-cycles", type=int, default=80)
    ap.add_argument("--max-minutes", type=float, default=90.0)
    ap.add_argument("--trace-stride", type=int, default=1,
                    help="write every Nth trace sample (the last one is always kept). The trace is "
                         "90%% of the output file; a Monte Carlo of hundreds of runs does not need "
                         "it at the 0.5 s actuator cadence.")
    ap.add_argument("--from-nominal", action="store_true",
                    help="start from the real handover state in descent_nominal.npz")
    ap.add_argument("--pos-sigma", type=float, default=0.0, help="IC position dispersion [m]")
    ap.add_argument("--vel-sigma", type=float, default=0.0, help="IC velocity dispersion [m/s]")
    ap.add_argument("--seed", type=int, default=0, help="dispersion seed (recorded in the output)")
    ap.add_argument("--nav", action="store_true",
                    help="close the loop on an EKF estimate instead of the true state: the filter "
                         "starts from an OD error, propagates on the IMU with PINN gravity, and is "
                         "never corrected (no camera/altimeter yet)")
    ap.add_argument("--nav-seed", type=int, default=1)
    ap.add_argument("--imu-grade", default=None, help="tactical | navigation")
    ap.add_argument("--imu-rate", type=float, default=None, help="IMU sample rate [Hz]")
    ap.add_argument("--od-pos", type=float, default=None, help="OD position 1-sigma [m]")
    ap.add_argument("--od-vel", type=float, default=None, help="OD velocity 1-sigma [m/s]")
    ap.add_argument("--no-altimeter", dest="altimeter", action="store_false", default=True)
    ap.add_argument("--no-cameras", dest="cameras", action="store_false", default=True)
    ap.add_argument("--perfect-imu", action="store_true",
                    help="zero every IMU error, leaving only the OD error and the gravity gap")
    ap.add_argument("--dmc-sigma", type=float, default=None,
                    help="DMC random-walk strength [m/s^2 per sqrt(s)]. 0 disables dynamic model "
                         "compensation entirely (the states are frozen at zero with zero prior), "
                         "which is the A/B control. Default: config_nav.EKF_DMC_SIGMA.")
    ap.add_argument("--nav-gravity", choices=("poly", "pinn"), default="pinn",
                    help="gravity the EKF propagates on. SEPARATE from --plan-gravity on purpose: "
                         "the onboard FILTER has to run a cheap model, but the SCvx guidance here "
                         "is a ground-side data generator being replaced by a PINN, so it may use "
                         "the exact field. Defaults to pinn so the navigation is unchanged.")
    ap.add_argument("--plan-gravity", choices=("poly", "pinn"), default="poly",
                    help="gravity model the GUIDANCE plans with. The truth is always the 49 152-"
                         "facet polyhedral field, so 'pinn' introduces a real onboard model error")
    ap.add_argument("--nominal", default=os.path.join(HERE, "Figures&data", "descent.npz"),
                    help="converged open-loop nominal used to warm-start CYCLE 0 only. Every "
                         "later cycle warm-starts from its predecessor as before.")
    ap.add_argument("--no-nominal-warm", dest="nominal_warm", action="store_false", default=True,
                    help="cold-start cycle 0 from the straight-line guess instead")
    ap.add_argument("--cam-period", type=float, default=None,
                    help="seconds between camera fixes, per camera. Default is config_nav's "
                         "CAM_RATE_HZ (60 s, the autoNGC cadence).")
    ap.add_argument("--tol-obj", type=float, default=1e-4,
                    help="relative convergence tolerance on the stage objective (fuel in kg for "
                         "stage 2). 1e-3 = agree to 0.1 %% of the burn.")
    ap.add_argument("--commit-alt", type=float, default=5.0,
                    help="below this altitude [m] the guidance stops re-planning and flies the "
                         "last converged plan to contact, holding its final thrust command once "
                         "the plan runs out. 0 disables.")
    ap.add_argument("--tol-vc-m", dest="tol_vc_m", type=float, default=1e-3,
                    help="virtual-control tolerance in METRES of position (nu * L). Physical, so "
                         "the bar means the same thing at every cycle however far L has collapsed.")
    ap.add_argument("--tol-abs", type=float, default=1e-4,
                    help="absolute objective tolerance for stage 2 [kg]")
    ap.add_argument("--engine-off", action="store_true",
                    help="the engine is SHUT DOWN at handover: delivered thrust starts at 0 N and "
                         "the first-order lag has to bring it up to what the plan asks for")
    ap.add_argument("--no-rescale", dest="rescale", action="store_false", default=True,
                    help="keep the handover scaling for the whole descent (the old behaviour)")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()

    r0_ov = v0_ov = None
    if a.from_nominal:
        zn = np.load(os.path.join(C.THESIS, "scvx_landing", "descent_nominal.npz"))
        r0_ov = np.asarray(zn["X"][0:3, 0], float)
        v0_ov = np.asarray(zn["X"][3:6, 0], float)
        m_hand = float(np.exp(zn["X"][6, 0]))
    else:
        m_hand = C.MWET
    if a.pos_sigma > 0 or a.vel_sigma > 0:
        rng = np.random.default_rng(a.seed)
        def sphere():
            u = rng.normal(size=3); return u / np.linalg.norm(u)
        if r0_ov is None:
            raise SystemExit("dispersion requires --from-nominal")
        dr = sphere() * rng.normal(0.0, a.pos_sigma)
        dv = sphere() * rng.normal(0.0, a.vel_sigma)
        r0_ov = r0_ov + dr; v0_ov = v0_ov + dv
        print(f"DISPERSION seed {a.seed}: |dr| {np.linalg.norm(dr):.3f} m, "
              f"|dv| {np.linalg.norm(dv):.4f} m/s")
        print(f"  dr = {dr}\n  dv = {dv}")
    P = PS.build(r0_override=r0_ov, v0_override=v0_ov, m_wet=m_hand)
    sc, n = P["scales"], P["n_hat"]
    # TWO fields: the guidance plans with plan_fld, the world runs on truth_fld.
    # When they are the same object the loop is being told the truth exactly, which
    # is what every run before this one assumed.
    import gravity as GR
    truth_fld = GR.field()
    plan_fld = GR.pinn_field() if a.plan_gravity == "pinn" else truth_fld
    # THREE fields now, not two. The guidance and the filter had been sharing plan_fld, so asking
    # the solver for exact gravity silently gave it to the EKF as well and the "gravity-model gap"
    # the filter exists to measure would have vanished without anyone saying so.
    nav_fld = GR.pinn_field() if a.nav_gravity == "pinn" else truth_fld
    dyn = Dynamics(sc, field=plan_fld)      # what the solver believes
    dyn_t = Dynamics(sc, field=truth_fld)   # what actually happens
    rf_nd = P["rf_nd"]; Pl = np.eye(3) - np.outer(n, n)

    def ground_h(r_nd):
        """Height above the REAL surface [m], from a NON-DIMENSIONAL state.

        Everything on the truth side that asks "where is the ground" asks this. It used to be
        n . (r - r_site), the height above the TANGENT PLANE at the site, while the lidar
        ray-cast against the mesh -- and the facet under the site is tilted 21.3 deg from the
        local vertical, so the two diverged by ~0.39 m per metre of lateral miss. A run then
        reported altitude +0.268 m while sitting 0.58 m INSIDE the body, and the lidar, correctly
        ray-casting from a point inside a 34 km body, returned 11 112 m off the far wall. The
        filter had no innovation gate and swallowed it for a 414 m position jump.

        The GUIDANCE keeps its terminal constraint on the tangent plane: that is what keeps the
        subproblem convex, and the two surfaces agree exactly at the site, which is where it aims.
        """
        return GR.height_above_surface(np.asarray(r_nd, float)[0:3] * sc.L, n)
    act = Actuator(tau=a.tau, slew_max_deg_s=a.slew, t_min=C.TMIN, t_max=C.TMAX)
    # The truth state is carried DIMENSIONALLY (r [m], v [m/s], z) so that rescaling the solver
    # each cycle cannot disturb it. It is converted into the current scales only to be propagated.
    r_site = P["r_site"]
    xd = np.concatenate([P["x_init"][0:3] * sc.L, P["x_init"][3:6] * sc.v, [np.log(m_hand)]])
    x = np.concatenate([xd[0:3] / sc.L, xd[3:6] / sc.v, [xd[6]]])
    m0 = float(np.exp(xd[6]))

    print(f"handover: altitude {float(n @ (xd[0:3]-r_site))/1e3:.3f} km, "
          f"|v| {np.linalg.norm(xd[3:6]):.4f} m/s, m {m0:.2f} kg")
    print(f"per-cycle rescaling: {'ON' if a.rescale else 'OFF'}   "
          f"plan gravity: {a.plan_gravity}   nav gravity: {a.nav_gravity}   "
          f"truth gravity: poly 49k")
    print(f"K {a.K}, guidance cycle {a.dt_guid:g} s, actuator sub-step {a.dt_act:g} s, "
          f"tau {a.tau:g} s, slew {a.slew:g} deg/s\n")

    # ---- NAVIGATION -----------------------------------------------------------------------------
    # The truth starts at the nominal handover state. The FILTER starts at that state plus an OD
    # error draw, and from here on the guidance sees only the filter. With no position-fixing
    # sensor the filter can never discover the offset -- it can only add IMU drift to it -- so this
    # run measures what the OD error alone costs, which is the baseline the cameras must beat.
    nav = imu_dev = None
    wz = float(C.W_BODY[2])
    if a.nav:
        sys.path.insert(0, os.path.join(HERE, "..", "Navigation"))
        import config_nav as CN, ekf as NEKF, imu as NIMU, frames as NFR
        globals().update(CN=CN, NEKF=NEKF, NIMU=NIMU, NFR=NFR)
        rng = np.random.default_rng(a.nav_seed)
        ip = CN.imu_params(a.imu_grade, a.imu_rate)
        imu_dev = NIMU.IMU(grade=a.imu_grade, rate=a.imu_rate, seed=a.nav_seed)
        if a.perfect_imu:
            imu_dev.perfect(); ip = dict(imu_dev.p)
        # The truth is in ACAF; the filter is in ACI. initial_state_and_cov does the conversion,
        # including the w x r term that velocity needs and position does not. At t = 0 the two
        # frames coincide, so this is the identity for a nominal handover -- written out anyway so
        # a run starting at t != 0 is not silently wrong.
        x0n, P0n, dr0, dv0 = NEKF.initial_state_and_cov(
            xd[0:3], xd[3:6], rng,
            pos_sigma=a.od_pos if a.od_pos is not None else CN.OD_POS_SIGMA,
            vel_sigma=a.od_vel if a.od_vel is not None else CN.OD_VEL_SIGMA,
            t=0.0, wz=wz)
        # --dmc-sigma 0 is the control: freeze the DMC states at zero AND give them no prior, so
        # no measurement can move them and the filter is exactly the 9-state one again.
        if a.dmc_sigma is not None and a.dmc_sigma == 0.0:
            P0n[CN.IA, CN.IA] = 0.0
        nav = NEKF.EKF(x0n, P0n, nav_fld, ip, wz=wz, dmc_sigma=a.dmc_sigma)
        import perception as NPER
        globals().update(NPER=NPER)
        sensors = []
        if a.altimeter:
            sensors.append(("lidar", NPER.Lidar()))
        if a.cameras:
            # boresights along +thrust and -thrust: the forward camera sees the surface early
            # (thrust 131 deg off the vertical), the aft one late (thrust 0 deg). They hand over.
            cam_rate = None if a.cam_period is None else 1.0 / a.cam_period
            sensors.append(("cam+", NPER.Camera(sign=+1, name="cam+", rate=cam_rate)
                            if cam_rate else NPER.Camera(sign=+1, name="cam+")))
            sensors.append(("cam-", NPER.Camera(sign=-1, name="cam-", rate=cam_rate)
                            if cam_rate else NPER.Camera(sign=-1, name="cam-")))
        sens_next = {k: 0.0 for k, _ in sensors}
        sens_hits = {k: 0 for k, _ in sensors}
        meas = []      # every measurement actually taken, for plotting against truth
        print(f"NAV ON  seed {a.nav_seed}  IMU {a.imu_grade or CN.IMU_GRADE}"
              f"{' (perfect)' if a.perfect_imu else ''} @ {ip['rate']:g} Hz")
        print(f"  OD handover error: |dr| {np.linalg.norm(dr0):.3f} m, "
              f"|dv| {np.linalg.norm(dv0):.5f} m/s")
        print(f"  true IMU bias |b| {np.linalg.norm(imu_dev.bias0):.3e} m/s^2\n")

    nominal = None
    if a.nominal_warm and os.path.exists(a.nominal):
        _z = np.load(a.nominal, allow_pickle=True)
        nominal = dict(X=np.array(_z["X"], float), U=np.array(_z["U"], float), s=float(_z["s"]),
                       scales=C.Scales(L=float(_z["L"]), T=float(_z["T"]), A=float(_z["A"])))
        print(f"cycle-0 warm start: nominal {os.path.basename(a.nominal)}  "
              f"K {nominal['X'].shape[1]}, tf {nominal['s']*nominal['scales'].T:.1f} s")
    elif a.nominal_warm:
        print(f"cycle-0 warm start: {a.nominal} not found -- cold starting")

    t_s = 0.0; warm = None; mag = None; d = None
    # fallback bookkeeping: the last plan that converged, how far into it the vehicle already is,
    # and how much of the current cycle was actually flown
    prev_plan = None; plan_age = 0.0; flown_last = 0.0
    n_fallback = 0; fallback_cycles = []
    committed = False; n_commit = 0
    hist = []; trace = []; landed = False; t_wall0 = time.perf_counter()
    for cyc in range(a.max_cycles):
        if (time.perf_counter() - t_wall0) / 60.0 > a.max_minutes:
            print("\nstopped: wall-clock budget reached"); break
        # RESCALE FIRST, THEN BUILD EVERYTHING FROM THE NEW SCALES. Order matters and got this
        # wrong once: with `common` assembled before the rescale, the solve ran on the old scales,
        # dynamics and rf_nd while the warm start had already been converted to the new ones, and
        # the reported miss compared an old-scale trajectory against a new-scale target -- 7.6 m of
        # pure unit mismatch on a trajectory whose real miss was 14 mm.
        #
        # L is the range still to fly and T the time to cross it under full acceleration, so the
        # solver always sees O(1) variables however close the vehicle has come. Held at the handover
        # values instead, positions run at 0.007 and sigma at 0.12 by the terminal phase and CLARABEL
        # starts reporting inaccurate solutions.
        if a.rescale:
            sc_old = sc
            sc = C.Scales.build(xd[0:3], r_site, m_wet=float(np.exp(xd[6])))
            dyn = Dynamics(sc, field=plan_fld)
            dyn_t = Dynamics(sc, field=truth_fld)
            rf_nd = r_site / sc.L
            if warm is not None:
                warm = rescale_plan(*warm, sc_old, sc)
            # The FALLBACK plan has to be rescaled too, or a cycle that falls back after a rescale
            # flies a trajectory expressed in the previous cycle's units.
            if prev_plan is not None:
                prev_plan = rescale_plan(*prev_plan, sc_old, sc)
        x = np.concatenate([xd[0:3] / sc.L, xd[3:6] / sc.v, [xd[6]]])
        # WHAT THE GUIDANCE IS ALLOWED TO SEE. With --nav this is the filter's estimate; the mass
        # comes from the truth because propellant consumption is instrumented, not inferred.
        #
        # THE ESTIMATE HAS TO COME BACK OUT OF ACI. The filter works inertially because that is
        # where the accelerometer senses; the guidance works in ACAF because that is where the
        # landing site is stationary. Position is a pure rotation between them, but VELOCITY
        # carries the frame-rate term and must have w x r removed before rotating, or the plan
        # starts from a velocity that is wrong by |w||r| = 6.1 m/s at handover -- most of the
        # 8.6 m/s the vehicle actually has.
        if nav is not None:
            r_nav = NFR.vec_inertial_to_rot(nav.x[CN.IR], t_s, wz)
            v_nav = NFR.vec_inertial_to_rot(
                nav.x[CN.IV] - np.cross([0.0, 0.0, wz], nav.x[CN.IR]), t_s, wz)
            xg = np.concatenate([r_nav / sc.L, v_nav / sc.v, [xd[6]]])
        else:
            xg = x

        # THE PLAN STARTS FROM THE THRUST THE ENGINE IS ACTUALLY PRODUCING. This is the point of
        # carrying thrust as a state: without it every re-plan is free to command a thrust step,
        # and the actuator then spends seconds slewing to it while pointing the wrong way. Measured
        # open loop, the formulation without it missed by 44.6 m against 9.9 m with it.
        # On the first cycle there is nothing to match yet, so the solver chooses the handover
        # thrust and the actuator is initialised to whatever it asks for.
        a_nd = None if mag is None else (mag * d) / float(np.exp(xd[6])) / sc.A
        # The state is [r, v, z] now -- the thrust left the state when the hold became ZOH, and
        # enters as a_init, which pins the FIRST CONTROL instead of the first state.
        x_plan = np.asarray(xg, float)
        # CYCLE 0 ONLY. warm is None exactly once; from cycle 1 it carries the previous solution.
        if warm is None and nominal is not None:
            warm = nominal_warm(nominal, sc, x_plan)
        common = dict(dyn=dyn, scales=sc, x_init=x_plan, rf_nd=rf_nd, n_hat=n,
                      K=a.K, iters=a.iters, verbose=False, a_init=a_nd)
        xd_solve = xd.copy()           # the state the solve actually saw
        t_solve = t_s                  # the state was sampled HERE, before flying
        t0 = time.perf_counter()

        # COMMIT. Below a few metres there is nothing left to plan, and asking anyway is not
        # merely wasteful -- it is infeasible, structurally and at every scale we tried. The
        # terminal constraints need time the vehicle no longer has:
        #
        #     terminal pointing needs   tf >= theta_err / SLEW_MAX_DEG_S
        #     but contact arrives in    alt / |v_vert|
        #
        # and below alt ~ v_touch*theta/omega those cross. Measured at 1.58 cm (524 ms of slew
        # against 53 ms of altitude) and again at 0.70 m after the terminal cone was opened to
        # 1 deg (426 ms against 280 ms) -- opening the cone moved the failure point and did not
        # remove it, because the pointing error simply grows to fill the new cone. Relaxing t_lo,
        # K, or the scales does not touch it either; all were measured.
        #
        # So do what the flown missions do: commit. Hayabusa2 enters FREE FALL from 8.5 m and
        # OSIRIS-REx commits at Matchpoint around 54 m. Here the vehicle keeps flying the last
        # converged plan, and once that plan runs out it HOLDS its final thrust command to contact
        # -- which is stable, because the held thrust decelerates: 17 N on 594 kg against 5.6e-3
        # of gravity is 0.023 m/s^2 upward, so the touchdown speed falls rather than grows.
        alt_guid = ground_h(xg)          # the height above terrain the FILTER believes it has
        committed = bool(a.commit_alt > 0.0 and prev_plan is not None
                         and alt_guid < a.commit_alt)
        if committed:
            X, U, s = prev_plan
            plan_age += flown_last
            if n_commit == 0:
                print(f"cycle {cyc:3d}: COMMIT at {alt_guid:.2f} m (< {a.commit_alt:g} m) -- no "
                      f"more re-planning, flying the last plan to contact")
            n_commit += 1
            r1 = r2 = None
            wall = time.perf_counter() - t0
            miss_plan = float("nan")
            tgo = max(0.0, sc.t_dim(s) - plan_age)
        else:
            r1 = ptr.run(mode="error", warm=warm,
                         **{**common, "iters": a.iters1, "tol_defect_m": a.tol1_m,
                            "tol_defect_v": 0.05})
            r2 = None
            if r1.status in ("converged", "maxiter"):
                r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + 0.1 / sc.L, tol_vc_m=a.tol_vc_m,
                             tol_abs=a.tol_abs, tol_obj=a.tol_obj,
                             warm=(r1.X, r1.U, r1.s), **common)
                if r2.status not in ("converged", "maxiter"):
                    r2 = None
            wall = time.perf_counter() - t0

        # A FAILED RE-PLAN IS NOT A FAILED FLIGHT. The vehicle already has a trajectory in hand;
        # the honest response to "I could not compute a new one" is to keep flying the last good
        # one, which is what a real GNC system does. Measured: the endgame subproblem goes
        # genuinely infeasible below roughly v_touch*theta/SLEW_MAX metres of altitude, because
        # there is no longer time to slew the thrust inside TERMINAL_POINT_DEG before contact --
        # and there is nothing wrong with the PLAN at that point, only with asking for another.
        #
        # This cannot run away: the stale plan carries its own horizon, and the flying loop below
        # breaks once dt_ahead >= s. The count is reported so a mid-flight failure is visible in
        # the results rather than silently absorbed.
        if r2 is None and not committed:
            if prev_plan is None:
                print(f"cycle {cyc:3d}: STAGE 1 {r1.status} with no plan to fall back on"); break
            X, U, s = prev_plan
            plan_age += flown_last
            remaining = sc.t_dim(s) - plan_age
            # ONLY IN THE LAST CYCLE. The fallback is legitimate when the plan in hand already
            # reaches the ground -- less than one guidance cycle of it is left, so re-planning
            # would buy nothing and the endgame is where the constraint set goes degenerate
            # anyway. A failure with MORE than a cycle still to fly is a different animal: the
            # vehicle would be committing to a stale trajectory over a stretch it was supposed to
            # correct, and that must stop the run loudly rather than be absorbed.
            if remaining >= a.dt_guid:
                print(f"cycle {cyc:3d}: re-plan failed (S1 {r1.status}) with {remaining:.1f} s of "
                      f"plan left -- more than one {a.dt_guid:g} s cycle, so this is not the "
                      f"endgame. Stopping."); break
            # AND THE PLAN MUST STILL HAVE SOMETHING IN IT. A spent plan (remaining <= 0) flies
            # nothing: the loop below breaks immediately at dt_ahead >= s, flown_last stays 0, the
            # time does not advance and the cycle repeats identically until max_cycles. Measured as
            # an infinite loop at 0.59 m altitude, the same line printed over and over.
            if remaining <= 0.0:
                print(f"cycle {cyc:3d}: re-plan failed (S1 {r1.status}) and the previous plan is "
                      f"spent ({remaining:.1f} s left). Nothing left to fly. Stopping."); break
            n_fallback += 1
            fallback_cycles.append(int(cyc))
            print(f"cycle {cyc:3d}: re-plan failed (S1 {r1.status}) in the last cycle -- flying "
                  f"the previous plan, {remaining:.1f} s of it left")
            miss_plan = float("nan")
        elif r2 is not None:
            X, U, s = r2.X, r2.U, r2.s
            plan_age = 0.0
            prev_plan = (X.copy(), U.copy(), s)
            Pl_ = np.eye(3) - np.outer(n, n)
            miss_plan = float(np.linalg.norm(Pl_ @ (r2.X[0:3, -1] - rf_nd))) * sc.L
        if not committed:
            tgo = sc.t_dim(s) - plan_age
        if mag is None:
            # THE ENGINE STATE AT HANDOVER IS A TRUTH-SIDE FACT, NOT A PLANNING ONE. SCvx keeps
            # planning for a lit engine -- its thrust bounds are Tmin <= ||T|| <= Tmax at every node
            # and there is no zero-thrust state in the formulation. Allowing one would make the
            # feasible set {0} union [Tmin, Tmax], which is disjoint, and lossless convexification
            # does not cover that. So the shutdown lives here, in the delivered thrust: the actuator
            # starts at 0 N and the first-order lag brings it up to whatever the plan commands.
            # The attitude is taken as already correct -- the braking phase and coast leave time to
            # slew -- so only the magnitude starts from zero.
            T0, _ = plan_thrust(X, U, s, 0.0, sc)
            _, d = act.clip_command(T0)
            mag = 0.0 if a.engine_off else act.clip_command(T0)[0]

        n_sub = max(1, int(round(a.dt_guid / a.dt_act)))
        h_act = sc.t_nd(a.dt_act)
        alt0 = float(n @ (x[0:3] - rf_nd)) * sc.L
        flown_last = 0.0
        for i in range(n_sub):
            dt_ahead = sc.t_nd(plan_age + i * a.dt_act)
            if dt_ahead >= s and committed:
                # the plan is spent and there is no new one coming: HOLD its final command. The
                # plan ends where the FILTER thinks the ground is, and the filter was measured
                # 0.62 m low at touchdown, so a plan that has run out does not mean the vehicle
                # has arrived. Clamping to the last plan point flies that gap on the last thrust
                # the solver asked for.
                dt_ahead = s * (1.0 - 1e-9)
            elif dt_ahead >= s:
                break
            T_cmd, _ = plan_thrust(X, U, s, dt_ahead, sc)
            mag_cmd, d_cmd = act.clip_command(T_cmd)
            T_vec = mag * d
            # t_s is ALREADY the current time -- it is advanced by the propagation loop
            # below. Adding i*dt_act again double-counts it, so each cycle's samples ran to
            # +119 s of a 60 s cycle and then jumped BACKWARD 59 s at the boundary: 34
            # backward jumps across the descent, which makes every time series discontinuous.
            # The DIRECTIONS and the MASS go in as vectors, not just their angle to the local
            # vertical. An angle to one reference axis is 2 of the 3 rotational degrees of freedom:
            # the azimuth about that axis is missing, so the 3D thrust direction cannot be
            # recovered from ang_act afterwards, and anything that needs the real boresight (the
            # IMU bias frame, a lidar ray, a camera line of sight) cannot be post-processed from a
            # run that stored only the angle. r goes in for the same reason.
            _tr = dict(t=t_s, T_cmd=float(mag_cmd), T_act=float(mag),
                       ang_cmd=float(np.degrees(np.arccos(np.clip(d_cmd @ n, -1, 1)))),
                       ang_act=float(np.degrees(np.arccos(np.clip(d @ n, -1, 1)))),
                       alt=ground_h(x),
                       d_act=[float(q) for q in d], d_cmd=[float(q) for q in d_cmd],
                       mass=float(np.exp(x[6])),
                       r=[float(q) * sc.L for q in x[0:3]],
                       v=[float(q) * sc.v for q in x[3:6]])
            if nav is not None:
                # truth and estimate as full vectors, so the plot can show either the states
                # themselves or their difference without re-deriving anything
                # LIVE truth, not xd. xd is refreshed once per cycle, so using it here recorded a
                # STAIRCASE against a smoothly-moving estimate and turned the error plot into a
                # sawtooth at the guidance-cycle period -- an artefact of the recording, not of the
                # navigation. x is the current non-dimensional truth inside the sub-step loop.
                _tr.update(sens=[int(sens_hits[k]) for k, _ in sensors],
                           r_true=[float(v) for v in x[0:3] * sc.L],
                           v_true=[float(v) for v in x[3:6] * sc.v],
                           # the estimate is rotated back to ACAF so it lines up with r_true /
                           # v_true above; the bias stays in ACI, which is where IMU.bias lives
                           r_nav=[float(v) for v in NFR.vec_inertial_to_rot(
                               nav.x[CN.IR], t_s, wz)],
                           v_nav=[float(v) for v in NFR.vec_inertial_to_rot(
                               nav.x[CN.IV] - np.cross([0.0, 0.0, wz], nav.x[CN.IR]), t_s, wz)],
                           sig_p=nav.sigma_pos(), sig_v=nav.sigma_vel(),
                           b_nav=[float(v) for v in nav.bias_inertial],
                           a_dmc=[float(v) for v in nav.dmc_inertial],
                           b_true=[float(v) for v in imu_dev.bias])
            trace.append(_tr)
            # THE FILTER RUNS ON THE IMU, IN PARALLEL WITH THE TRUTH. The thrust is constant over
            # an actuator sub-step IN ACAF, so in ACI that same physical thrust ROTATES at w --
            # w*dt_act = 1.7e-4 rad per sub-step. Re-rotating at every IMU sample is what keeps
            # the propagation first-order accurate; freezing a_I across the sub-step costs 1.3e-2 m
            # over 200 s no matter how finely the filter integrates. Measured; see test_ekf.py.
            #
            # Nothing is rotated back. The accelerometer senses in ACI and the filter works there.
            if nav is not None:
                dti = 1.0 / nav.p["rate"]
                n_imu = max(1, int(round(a.dt_act / dti)))
                a_true_rot = (mag * d) / float(np.exp(x[6]))        # true specific force [m/s^2]
                for j in range(n_imu):
                    tj = t_s + j * dti
                    a_I = NFR.vec_rot_to_inertial(a_true_rot, tj, wz)
                    d_I = NFR.vec_rot_to_inertial(d, tj, wz)
                    z_I = imu_dev.measure(a_I, dti, d_I=d_I)
                    nav.propagate(z_I, dti, tj, d_I=d_I)
                    # MEASUREMENT UPDATES, each at its own rate and only while its activation
                    # condition holds. Activation is evaluated on the TRUE state -- whether a
                    # sensor can see the surface is a fact about the world, not about the estimate.
                    r_true_now = x[0:3] * sc.L
                    for key, dev in sensors:
                        if tj < sens_next[key] or not dev.active(r_true_now, n, r_site, d):
                            continue
                        sens_next[key] = tj + 1.0 / dev.rate
                        # thrust_dir sets the boresight, t sets the frame the block delivers in:
                        # the camera rotates its fix and its covariance into ACI, the lidar rotates
                        # only its Jacobian because a range is a scalar and therefore frame-free.
                        zk, Hk, Rk = dev.measure(r_true_now, n, r_site, rng,
                                                 thrust_dir=d, t=tj)
                        hp = dev.predict(nav.x, n, r_site, thrust_dir=d, t=tj)
                        if not np.all(np.isfinite(hp)):
                            continue        # the on-board shape model sees nothing along that ray
                        # record BEFORE the update, so the residual is the pre-fit innovation --
                        # the quantity that says whether the filter and the sensor agree, rather
                        # than the post-fit residual which is shrunk by the update itself
                        meas.append(dict(t=tj, s=key, z=[float(v) for v in np.atleast_1d(zk)],
                                         h=[float(v) for v in np.atleast_1d(hp)],
                                         sig=float(np.sqrt(Rk[0, 0]))))
                        nav.update(zk, Hk, Rk, hp)
                        sens_hits[key] += 1
            nst = max(1, int(round(a.dt_act / a.dt_prop)))
            hp = h_act / nst
            for _ in range(nst):
                xn = rk4(dyn_t, x, T_vec, hp, sc)
                if ground_h(xn) <= 0.0:                          # bisect the surface crossing
                    lo, hi = 0.0, hp
                    for _ in range(50):
                        mid = 0.5 * (lo + hi)
                        xm = rk4(dyn_t, x, T_vec, mid, sc)
                        if ground_h(xm) <= 0.0:
                            hi, xn = mid, xm
                        else:
                            lo = mid
                    x = xn; landed = True; t_s += sc.t_dim(hi); break
                x = xn; t_s += sc.t_dim(hp)
            if landed:
                break
            flown_last += a.dt_act
            mag, d, _ = act.step(mag, d, T_cmd, a.dt_act)
        xd = np.concatenate([x[0:3] * sc.L, x[3:6] * sc.v, [x[6]]])
        alt = ground_h(x)
        spd = float(np.linalg.norm(xd[3:6]))
        if nav is not None:
            r_nav_e = NFR.vec_inertial_to_rot(nav.x[CN.IR], t_s, wz)
            v_nav_e = NFR.vec_inertial_to_rot(
                nav.x[CN.IV] - np.cross([0.0, 0.0, wz], nav.x[CN.IR]), t_s, wz)
            nav_pe = float(np.linalg.norm(r_nav_e - xd[0:3]))
            nav_ve = float(np.linalg.norm(v_nav_e - xd[3:6]))
            # NOT `act` -- that name is the Actuator, and rebinding it here replaced the
            # object with a string on the next cycle.
            sens_str = " ".join(f"{k}:{sens_hits[k]}" for k, _ in sensors) if sensors else "none"
            print(f"        NAV | pos err {nav_pe:8.3f} m  vel err {nav_ve:9.5f} m/s  "
                  f"sig_p {nav.sigma_pos():8.2f} m  sig_v {nav.sigma_vel():8.4f} m/s  "
                  f"bias err {float(np.linalg.norm(nav.bias_inertial - imu_dev.bias)):.3e}  "
                  f"[{sens_str}]"
                  # a gate that never fires and a gate that fires constantly are both bugs, so the
                  # count belongs next to the error it is there to prevent
                  + (f"  rej {nav.n_rej}" if nav.n_rej else ""))
        s1_txt = (f"| S1 {r1.iters:>2}it {r1.status:<11} D {r1.miss_m:8.4f} "
                  f"def {r1.defect_m:8.2e} " if r1 is not None
                  else f"| COMMITTED, no re-plan                            ")
        s2_txt = (f"| S2 {r2.iters:>2}it {r2.status:<11} miss {miss_plan:8.4f} "
                  f"def {r2.defect_m:8.2e} " if r2 is not None
                  else f"| plan age {plan_age:6.1f} s                       ")
        print(f"cyc {cyc:3d} | t {t_s:7.1f} alt {alt:9.2f} |v| {spd:7.4f} m {float(np.exp(x[6])):7.3f} "
              + s1_txt
              + s2_txt
              + f"| tgo {tgo:7.1f} T {mag_cmd:5.2f}/{mag:5.2f} | L {sc.L:8.1f} T {sc.T:6.1f} "
                f"s_nd {tgo/sc.T:5.2f} {wall:6.1f}s", flush=True)
        # t_solve and t_go belong together; t is AFTER the cycle was flown. Pairing the
        # two made the predicted landing time read one whole cycle too late.
        hist.append(dict(cyc=cyc, t_solve=t_solve, t=t_s, alt=alt, speed=spd,
                         mass=float(np.exp(xd[6])), t_go=tgo, wall=wall,
                         it1=(None if r1 is None else r1.iters),
                         it2=(None if r2 is None else r2.iters),
                         status=("committed" if committed else
                                 "reused_plan" if r2 is None else r2.status),
                         status1=(None if r1 is None else r1.status),
                         D_m=(None if r1 is None else r1.miss_m),
                         defect1_m=(None if r1 is None else r1.defect_m),
                         defect2_m=(None if r2 is None else r2.defect_m),
                         miss_plan_m=miss_plan, plan_age_s=plan_age,
                         T_cmd=float(mag_cmd), T_act=float(mag),
                         # ACI -> ACAF before differencing. The printed NAV line already does
                         # this; these recorded copies did not, so they were reporting the angle
                         # between the frames (269 m growing to 3420 m as Eros turned 39 deg)
                         # rather than the navigation error, which is ~0.5 m.
                         nav_pos_err=(None if nav is None else nav_pe),
                         nav_vel_err=(None if nav is None else nav_ve),
                         nav_sig_pos=(None if nav is None else nav.sigma_pos()),
                         # Enough to replay any cycle in isolation and to see WHICH convergence
                         # criterion was still failing, without re-flying the whole descent.
                         xd=[float(v) for v in xd_solve],
                         a_nd=(None if a_nd is None else [float(v) for v in a_nd]),
                         scales=[sc.L, sc.T, sc.A],
                         hist1=(None if r1 is None else r1.history),
                         hist2=(None if r2 is None else r2.history)))
        if landed:
            print(f"\nTOUCHDOWN at t = {t_s:.2f} s"); break
        frac = min(sc.t_nd(a.dt_guid) / s, 0.95)
        warm = shift_plan(X, U, s, frac)

    xd = np.concatenate([x[0:3] * sc.L, x[3:6] * sc.v, [x[6]]])
    # THE MISS IS MEASURED WHERE IT ACTUALLY LANDED, on the ground, in three dimensions.
    # It used to be |Pl (r - r_site)|, the separation projected into the plane perpendicular to the
    # local vertical. But the vehicle touches down on the FACET, which is 21.3 deg away from that
    # plane, so both points lie in the facet and their true separation is simply |r - r_site|. The
    # projected version understated it by 1/cos of whatever part of the tilt the miss direction ran
    # along -- up to 7%. miss_plane is kept because the guidance's own terminal constraint and its
    # reported miss_plan_m are both in the datum plane, and the two numbers should be comparable.
    miss = float(np.linalg.norm(xd[0:3] - r_site))
    miss_plane = float(np.linalg.norm(Pl @ (xd[0:3] - r_site)))
    alt = ground_h(x)
    spd = float(np.linalg.norm(xd[3:6]))
    vert = float(n @ xd[3:6])
    ang = float(np.degrees(np.arccos(np.clip(d @ n, -1, 1))))
    print(f"\n{'='*66}\nWHERE IT LANDS\n{'='*66}")
    print(f"  landed            : {landed}")
    print(f"  time of flight    : {t_s:.2f} s")
    print(f"  miss from site    : {miss:.4f} m   (on the surface; "
          f"{miss_plane:.4f} m projected into the datum plane)")
    print(f"  altitude          : {alt:.5f} m")
    print(f"  touchdown speed   : {spd:.4f} m/s   (vertical {vert:+.4f}, limit {C.VTOUCH})")
    print(f"  thrust angle      : {ang:.2f} deg from the local vertical")
    print(f"  propellant        : {m0 - float(np.exp(xd[6])):.4f} kg")
    print(f"  cycles            : {len(hist)}, mean solve {np.mean([h['wall'] for h in hist]):.1f} s")
    # A re-plan that failed is not hidden: if the vehicle flew stale plans, that is part of the
    # result and belongs next to the miss, not buried in the log.
    print(f"  committed cycles  : {n_commit}"
          + (f"  (below {a.commit_alt:g} m, flew the last plan to contact)" if n_commit else ""))
    print(f"  re-plan failures  : {n_fallback}"
          + (f"  -> flew the previous plan on cycles {fallback_cycles}" if n_fallback else ""))
    os.makedirs(a.out, exist_ok=True)
    # Touchdown is the single most important sample in the trace (it carries the landing point and
    # the final navigation error), and it is the one decimation would drop, so put it back.
    trace_out = trace[::max(a.trace_stride, 1)]
    if trace and trace_out[-1] is not trace[-1]:
        trace_out = trace_out + [trace[-1]]
    json.dump(dict(landed=bool(landed), t=t_s, miss=miss, miss_plane=miss_plane,
                   alt=alt, speed=spd, vert=vert,
                   tol_vc_m=a.tol_vc_m, tol_abs=a.tol_abs, tol_obj=a.tol_obj,
                   plan_gravity=a.plan_gravity, nav_gravity=a.nav_gravity,
                   nav=bool(a.nav), nav_seed=a.nav_seed,
                   nav_rej=(int(nav.n_rej) if nav is not None else 0),
                   dmc_sigma=(None if nav is None else nav.dmc_sigma),
                   gate_nis=(None if nav is None else nav.gate_nis),
                   sensor_names=[k for k, _ in sensors] if a.nav else [],
                   meas=(meas if a.nav else []),
                   seed=a.seed, pos_sigma=a.pos_sigma, vel_sigma=a.vel_sigma,
                   from_nominal=bool(a.from_nominal),
                   ang=ang, fuel=m0 - float(np.exp(xd[6])), K=a.K, dt_guid=a.dt_guid,
                   tau=a.tau, slew=a.slew, commit_alt=a.commit_alt, n_commit=int(n_commit),
                   n_fallback=int(n_fallback),
                   fallback_cycles=fallback_cycles,
                   term_point_deg=C.TERMINAL_POINT_DEG,
                   cycles=hist, trace=trace_out),
              open(os.path.join(a.out, f"closed_loop{a.tag}.json"), "w"), indent=1)
    print(f"\nwrote {a.out}/closed_loop{a.tag}.json")


if __name__ == "__main__":
    main()
