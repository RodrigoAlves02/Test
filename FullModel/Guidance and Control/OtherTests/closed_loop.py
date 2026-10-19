#!/usr/bin/env python
"""Closed loop: SCvx -> actuator -> truth dynamics, repeated until touchdown.

    x (true)  ->  SCvx  ->  T_cmd  ->  actuator  ->  T_applied  ->  truth dynamics  ->  x
                                       (lag + slew)

Navigation is not in this loop yet, so guidance is fed the TRUE state. What is being tested is
whether the guidance and the actuator can fly the vehicle to the ground at all once the commanded
thrust is no longer delivered instantly or turned instantly -- the two limits the actuator imposes.

THE ACTUATOR IS PART OF THE PLANT, NOT A POST-PROCESSING STEP. The applied thrust changes
continuously across a guidance interval, so it is carried as a state and integrated with everything
else:

    y = [ r(3), v(3), m, T(3) ]     ten states, inertial frame

    rdot = v
    vdot = R(wt) g(R(-wt) r) + T/m
    mdot = -|T| / (Isp g0)
    Tdot = d/dt ( |T| d )  with |T| relaxing to |T_cmd| at rate 1/tau
                            and d rotating toward d_cmd at at most slew_max

Treating the actuator as a separate discrete update would hold the thrust constant across an
interval it is demonstrably not constant over.

Integrated with DOP853 at rtol 1e-12, one call per guidance interval, with the command held. The
integrator is not told about the actuator's own kink at the moment the slew finishes; at this
tolerance it simply shortens its step through it.

Touchdown is a terminal event on altitude above the site plane, so the landing epoch is found by
root-finding rather than by overshooting into the surface and interpolating afterwards.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TBB_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
GC = os.path.join(HERE, "..")
SCVX = os.path.join(HERE, "..", "..", "..", "scvx_landing")
sys.path.insert(0, GC)
sys.path.insert(0, SCVX)

import guidance as GD                                                # noqa: E402
from actuator_model import Actuator                                  # noqa: E402

WZ = GD.WZ
OMEGA = GD.OMEGA


def _rk8_tableau():
    """The 12 propagating stages of DOP853, the same tableau the truth-dynamics validation selected
    (FullModel/Truth Dynamics/Validation/truth_dynamics.py), taken from SciPy rather than retyped."""
    from scipy.integrate._ivp import dop853_coefficients as d
    n = d.N_STAGES
    return (np.array(d.A[:n, :n], float), np.array(d.B[:n], float), np.array(d.C[:n], float))


def rk8_fixed(f, t0, t1, y0, dt, args=()):
    """Fixed-step RK8 over [t0, t1], trimming only the last step.

    A fixed-step integrator cannot collapse its step, so its cost per arc is known in advance and it
    is structurally immune to the step-size blow-up an adaptive method suffers on a non-smooth
    right-hand side. That bounded cost is what makes it the right choice inside a Monte Carlo.
    """
    A, B, C = _RK8
    n = max(1, int(np.ceil((t1 - t0) / dt)))
    h_last = (t1 - t0) - (n - 1) * dt
    y = np.asarray(y0, float).copy()
    t = t0
    for i in range(n):
        h = dt if i < n - 1 else h_last
        if h <= 0.0:
            break
        k = []
        for j in range(len(B)):
            yj = y + h * sum(A[j, m] * k[m] for m in range(j)) if j else y
            k.append(np.asarray(f(t + C[j] * h, yj, *args), float))
        y = y + h * sum(B[j] * k[j] for j in range(len(B)))
        t += h
    return t, y


_RK8 = _rk8_tableau()


def plan_thrust(plan, dt_ahead, t_abs):
    """The thrust the plan calls for dt_ahead seconds after it was made, in the INERTIAL frame.

    MPC conventionally applies u(tau = 0) and holds it for the whole cycle. That is wrong here, and
    expensively so. Minimum-fuel control is bang-bang, so the optimal solution has a SWITCH, and the
    switch generally falls in the interior of a cycle -- holding node 0 defers it by up to one full
    cycle. The minimum-fuel trajectory has no margin to absorb that: it brakes as late as physically
    possible, 7.4 m of altitude margin at 244 m in the nominal, while one 10 s cycle at 4.4 m/s eats
    44 m. Measured, the vehicle was still coasting at Tmin one cycle after the landing had already
    become unrecoverable, and touchdown speed scaled straight with the cycle length: 1.47 m/s at
    30 s, 0.84 at 10 s, 0.52 at 5 s, against a VTOUCH of 0.30.

    The plan already carries the switch at its own resolution -- K = 80 nodes over a ~60 s terminal
    horizon is about 1 s -- so read it there instead of throwing it away. This is also what the
    optimiser assumed: U is a zero-order hold on ITS node grid, not on the guidance grid.
    """
    K = plan.X.shape[1]
    h = plan.sigma / (K - 1)
    k = int(np.clip(np.floor(dt_ahead / h), 0, K - 2))
    T_B = float(np.exp(plan.X[6, k])) * plan.U[0:3, k]
    return GD.body_to_inertial_vec(T_B, t_abs)


def rhs(t, y, T_vec, g_body, alpha):
    """Seven-state inertial translational dynamics with the thrust held CONSTANT.

    The actuator is deliberately NOT a state here. Carrying it as one requires writing the rate
    limiter as a differential equation, and a rate limiter is not a differential equation -- it is a
    switch. Written that way the thrust direction turns at the full slew rate until it reaches the
    command and then stops, which as an ODE means it overshoots and reverses: a sliding mode. Handing
    that to an adaptive integrator at rtol 1e-12 collapses the step size. Measured on a single 30 s
    closed-loop arc: 518 639 gravity calls and 225 s of wall clock, against 824 calls and 0.3 s for
    the arc before it, with the traceback inside solve_ivp rather than inside the solver. Softening
    the switch into a boundary layer helped but did not remove it, because ang = arccos(d . d_cmd) is
    not Lipschitz where it matters.

    The actuator is a zero-order-hold device and Actuator.step already integrates it EXACTLY in
    closed form. So hold the thrust over a short sub-step, advance the actuator analytically at the
    sub-step boundaries, and let the integrator see only the smooth translational dynamics. That is
    both faster and a more faithful model of the hardware.
    """
    r_I, v_I, m = y[0:3], y[3:6], y[6]
    c, s_ = np.cos(WZ * t), np.sin(WZ * t)
    r_B = np.array([c * r_I[0] + s_ * r_I[1], -s_ * r_I[0] + c * r_I[1], r_I[2]])
    gB = g_body(r_B)
    a = np.array([c * gB[0] - s_ * gB[1], s_ * gB[0] + c * gB[1], gB[2]])
    Tn = float(np.linalg.norm(T_vec))
    if Tn > 0.0:
        a = a + T_vec / m
    return np.concatenate([v_I, a, [-alpha * Tn]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-frac", type=float, default=0.75,
                    help="start this far along the cached solution; 0 = handover, 0.75 = final "
                         "quarter. A full descent is ~77 guidance cycles at ~22 s of SCvx each, "
                         "which is an hour of wall clock to answer a question the last quarter "
                         "answers just as well.")
    ap.add_argument("--dt-guid", type=float, default=30.0, help="guidance cycle [s]")
    ap.add_argument("--tau", type=float, default=0.5, help="thrust response constant [s]")
    ap.add_argument("--dt-act", type=float, default=0.5,
                    help="actuator sub-step within a guidance cycle [s]")
    ap.add_argument("--hold-first", action="store_true",
                    help="hold u(tau=0) for the whole cycle (conventional MPC; defers the "
                         "bang-bang switch by up to one cycle)")
    ap.add_argument("--integrator", choices=("dop853", "rk8"), default="rk8",
                    help="adaptive DOP853, or fixed-step RK8 (bounded cost per arc)")
    # one RK8 step per actuator sub-step: 720 gravity calls per guidance cycle against DOP853's
    # 2820, at the same round-off-limited accuracy. See solver_fix/solver_fix.txt section 7.
    ap.add_argument("--dt-prop", type=float, default=0.5,
                    help="fixed step for --integrator rk8 [s]")
    ap.add_argument("--slew", type=float, default=2.0, help="slew limit [deg/s]")
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--atol", type=float, default=1e-14)
    ap.add_argument("--max-cycles", type=int, default=200)
    # Cap the solver, not just the loop. The wall-clock budget is only checked BETWEEN cycles, so a
    # single runaway solve escapes it entirely -- measured at one cycle burning 40+ minutes against
    # 1.3-7.0 s for the ones before it, with the 120-iteration default. The tuned configuration
    # needs 7-25 iterations on every problem in the selection battery, so a cycle that wants more
    # than 40 has failed and should be reported as a failure rather than waited out.
    ap.add_argument("--iters", type=int, default=40, help="max SCvx iterations per cycle")
    ap.add_argument("--watchdog", type=float, default=0.0,
                    help="dump a traceback every N seconds (diagnostic)")
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--no-warm", dest="warm", action="store_false", default=True,
                    help="disable receding-horizon warm starting")
    ap.add_argument("--cold", action="store_true",
                    help="do not seed the first cycle from the nominal")
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.watchdog > 0:
        # Dump where the process actually is if a cycle overruns, instead of inferring it.
        faulthandler.dump_traceback_later(a.watchdog, repeat=True, exit=False)

    import tune
    z = np.load(os.path.join(SCVX, "descent_nominal.npz"))
    rf = np.asarray(z["rf"], float)
    X_old, sig_old = z["X"], float(z["sigma"])
    K_old = X_old.shape[1]
    k0 = int(round(a.start_frac * (K_old - 1)))
    x0 = X_old[:, k0].copy()                 # rotating frame, taken from the old solution
    t_old = k0 / (K_old - 1) * sig_old
    print(f"starting {a.start_frac:.0%} along the cached solution: node {k0}/{K_old-1}, "
          f"t = {t_old:.1f} s of {sig_old:.1f} s, {sig_old - t_old:.1f} s of flight remaining")

    # Seed from the remainder of the nominal, t_go included. The loop starts part-way along, so the
    # useful guess is the tail of the cached solution from node k0 with the elapsed time removed --
    # not the whole thing. Cold-started, the first cycle exhausts its iteration budget and the
    # planned mission duration then wanders by ~380 s over six cycles (solver_fix/step22_persist).
    nom = (X_old[:, k0:], np.asarray(z["U"], float)[:, k0:], sig_old - t_old)
    gd = GD.SCvxGuidance(warm_start=a.warm, nominal=None if a.cold else nom,
                         iters=a.iters)
    ctx = gd.ctx
    n_hat = np.asarray(ctx.n_hat, float)
    g_body = ctx.truth.g
    alpha = 1.0 / (tune.ISP * 9.80665)
    act = Actuator(tau=a.tau, slew_max_deg_s=a.slew, t_min=tune.TMIN, t_max=tune.TMAX)
    # count gravity calls so the two phases can be priced separately rather than inferred
    _gcount = [0]
    _graw = g_body
    def g_body(r, _c=_gcount, _f=_graw):
        _c[0] += 1
        return _f(r)

    # expressed inertially at t = 0, where the two frames coincide
    r_I = x0[0:3].copy()
    v_I = x0[3:6] + np.cross(OMEGA, x0[0:3])
    m = float(np.exp(x0[6]))
    y = np.concatenate([r_I, v_I, [m]])                   # 7 states; the actuator is not one of them
    # THE ENGINE IS ALREADY LIT. This run picks up an existing descent three quarters of the way
    # down, so the actuator state at t = 0 is whatever the nominal was delivering at node k0 -- not
    # a cold engine pointing at the local vertical. Starting it off injects a transient that belongs
    # to no trajectory: a 15 N thrust step, and a 139 deg pointing error that the 2 deg/s rate limit
    # then needs ~70 s to slew out, on an arc only ~430 s long. The vehicle spends its first two
    # guidance cycles recovering from an initial condition that was never real, and every metric
    # downstream -- miss, touchdown speed, propellant -- inherits it.
    #
    # U[0:3] is an acceleration (the optimiser carries thrust over mass), so multiply by the mass at
    # that node. At t = 0 the rotating and inertial frames coincide, which is the same convention
    # r_I and v_I above are built on, so this direction needs no rotation.
    T0 = float(np.exp(x0[6])) * np.asarray(z["U"], float)[0:3, k0]
    mag_act, d_act = act.clip_command(T0)
    t = 0.0

    def altitude(t, y, *args):
        # solve_ivp hands `args` to the events as well as to the right-hand side, so this has to
        # accept them even though it uses none of them.
        c, s = np.cos(WZ * t), np.sin(WZ * t)
        r_B = np.array([c * y[0] + s * y[1], -s * y[0] + c * y[1], y[2]])
        return float(n_hat @ (r_B - rf))
    altitude.terminal = True
    altitude.direction = -1

    print(f"start: altitude {altitude(0.0, y)/1e3:.3f} km, |v| {np.linalg.norm(v_I):.4f} m/s, "
          f"m {m:.2f} kg")
    print(f"       engine already lit: |T| {mag_act:.2f} N at "
          f"{np.degrees(np.arccos(np.clip(d_act @ n_hat, -1, 1))):.1f} deg from local vertical")
    print(f"guidance cycle {a.dt_guid:g} s, tau {a.tau:g} s, slew {a.slew:g} deg/s, "
          f"{a.integrator}" + (f" dt {a.dt_prop:g} s" if a.integrator == "rk8"
                            else f" rtol {a.rtol:.0e}") + "\n")

    T_t, T_cmd_mag, T_act_mag = [], [], []
    ang_cmd, ang_act, ang_err = [], [], []
    cyc = []
    t_wall0 = time.perf_counter()
    landed = False

    for k in range(a.max_cycles):
        if (time.perf_counter() - t_wall0) / 60.0 > a.max_minutes:
            print(f"\nstopped: wall-clock budget of {a.max_minutes:g} min reached")
            break
        if a.watchdog > 0:      # diagnostic only: lets a hung cycle be replayed in isolation
            np.savez(os.path.join(a.out, "cycle_state.npz"), k=k, t=t, y=y)
        T_cmd, plan = gd.command(t, y[0:3], y[3:6], y[6])
        if T_cmd is None:
            print(f"cycle {k:3d}: GUIDANCE FAILED -> {plan.error}")
            break
        mag_cmd, d_cmd = act.clip_command(T_cmd)

        _tp = time.perf_counter()
        _n0 = _gcount[0]
        # SUB-STEPPED ACTUATOR, SMOOTH ARCS BETWEEN. Over each sub-step the thrust is a constant
        # vector, so the integrator sees no switch; the actuator is advanced between sub-steps by
        # Actuator.step, which is the exact zero-order-hold solution and not an approximation.
        t_end = t + a.dt_guid
        n_sub = max(1, int(round(a.dt_guid / a.dt_act)))
        h = a.dt_guid / n_sub
        landed_here = False
        t_plan0 = t
        for _i in range(n_sub):
            # follow the plan across the cycle rather than holding its first node
            if not a.hold_first:
                T_cmd = plan_thrust(plan, t - t_plan0, t)
                mag_cmd, d_cmd = act.clip_command(T_cmd)
            T_vec = mag_act * d_act
            t_to = min(t + h, t_end)
            if a.integrator == "rk8":
                t_new, y_new = rk8_fixed(rhs, t, t_to, y, a.dt_prop, args=(T_vec, g_body, alpha))
                hit = altitude(t_new, y_new) <= 0.0
                if hit:
                    # REFINE THE CROSSING. Stopping at the sub-step boundary reports touchdown
                    # wherever the grid happens to fall -- measured at -0.43 m below the surface at
                    # 2.3 m/s, which then contaminates the reported miss and touchdown speed. A
                    # fixed-step method has no event root-finder, so bisect the arc explicitly:
                    # 40 halvings take the crossing time to ~1e-12 of the sub-step.
                    lo, hi = t, t_new
                    for _ in range(40):
                        mid = 0.5 * (lo + hi)
                        _tm, y_m = rk8_fixed(rhs, t, mid, y, a.dt_prop, args=(T_vec, g_body, alpha))
                        if altitude(mid, y_m) <= 0.0:
                            hi, t_new, y_new = mid, mid, y_m
                        else:
                            lo = mid
                        if hi - lo < 1e-9:
                            break
                ss = type("R", (), dict(t=np.array([t_new]), y=y_new[:, None], success=True,
                                        t_events=([np.array([t_new])] if hit else [np.array([])])))
            else:
                ss = solve_ivp(rhs, (t, t_to), y, method="DOP853",
                               rtol=a.rtol, atol=a.atol, args=(T_vec, g_body, alpha),
                               events=altitude, dense_output=True)
                if not ss.success:
                    raise RuntimeError(ss.message)
            T_t.append(t); T_cmd_mag.append(mag_cmd); T_act_mag.append(mag_act)
            ang_cmd.append(np.degrees(np.arccos(np.clip(d_cmd @ n_hat, -1, 1))))
            ang_act.append(np.degrees(np.arccos(np.clip(d_act @ n_hat, -1, 1))))
            ang_err.append(np.degrees(np.arccos(np.clip(d_act @ d_cmd, -1, 1))))
            y = ss.y[:, -1]; t = ss.t[-1]
            if ss.t_events is not None and len(ss.t_events[0]):
                landed_here = True
                break
            mag_act, d_act, _rate = act.step(mag_act, d_act, T_cmd, h)
        _prop_s = time.perf_counter() - _tp
        _prop_n = _gcount[0] - _n0
        s = ss
        y = s.y[:, -1]
        t = s.t[-1]
        alt = altitude(t, y)
        spd = float(np.linalg.norm(y[3:6] - np.cross(OMEGA, y[0:3])))   # speed in the rotating frame
        cyc.append(dict(k=k, t=t, alt=alt, speed=spd, mass=float(y[6]),
                        T_cmd=float(mag_cmd), T_act=float(mag_act),
                        t_go=float(plan.sigma), solve_s=plan.solve_s, prop_s=_prop_s,
                        iters=int(plan.iters), reason=str(plan.reason)))
        print(f"           solve {plan.solve_s:6.1f}s ({plan.iters} it, {plan.reason}"
              f"/{plan.exit_via})   propagate {_prop_s:6.1f}s ({_prop_n} gravity calls)")
        print(f"cycle {k:3d}: t {t:7.1f}s  alt {alt:9.2f} m  |v| {spd:7.4f} m/s  "
              f"|T| cmd {mag_cmd:5.2f} act {mag_act:5.2f} N  "
              f"t_go {plan.sigma:7.1f}s  solve {plan.solve_s:5.1f}s")

        if landed_here:
            landed = True
            print(f"\nTOUCHDOWN at t = {t:.2f} s")
            break

    c, sn = np.cos(WZ * t), np.sin(WZ * t)
    r_B = np.array([c * y[0] + sn * y[1], -sn * y[0] + c * y[1], y[2]])
    v_B = np.array([c * y[3] + sn * y[4], -sn * y[3] + c * y[4], y[5]]) - np.cross(OMEGA, r_B)
    miss = float(np.linalg.norm(r_B - rf))
    print(f"  landed        : {landed}")
    print(f"  altitude      : {altitude(t, y):.4f} m")
    print(f"  miss from site: {miss:.3f} m")
    print(f"  touchdown speed: {np.linalg.norm(v_B):.4f} m/s")
    print(f"  propellant     : {m - float(y[6]):.4f} kg")
    print(f"  cycles {len(cyc)}, mean solve {gd.mean_solve_s:.1f} s, "
          f"wall {(time.perf_counter()-t_wall0)/60:.1f} min")

    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, "closed_loop.json")
    json.dump(dict(dt_guid=a.dt_guid, tau=a.tau, slew=a.slew, rtol=a.rtol,
                   landed=bool(landed), t_land=float(t), miss=miss,
                   touchdown_speed=float(np.linalg.norm(v_B)),
                   propellant=float(m - y[6]), cycles=cyc,
                   trace=dict(t=T_t, T_cmd=T_cmd_mag, T_act=T_act_mag,
                              ang_cmd=ang_cmd, ang_act=ang_act, ang_err=ang_err)),
              open(f, "w"), indent=1)
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
