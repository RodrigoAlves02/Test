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
import problem_setup as PS
from actuator_model import Actuator                                      # noqa: E402


def plan_thrust(X, U, s, dt_ahead, scales):
    """Thrust the plan calls for, dt_ahead (non-dimensional) after it was made. Newtons."""
    K = X.shape[1]
    h = s / (K - 1)
    k = int(np.clip(np.floor(dt_ahead / h), 0, K - 2))
    m_k = float(np.exp(X[6, k]))
    return m_k * U[0:3, k] * scales.A, k


def shift_plan(X, U, s, frac):
    """The remaining plan, resampled onto the same node count -- the next cycle's warm start."""
    K = X.shape[1]
    tau = np.linspace(0.0, 1.0, K)
    tq = np.clip(frac + np.linspace(0.0, 1.0, K) * (1.0 - frac), 0.0, 1.0)
    Xs = np.array([np.interp(tq, tau, X[i]) for i in range(X.shape[0])])
    Us = np.array([np.interp(tq, tau, U[i]) for i in range(U.shape[0])])
    return Xs, Us, s * (1.0 - frac)


def rhs(dyn, x, T_vec_N, scales):
    """Truth right-hand side with a CONSTANT thrust vector in newtons (non-dimensional time)."""
    m = float(np.exp(x[6]))
    a = T_vec_N / m / scales.A
    return dyn.f(x, np.concatenate([a, [float(np.linalg.norm(a))]]))


def rk4(dyn, x, T_vec_N, h, scales):
    k1 = rhs(dyn, x, T_vec_N, scales)
    k2 = rhs(dyn, x + 0.5 * h * k1, T_vec_N, scales)
    k3 = rhs(dyn, x + 0.5 * h * k2, T_vec_N, scales)
    k4 = rhs(dyn, x + h * k3, T_vec_N, scales)
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


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
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()

    P = PS.build(); dyn, sc, n = P["dyn"], P["scales"], P["n_hat"]
    rf_nd = P["rf_nd"]; Pl = np.eye(3) - np.outer(n, n)
    act = Actuator(tau=a.tau, slew_max_deg_s=a.slew, t_min=C.TMIN, t_max=C.TMAX)
    x = P["x_init"].copy()
    m0 = float(np.exp(x[6]))

    print(f"handover: altitude {float(n @ (x[0:3]-rf_nd))*sc.L/1e3:.3f} km, "
          f"|v| {np.linalg.norm(x[3:6])*sc.v:.4f} m/s, m {m0:.2f} kg")
    print(f"K {a.K}, guidance cycle {a.dt_guid:g} s, actuator sub-step {a.dt_act:g} s, "
          f"tau {a.tau:g} s, slew {a.slew:g} deg/s\n")

    t_s = 0.0; warm = None; mag = None; d = None
    hist = []; trace = []; landed = False; t_wall0 = time.perf_counter()
    for cyc in range(a.max_cycles):
        if (time.perf_counter() - t_wall0) / 60.0 > a.max_minutes:
            print("\nstopped: wall-clock budget reached"); break
        common = dict(dyn=dyn, scales=sc, x_init=x, rf_nd=rf_nd, n_hat=n,
                      K=a.K, iters=a.iters, verbose=False)
        t_solve = t_s                  # the state was sampled HERE, before flying
        t0 = time.perf_counter()
        r1 = ptr.run(mode="error", warm=warm,
                     **{**common, "iters": a.iters1, "tol_defect_m": a.tol1_m,
                        "tol_defect_v": 0.05})
        if r1.status not in ("converged", "maxiter"):
            print(f"cycle {cyc:3d}: STAGE 1 {r1.status}"); break
        r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + 0.1 / sc.L,
                     warm=(r1.X, r1.U, r1.s), **common)
        wall = time.perf_counter() - t0
        X, U, s = r2.X, r2.U, r2.s
        tgo = sc.t_dim(s)
        if mag is None:                       # the engine is already lit at handover
            T0, _ = plan_thrust(X, U, s, 0.0, sc)
            mag, d = act.clip_command(T0)

        n_sub = max(1, int(round(a.dt_guid / a.dt_act)))
        h_act = sc.t_nd(a.dt_act)
        alt0 = float(n @ (x[0:3] - rf_nd)) * sc.L
        for i in range(n_sub):
            dt_ahead = sc.t_nd(i * a.dt_act)
            if dt_ahead >= s:
                break
            T_cmd, _ = plan_thrust(X, U, s, dt_ahead, sc)
            mag_cmd, d_cmd = act.clip_command(T_cmd)
            T_vec = mag * d
            # t_s is ALREADY the current time -- it is advanced by the propagation loop
            # below. Adding i*dt_act again double-counts it, so each cycle's samples ran to
            # +119 s of a 60 s cycle and then jumped BACKWARD 59 s at the boundary: 34
            # backward jumps across the descent, which makes every time series discontinuous.
            trace.append(dict(t=t_s, T_cmd=float(mag_cmd), T_act=float(mag),
                              ang_cmd=float(np.degrees(np.arccos(np.clip(d_cmd @ n, -1, 1)))),
                              ang_act=float(np.degrees(np.arccos(np.clip(d @ n, -1, 1)))),
                              alt=float(n @ (x[0:3] - rf_nd)) * sc.L))
            nst = max(1, int(round(a.dt_act / a.dt_prop)))
            hp = h_act / nst
            for _ in range(nst):
                xn = rk4(dyn, x, T_vec, hp, sc)
                if float(n @ (xn[0:3] - rf_nd)) <= 0.0:          # bisect the surface crossing
                    lo, hi = 0.0, hp
                    for _ in range(50):
                        mid = 0.5 * (lo + hi)
                        xm = rk4(dyn, x, T_vec, mid, sc)
                        if float(n @ (xm[0:3] - rf_nd)) <= 0.0:
                            hi, xn = mid, xm
                        else:
                            lo = mid
                    x = xn; landed = True; t_s += sc.t_dim(hi); break
                x = xn; t_s += sc.t_dim(hp)
            if landed:
                break
            mag, d, _ = act.step(mag, d, T_cmd, a.dt_act)
        alt = float(n @ (x[0:3] - rf_nd)) * sc.L
        spd = float(np.linalg.norm(x[3:6])) * sc.v
        print(f"cycle {cyc:3d}: t {t_s:7.1f}s  alt {alt:9.2f} m  |v| {spd:7.4f} m/s  "
              f"|T| cmd {mag_cmd:5.2f} act {mag:5.2f} N  t_go {tgo:7.1f}s  "
              f"solve {wall:5.1f}s ({r1.iters}+{r2.iters} it)", flush=True)
        # t_solve and t_go belong together; t is AFTER the cycle was flown. Pairing the
        # two made the predicted landing time read one whole cycle too late.
        hist.append(dict(cyc=cyc, t_solve=t_solve, t=t_s, alt=alt, speed=spd,
                         mass=float(np.exp(x[6])), t_go=tgo, wall=wall,
                         it1=r1.iters, it2=r2.iters, status=r2.status))
        if landed:
            print(f"\nTOUCHDOWN at t = {t_s:.2f} s"); break
        frac = min(sc.t_nd(a.dt_guid) / s, 0.95)
        warm = shift_plan(X, U, s, frac)

    miss = float(np.linalg.norm(Pl @ (x[0:3] - rf_nd))) * sc.L
    alt = float(n @ (x[0:3] - rf_nd)) * sc.L
    spd = float(np.linalg.norm(x[3:6])) * sc.v
    vert = float(n @ x[3:6]) * sc.v
    ang = float(np.degrees(np.arccos(np.clip(d @ n, -1, 1))))
    print(f"\n{'='*66}\nWHERE IT LANDS\n{'='*66}")
    print(f"  landed            : {landed}")
    print(f"  time of flight    : {t_s:.2f} s")
    print(f"  miss from site    : {miss:.4f} m")
    print(f"  altitude          : {alt:.5f} m")
    print(f"  touchdown speed   : {spd:.4f} m/s   (vertical {vert:+.4f}, limit {C.VTOUCH})")
    print(f"  thrust angle      : {ang:.2f} deg from the local vertical")
    print(f"  propellant        : {m0 - float(np.exp(x[6])):.4f} kg")
    print(f"  cycles            : {len(hist)}, mean solve {np.mean([h['wall'] for h in hist]):.1f} s")
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(landed=bool(landed), t=t_s, miss=miss, alt=alt, speed=spd, vert=vert,
                   ang=ang, fuel=m0 - float(np.exp(x[6])), K=a.K, dt_guid=a.dt_guid,
                   tau=a.tau, slew=a.slew, cycles=hist, trace=trace),
              open(os.path.join(a.out, "closed_loop.json"), "w"), indent=1)
    print(f"\nwrote {a.out}/closed_loop.json")


if __name__ == "__main__":
    main()
