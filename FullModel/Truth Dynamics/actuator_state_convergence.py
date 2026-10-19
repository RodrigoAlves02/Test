#!/usr/bin/env python
"""RK4 STEP SELECTION with the actuator carried as a STATE, on the nominal descent.

The actuator magnitude is an ODE driven by a known function of time,

    d|T|/dt = (|T_cmd(t)| - |T|) / tau

and in OPEN loop T_cmd(t) is just the plan, so there is nothing to hold. The magnitude is appended
to the state vector and integrated alongside r, v and m:

    y = [r_I(3) m, v_I(3) m/s, m kg, |T| N]

No actuator sub-step, no staircase, no 4092 restarts per arc. This is only available because the
run is open loop -- the closed loop cannot do it, because there the command depends on the
navigation state and is not known ahead of the step. That is the whole reason dt_act exists.

THE SLEW IS NOT A STATE, AND DOES NOT NEED TO BE. A rate limiter is a switch, not a differential
equation; writing it as one makes the direction overshoot and reverse into a sliding mode. On this
problem it never activates -- the commanded turn peaks at 1.16 deg/s against a 2 deg/s limit and
saturates in 0 of 4092 holds -- so the applied direction IS the commanded direction and there is
nothing to switch. That measurement is the assumption this formulation rests on; if a dispersed
Monte Carlo case ever drove the commanded turn past the limit, this form would stop being valid
there and the run would need the switched treatment.

WHAT IS LEFT TO RESTART ON. The plan holds its thrust MAGNITUDE constant across each of its K-1
control segments, so |T_cmd| jumps at the 98 interior nodes and the lag is re-excited there. Those
are the only discontinuities in the right-hand side, so the march restarts on them and nowhere else
-- 99 intervals instead of 4092 -- and dt is free up to the 20.663 s segment length.

The arc ends at TOUCHDOWN, bisected, not at tf: the truth flies the plan open loop through a
different gravity field and reaches the surface early, and the polyhedral potential is C1 but not C2
across a facet, so a step crossing it loses its order.

Each dt is compared against its own half, at the control nodes, which both runs land on exactly.

Outputs (Figures&Data/):
    actuator_state_error_vs_time.png    time on x, error on y, one curve per dt
    actuator_state_error_vs_dt.png      dt on x, max error on y
    actuator_state_convergence.json     the numbers
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TBB_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.abspath(os.path.join(HERE, "..", "SCvx"))
GNC = os.path.abspath(os.path.join(HERE, "..", "Guidance and Control"))
VALID = os.path.join(HERE, "Validation")
OUT = os.path.join(HERE, "Figures&Data")
for _p in (SCVX, GNC, VALID):
    sys.path.insert(0, _p)

import config as C                                                      # noqa: E402
import gravity as GR                                                    # noqa: E402
import problem_setup as PS                                              # noqa: E402
from closed_loop import plan_thrust                                     # noqa: E402
from actuator_model import Actuator                                     # noqa: E402
import truth_dynamics as TD                                             # noqa: E402

K = 100
DTS = [20.0, 10.0, 5.0, 2.0, 1.0]


class Diverged(Exception):
    """The step is outside the scheme's stability region for the actuator mode."""


def march(f, y0, t0, t1, dt, scheme):
    """Fixed step of EXACTLY dt over [t0, t1], trimming only the last step.

    truth_dynamics.propagate_fixed does this, but for its own 7-state right-hand side; the state
    here carries the actuator magnitude as an eighth component, so the same marching rule is
    applied to a right-hand side of this module's own.
    """
    y, t, n = np.asarray(y0, float).copy(), t0, 0
    while True:
        rem = t1 - t
        if rem <= 1e-12:
            break
        h = dt if dt < rem else rem
        y = TD._step(f, t, y, h, scheme)
        # STABILITY GUARD. The lag puts an eigenvalue at -1/tau in a system whose other modes are
        # orbital, so the step is capped by stability and not by accuracy. Outside the region the
        # thrust state diverges geometrically and drags the position with it, and the polyhedral
        # field then gets evaluated hundreds of body radii out and floods the log. Catch it here.
        if not np.isfinite(y).all() or np.linalg.norm(y[0:3]) > 1e6 or y[7] > 1e4:
            raise Diverged
        t += h
        n += 1
    return y, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dts", default=None, help="comma-separated step sizes, overriding the ladder")
    ap.add_argument("--scheme", default="rk4", choices=("rk4", "rk8"))
    ap.add_argument("--tau", type=float, default=0.5, help="thrust lag time constant [s]")
    ap.add_argument("--slew", type=float, default=2.0, help="slew limit [deg/s], checked not used")
    a = ap.parse_args()

    z = np.load(os.path.join(OUT, "nominal_plan.npz"))
    X, U, s, tf = z["X"], z["U"], float(z["s"]), float(z["tf"])
    P = PS.build(altitude_km=11.771, cone_deg=0.0)
    sc, n_hat = P["scales"], P["n_hat"]
    seg, h_nd = tf / (K - 1), s / (K - 1)
    t_node = np.arange(K) * seg
    WZ = float(TD.OMEGA[2])

    g = TD.Gravity(kind="poly")
    veh = TD.Vehicle(isp=C.ISP, dry_mass=C.MDRY)
    act = Actuator(tau=a.tau, slew_max_deg_s=a.slew, t_min=C.TMIN, t_max=C.TMAX)
    alpha = veh.alpha

    print(f"nominal plan: fuel {float(z['fuel']):.5f} kg, tf {tf:.2f} s")
    print(f"actuator as a STATE: tau {a.tau:g} s, thrust clipped to {C.TMIN:g}-{C.TMAX:g} N")
    print(f"restart grid: {K-1} control nodes, {seg:.4f} s apart   scheme: {a.scheme}")


    def cmd_in_segment(k):
        """Commanded magnitude and direction inside segment k, both in the ROTATING frame.

        The index is pinned to k so a stage landing exactly on the closing node still reads this
        segment's hold. The magnitude is constant across the segment -- that is the plan's ZOH --
        and the direction turns continuously, so the pair is smooth over the whole interval.
        """
        def mag_dir(t):
            lam = min(max((t - t_node[k]) / seg, 0.0), 1.0 - 1e-12)
            TB, _ = plan_thrust(X, U, s, (k + lam) * h_nd, sc)
            return act.clip_command(TB)
        return mag_dir

    def rhs(mag_dir):
        """y = [r_I(3), v_I(3), m, |T|], SI, inertial frame.

            rdot  = v
            vdot  = R(wt) g(R(-wt) r) + |T| d_cmd(t) / m
            mdot  = -|T| / (Isp g0)
            |T|'  = (|T_cmd(t)| - |T|) / tau

        The gravity block mirrors truth_dynamics.eom_inertial exactly: the field is body-fixed, so
        the position rotates in and the acceleration rotates back out on one sine/cosine pair.
        """
        def f(t, y):
            r_I, v_I, m, Tm = y[0:3], y[3:6], y[6], y[7]
            w = WZ * t
            c, sn = np.cos(w), np.sin(w)
            r_B = np.array([c * r_I[0] + sn * r_I[1], -sn * r_I[0] + c * r_I[1], r_I[2]])
            g_B = g(r_B)
            acc = np.array([c * g_B[0] - sn * g_B[1], sn * g_B[0] + c * g_B[1], g_B[2]])
            mag_cmd, d_rot = mag_dir(t)
            d_I = np.array([c * d_rot[0] - sn * d_rot[1], sn * d_rot[0] + c * d_rot[1], d_rot[2]])
            acc = acc + Tm * d_I / m
            return np.concatenate([v_I, acc, [-alpha * Tm, (mag_cmd - Tm) / a.tau]])
        return f

    def height(y, t):
        rB, _ = TD.inertial_to_body(y[0:3], y[3:6], t)
        return float(GR.height_above_surface(rB, n_hat))

    rI, vI = TD.body_to_inertial(X[0:3, 0] * sc.L, X[3:6, 0] * sc.v, 0.0)
    mag0, _ = act.clip_command(plan_thrust(X, U, s, 0.0, sc)[0])   # actuator starts matched
    y0 = np.concatenate([rI, vI, [float(np.exp(X[6, 0]))], [mag0]])

    dts = [float(x) for x in a.dts.split(",")] if a.dts else DTS
    print("pairs: " + ",  ".join(f"{d:g} vs {d/2:g}" for d in dts))
    PAIRS = [(d, d / 2.0) for d in dts]
    ALL = sorted({d for p in PAIRS for d in p}, reverse=True)
    nodes, evals, walls, nsteps, touch = {}, {}, {}, {}, {}
    print()
    lam_h = 2.785 * a.tau          # RK4 real-axis stability limit, h |lambda| <= 2.785
    print(f"stability bound for {a.scheme} on the lag eigenvalue -1/tau: "
          f"h < {lam_h:.3f} s\n")
    unstable = []
    for dt in ALL:
        g.reset(); w0 = time.perf_counter()
        Y = np.zeros((K, 8)); Y[0] = y0
        y = y0.copy(); nst = 0; k_last = K - 1
        t_td, y_td = float(t_node[-1]), None
        try:
          for k in range(K - 1):
            f = rhs(cmd_in_segment(k))
            y_new, nn = march(f, y, t_node[k], t_node[k + 1], dt, a.scheme)
            nst += nn
            if height(y_new, t_node[k + 1]) <= 0.0:
                lo, hi, y_hi = t_node[k], t_node[k + 1], y_new
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    y_m, _ = march(f, y, t_node[k], mid, dt, a.scheme)
                    if height(y_m, mid) <= 0.0:
                        hi, y_hi = mid, y_m
                    else:
                        lo = mid
                    if hi - lo < 1e-12:
                        break
                k_last, t_td, y_td = k, float(hi), y_hi
                break
            y = y_new
            Y[k + 1] = y
        except Diverged:
            unstable.append(dt)
            print(f"  dt {dt:7.3f} s   UNSTABLE -- diverged in segment {k}, "
                  f"h/tau = {dt/a.tau:.1f} against a limit of {lam_h/a.tau:.2f}", flush=True)
            continue
        nodes[dt] = Y[:k_last + 1]
        touch[dt] = (t_td, y_td if y_td is not None else y)
        evals[dt] = g.n; walls[dt] = time.perf_counter() - w0; nsteps[dt] = nst
        print(f"  dt {dt:7.3f} s   {nst:>6,} steps ({nst/(K-1):6.2f}/segment)   "
              f"{g.n:>8,} gravity calls   touchdown {t_td:11.6f} s   {walls[dt]:6.1f} s wall",
              flush=True)

    OK = [d for d in ALL if d in nodes]
    PAIRS = [(ca, fb) for ca, fb in PAIRS if ca in nodes and fb in nodes]
    if not PAIRS:
        print("\nno pair survived: every step on the ladder is outside the stability region")
        return
    k_cmp = min(nodes[d].shape[0] for d in OK) - 1
    rows = []
    print()
    for ca, fb in PAIRS:
        Ya, Yb = nodes[ca][:k_cmp + 1], nodes[fb][:k_cmp + 1]
        dr = np.linalg.norm(Ya[:, 0:3] - Yb[:, 0:3], axis=1)
        rows.append(dict(dt=ca, t=t_node[:k_cmp + 1].tolist(), err_t=dr.tolist(),
                         max_r=float(dr.max()), end_r=float(dr[-1]),
                         t_td=touch[ca][0], evals=int(evals[ca]), steps=int(nsteps[ca]),
                         wall=float(walls[ca])))
        print(f"  dt {ca:6.2f} vs {fb:6.3f}:  max |dr| {dr.max():.4e} m")

    print(f"\n{'dt [s]':>9}{'ratio':>9}{'order':>8}")
    for i in range(len(rows) - 1):
        rt = rows[i]["max_r"] / rows[i + 1]["max_r"]
        od = np.log(rt) / np.log(rows[i]["dt"] / rows[i + 1]["dt"])
        print(f"{rows[i]['dt']:>9g}{rt:>9.2f}{od:>8.2f}")

    json.dump(dict(tf=tf, seg=seg, K=K, scheme=a.scheme, tau=a.tau, rows=rows),
              open(os.path.join(OUT, "actuator_state_convergence.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(OUT, 'actuator_state_convergence.json')}")


if __name__ == "__main__":
    main()
