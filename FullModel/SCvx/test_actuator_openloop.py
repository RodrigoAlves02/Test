#!/usr/bin/env python
"""Fly the SAME open-loop plan with and without the actuator, to separate the two error sources.

The open-loop audit in verify.py feeds U[:, k] straight into the dynamics: no lag, no slew limit,
the commanded control delivered perfectly. That is an IDEAL-ACTUATOR number and it is not
comparable with the closed-loop result, which flies through a real actuator. Flown here through
the same actuator model the closed loop uses, on the identical plan, so the only difference between
the two rows is the hardware.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "Guidance and Control"))
import config as C, problem_setup as PS
from closed_loop import plan_thrust, rk4
from actuator_model import Actuator

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "Figures&data")


def fly(dyn, sc, X, U, s, n, rf_nd, act=None, dt_act=0.5, dt_prop=0.25):
    """Follow the plan's thrust profile through the TRUTH dynamics (7 states, thrust external).
    act=None delivers the commanded thrust perfectly."""
    x = X[0:7, 0].copy()
    mag, d = (None, None)
    if act is not None:
        T0, _ = plan_thrust(X, U, s, 0.0, sc)
        mag, d = act.clip_command(T0)             # engine already lit at handover
    tf = sc.t_dim(s)
    n_sub = int(np.floor(tf / dt_act))
    rec = []
    landed = False
    for i in range(n_sub):
        ta = sc.t_nd(i * dt_act)
        if ta >= s:
            break
        T_cmd, _ = plan_thrust(X, U, s, ta, sc)
        mag_cmd, d_cmd = (act.clip_command(T_cmd) if act is not None
                          else (float(np.linalg.norm(T_cmd)),
                                T_cmd / max(np.linalg.norm(T_cmd), 1e-12)))
        T_vec = (mag * d) if act is not None else T_cmd
        rec.append((i * dt_act, mag_cmd, mag_cmd if act is None else mag))
        nst = max(1, int(round(dt_act / dt_prop)))
        hp = sc.t_nd(dt_act) / nst
        for _ in range(nst):
            xn = rk4(dyn, x, T_vec, hp, sc)
            if float(n @ (xn[0:3] - rf_nd)) <= 0.0:
                lo, hi = 0.0, hp
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    xm = rk4(dyn, x, T_vec, mid, sc)
                    if float(n @ (xm[0:3] - rf_nd)) <= 0.0:
                        hi, xn = mid, xm
                    else:
                        lo = mid
                x = xn; landed = True; break
            x = xn
        if landed:
            break
        if act is not None:
            mag, d, _ = act.step(mag, d, T_cmd, dt_act)
    return x, landed, np.array(rec)


def main():
    z = np.load(os.path.join(OUT, "descent_K60.npz"))
    X, U, s = z["X"], z["U"], float(z["s"])
    P = PS.build(); dyn, sc, n = P["dyn"], P["scales"], P["n_hat"]
    rf_nd = P["rf_nd"]; Pl = np.eye(3) - np.outer(n, n)
    m0 = float(np.exp(X[6, 0]))
    print(f"plan: K={X.shape[1]}, tf={sc.t_dim(s):.1f} s, planned fuel "
          f"{m0-float(np.exp(X[6,-1])):.4f} kg\n")
    print(f"{'case':<34}{'miss [m]':>11}{'alt [m]':>10}{'speed [m/s]':>13}"
          f"{'fuel [kg]':>11}{'landed':>8}")
    rows = {}
    for lab, act in (("ideal actuator (perfect)", None),
                     ("real actuator (tau 0.5 s, 2 deg/s)",
                      Actuator(tau=0.5, slew_max_deg_s=2.0, t_min=C.TMIN, t_max=C.TMAX))):
        xe, landed, rec = fly(dyn, sc, X, U, s, n, rf_nd, act=act)
        miss = float(np.linalg.norm(Pl @ (xe[0:3] - rf_nd))) * sc.L
        alt = float(n @ (xe[0:3] - rf_nd)) * sc.L
        spd = float(np.linalg.norm(xe[3:6])) * sc.v
        fu = m0 - float(np.exp(xe[6]))
        print(f"{lab:<34}{miss:>11.4f}{alt:>10.4f}{spd:>13.4f}{fu:>11.4f}{str(landed):>8}")
        rows[lab] = dict(miss=miss, alt=alt, speed=spd, fuel=fu, landed=bool(landed))
    d_act = rows["real actuator (tau 0.5 s, 2 deg/s)"]["miss"] - rows["ideal actuator (perfect)"]["miss"]
    cl = json.load(open(os.path.join(OUT, "closed_loop.json")))
    print(f"\n  open loop, ideal actuator : {rows['ideal actuator (perfect)']['miss']:8.4f} m")
    print(f"  open loop, REAL actuator  : {rows['real actuator (tau 0.5 s, 2 deg/s)']['miss']:8.4f} m"
          f"   <- the actuator alone costs {d_act:+.4f} m")
    print(f"  CLOSED loop, real actuator: {cl['miss']:8.4f} m"
          f"   <- the loop adds a further "
          f"{cl['miss']-rows['real actuator (tau 0.5 s, 2 deg/s)']['miss']:+.4f} m")
    rows["closed_loop"] = dict(miss=cl["miss"], speed=cl["speed"], fuel=cl["fuel"])
    json.dump(rows, open(os.path.join(OUT, "actuator_openloop.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
