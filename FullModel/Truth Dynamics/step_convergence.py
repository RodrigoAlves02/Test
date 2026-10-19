#!/usr/bin/env python
"""TRUTH PROPAGATOR STEP SELECTION, on the nominal descent.

Takes the nominal guidance solution (handover 11.771 km, altitude alignment corridor -- the green
trace in SCvx/Figures&data/align_corridor.png) and propagates its thrust history through the
reference model: fixed-step RK8, INERTIAL frame, Werner-Scheeres polyhedral gravity.

No reference trajectory is used. The arc is propagated at dt and again at dt/2, and the difference
between the two is the error estimate at dt. For a scheme of order p the finer run carries 2^-p of
the coarser run's error, so their difference is (1 - 2^-p) of it. The ladder costs nothing extra
because every propagation is the fine partner of the one above and the coarse partner of the one
below. Each reported step is paired with its OWN half. Ten propagations, seven pairs.

TWO MODES.

  PLAN (default) -- flies the plan's own thrust profile: magnitude zero-order held across each of
  the K-1 control segments, direction turning continuously inside a segment as the plan's own
  kinematics say it does. This is the physics the SCvx subproblem actually solved.

  ACTUATOR (--actuator) -- flies what the engine delivers. The command is sampled and held every
  dt_act, clipped to [t_min, t_max], chased by a first-order lag of time constant tau, and turned
  by a rate limiter at slew_max. This is what the vehicle flies, so it is the step size that has to
  be defensible. The actuator state is advanced in CLOSED FORM at the chunk boundaries, so it is
  bit-identical between a dt and its half and the comparison isolates the integration error alone.

THE MARCH RESTARTS ON EVERY DISCONTINUITY OF THE RIGHT-HAND SIDE. In plan mode those are the K-1
control nodes, where the held magnitude jumps; in actuator mode they are the dt_act chunk edges,
where the applied thrust vector is refreshed. A step that spans one integrates the wrong
acceleration over part of itself, and no amount of order in the tableau recovers that: the local
error is first order in the step and its sign depends on where the jump falls inside the step, so
the ladder comes out non-monotone and the measurement says nothing about the scheme. Marching each
interval separately -- the state carries through, only the step boundary is forced -- makes the
right-hand side smooth over every step and lets the scheme show its actual order. Inside an interval
the step is exactly dt with only the final step trimmed to land on the edge, which is what
truth_dynamics.propagate_fixed does in flight.

In plan mode the plan's segment index is additionally PINNED to the segment being integrated: a
stage landing exactly on the closing node would otherwise floor into the next segment and read the
next hold's magnitude, putting a jump back inside a step that was restarted precisely to avoid one.

THE ARC ENDS AT TOUCHDOWN, NOT AT tf. The plan lands on its own tangent plane, but the truth flies
it open loop through a different gravity field and arrives somewhere else: at tf it is 2.03 m INSIDE
the body. Integrating those last metres is not just meaningless, it is the single largest error in
the study -- the polyhedral potential is C1 across a facet but not C2, its gradient jumping by
4*pi*G*rho, so a step that crosses the surface loses its order exactly as a step across a thrust
jump does. Left in, that one crossing contributed 4.5e-6 m at dt = 10 s against the 1e-10 m the rest
of the arc is worth, and the ladder measured the crossing rather than the scheme. The march
therefore stops at h = 0, bisected to 1e-12 s, which is what the flight loop does.

THE COMPARISON NEEDS NO INTERPOLATION. The sample times are common to every step size on the
ladder, so the error is read directly with both runs landing on them exactly. The last sample above
the surface closes the curve; the touchdown itself is reported separately, as the difference in
landing time and landing point between the two runs, which is the quantity the step size has to be
small enough not to move.

Outputs (Figures&Data/), suffixed _act in actuator mode:
    step_error_vs_time.png    error against time, one curve per dt
    step_error_vs_dt.png      max error against dt, with the fitted slope
    step_convergence.json     the numbers
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
import ptr                                                              # noqa: E402
from dynamics import Dynamics                                           # noqa: E402
from closed_loop import plan_thrust                                     # noqa: E402
from actuator_model import Actuator                                     # noqa: E402
import truth_dynamics as TD                                             # noqa: E402

K = 100
TOL = dict(iters=200, tol_obj=1e-7, tol_defect_m=1e-3, tol_defect_v=1e-4, tol_vc_m=1e-4)
ALIGN = dict(align_theta0_deg=45.0, align_h_m=100.0)

# The step sizes being assessed. Plan mode is bounded above by the 20.663 s control segment;
# actuator mode is bounded above by dt_act, because a step longer than the hold would span a
# refresh of the applied thrust.
DTS_PLAN = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
DTS_ACT = [0.5, 0.2, 0.1, 0.05, 0.025]
# RK4 falls four orders per decade of step rather than eight, so the interesting rungs sit lower.
DTS_PLAN_RK4 = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05]

BISECT_TOL_S = 1e-12   # the signal is ~1e-10 m and contact speed ~0.3 m/s; 1e-9 s would add noise
BISECT_MAX = 60


def applied_thrust_exact(t_i, h, mag_i, d_i, mag_cmd, d_cmd, tau, slew_max):
    """The thrust the engine ACTUALLY applies through one hold, as a function of time.

    Both actuator channels already have closed forms over a zero-order-hold interval, so there is no
    reason to show the integrator a frozen sample of a profile that is known exactly. Actuator.step
    evaluates these same two expressions -- but only at the far end of the hold, to hand on the next
    state, after which the propagation throws the shape away and applies the value from the near end
    across the whole interval.

    MAGNITUDE. With the command constant across the hold, d|T|/dt = (|T_cmd| - |T|)/tau integrates to

        mag(t) = mag_cmd + (mag_i - mag_cmd) exp(-(t - t_i)/tau)

    At the flight setting h = tau = 0.5 s the true magnitude covers 63 % of its gap INSIDE one stair,
    so holding the near-end value is wrong by 0.368*(mag_cmd - mag_i) averaged over the hold -- about
    4.4 N for half a second at a throttle switch, where the gap reaches 11.85 N.

    DIRECTION. The rate limiter is a constant-rate turn about the fixed common normal that stops on
    arrival, so with k the unit normal and theta_tot the angle between the two directions,

        theta(t) = min(theta_tot, slew_max (t - t_i))
        d(t)     = d_i cos(theta) + (k x d_i) sin(theta) + k (k . d_i)(1 - cos(theta))

    THE CORNER. theta(t) is only piecewise smooth: it has a derivative discontinuity where the turn
    completes, at t* = t_i + min(h, theta_tot/slew_max). On this problem the slew never saturates --
    theta_tot averages 0.09 deg against a 2 deg/s limit -- so t* lands about 0.045 s into a 0.5 s hold
    and the direction is constant after it. A step spanning t* loses its order for exactly the reason
    a step spanning a thrust jump or the surface does, so the caller is handed t* and must restart
    there. That is the whole cost of this: one extra restart per hold, no extra stages.

    Returns (T_rot(t), t*): the applied thrust in the ROTATING frame, and the corner to restart on.
    """
    c = float(np.clip(float(d_i @ d_cmd), -1.0, 1.0))
    th_tot = float(np.arccos(c))
    axis = np.cross(d_i, d_cmd)
    an = float(np.linalg.norm(axis))
    if th_tot <= 1e-12 or an < 1e-12 or slew_max <= 0.0:
        k_ax, th_tot = None, 0.0
    else:
        k_ax = axis / an
    t_star = t_i + (min(h, th_tot / slew_max) if th_tot > 0.0 else 0.0)

    def T_rot(t):
        dtl = max(t - t_i, 0.0)
        mag = mag_cmd + (mag_i - mag_cmd) * np.exp(-dtl / tau)
        if k_ax is None:
            return mag * d_i
        th = min(th_tot, slew_max * dtl)
        cs, sn = np.cos(th), np.sin(th)
        d = d_i * cs + np.cross(k_ax, d_i) * sn + k_ax * float(k_ax @ d_i) * (1.0 - cs)
        return mag * d

    return T_rot, t_star


def nominal_plan():
    """The green trace: nominal handover, altitude alignment corridor, PINN plan gravity."""
    P = PS.build(altitude_km=11.771, cone_deg=0.0)
    sc = P["scales"]
    dyn = Dynamics(sc, field=GR.pinn_field())
    cm = dict(dyn=dyn, scales=sc, x_init=P["x_init"], rf_nd=P["rf_nd"], n_hat=P["n_hat"],
              K=K, verbose=False, slew_max_deg_s=C.SLEW_MAX_DEG_S,
              term_point_deg=C.TERMINAL_POINT_DEG, term_point_nodes=C.TERMINAL_POINT_NODES,
              **ALIGN, **TOL)
    r1 = ptr.run(mode="error", warm=None, **cm)
    r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + 0.1 / sc.L, warm=(r1.X, r1.U, r1.s), **cm)
    return P, sc, r2


def load_plan():
    """The plan is an input to this study, not part of it -- cache it."""
    pf = os.path.join(OUT, "nominal_plan.npz")
    if os.path.exists(pf):
        z = np.load(pf)
        X, U, s, tf = z["X"], z["U"], float(z["s"]), float(z["tf"])
        sc = PS.build(altitude_km=11.771, cone_deg=0.0)["scales"]
        print(f"nominal plan: loaded from cache, fuel {float(z['fuel']):.5f} kg, tf {tf:.2f} s")
        return X, U, s, tf, sc, float(z["fuel"]), "cached"
    t0 = time.perf_counter()
    _P, sc, r = nominal_plan()
    X, U, s, tf = r.X, r.U, float(r.s), float(r.tf_s)
    np.savez_compressed(pf, X=X, U=U, s=s, tf=tf, fuel=float(r.fuel_kg))
    print(f"nominal plan: {r.status} in {r.iters} it, fuel {r.fuel_kg:.5f} kg, tf {tf:.2f} s "
          f"[{time.perf_counter()-t0:.0f}s]")
    return X, U, s, tf, sc, float(r.fuel_kg), str(r.status)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--actuator", action="store_true",
                    help="fly what the engine delivers (lag + slew + clip, held every dt_act)")
    ap.add_argument("--dt-act", dest="dt_act", type=float, default=0.5,
                    help="actuator sub-step / command hold [s]; the flight default is 0.5")
    ap.add_argument("--tau", type=float, default=0.5, help="thrust lag time constant [s]")
    ap.add_argument("--slew", type=float, default=2.0, help="slew rate limit [deg/s]")
    ap.add_argument("--lag-exact", dest="lag_exact", action="store_true",
                    help="show the integrator the true in-hold thrust profile instead of a frozen "
                         "sample of it; actuator mode only")
    ap.add_argument("--tie-act", dest="tie_act", action="store_true",
                    help="refresh the actuator at the INTEGRATOR step, i.e. dt_act = dt, so that "
                         "refining the ladder refines the actuator staircase as well. Without it "
                         "dt_act is frozen and every step at or above it collapses onto one step "
                         "per hold, which converges nothing.")
    ap.add_argument("--dts", default=None,
                    help="comma-separated step sizes to assess, overriding the built-in ladder; "
                         "each is still paired with its own half")
    ap.add_argument("--scheme", default="rk8", choices=("rk8", "rk4"),
                    help="integration scheme under test; the reference model is rk8")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if a.lag_exact and not a.actuator:
        ap.error("--lag-exact is only meaningful with --actuator")
    tag = ("_act" if a.actuator else "") + ("x" if a.lag_exact else "") \
          + ("" if a.scheme == "rk8" else "_" + a.scheme)
    X, U, s, tf, sc, fuel, status = load_plan()

    P = PS.build(altitude_km=11.771, cone_deg=0.0)
    n_hat = P["n_hat"]
    seg = tf / (K - 1)                       # control segment, seconds
    h_nd = s / (K - 1)                       # the same, non-dimensional
    WZ = float(TD.OMEGA[2])                  # truth_dynamics exposes OMEGA, not WZ

    # The grid the march is forced to land on: control nodes in plan mode, command holds in
    # actuator mode. The last interval is short, closing exactly on tf.
    if a.actuator:
        n_hold = int(np.ceil(tf / a.dt_act))
        edges = np.minimum(np.arange(n_hold + 1) * a.dt_act, tf)
    else:
        edges = np.arange(K) * seg
    DTS = DTS_ACT if a.actuator else DTS_PLAN
    if a.scheme == "rk4" and not a.actuator:
        DTS = DTS_PLAN_RK4
    if a.dts:
        DTS = [float(x) for x in a.dts.split(",")]
    PAIRS = [(d, d / 2.0) for d in DTS]
    ALL = sorted({d for p in PAIRS for d in p}, reverse=True)

    print(f"mode: {'ACTUATOR' if a.actuator else 'PLAN'}   tf {tf:.2f} s   "
          f"control segment {seg:.4f} s")
    print(f"restart grid: {len(edges)-1} intervals of "
          f"{a.dt_act if a.actuator else seg:g} s")
    if a.actuator:
        print(f"actuator: tau {a.tau:g} s, slew {a.slew:g} deg/s, "
              f"thrust {C.TMIN:g}-{C.TMAX:g} N, command held every {a.dt_act:g} s")
        print(f"in-hold thrust: {'EXACT (closed-form lag and turn, restart at the corner)' if a.lag_exact else 'frozen at the hold epoch (flight convention)'}")
    print("pairs: " + ",  ".join(f"{d:g} vs {d/2:g}" for d in DTS))
    print(f"{len(ALL)} propagations: " + ", ".join(f"{d:g}" for d in ALL))

    g = TD.Gravity(kind="poly")
    veh = TD.Vehicle(isp=C.ISP, dry_mass=C.MDRY)
    print(f"gravity: polyhedral, {g.faces:,} facets   frame: inertial   "
          f"scheme: {a.scheme}\n")

    def rot_to_inertial(T_rot, t):
        cw, sw = np.cos(WZ * t), np.sin(WZ * t)
        return np.array([cw * T_rot[0] - sw * T_rot[1], sw * T_rot[0] + cw * T_rot[1], T_rot[2]])

    def plan_cmd(t):
        """The thrust the plan asks for at time t, in the ROTATING frame. Newtons."""
        TB, _ = plan_thrust(X, U, s, min(sc.t_nd(t), s * (1 - 1e-12)), sc)
        return TB

    def thrust_plan_segment(k):
        """Plan thrust through segment k, inertial. The segment index is pinned to k."""
        def T(t):
            lam = min(max((t - edges[k]) / seg, 0.0), 1.0 - 1e-12)
            TB, _ = plan_thrust(X, U, s, (k + lam) * h_nd, sc)
            return rot_to_inertial(TB, t)
        return T

    def height(y, t):
        """Height above the REAL shape [m] from an INERTIAL state. Negative = inside the body."""
        rB, _ = TD.inertial_to_body(y[0:3], y[3:6], t)
        return float(GR.height_above_surface(rB, n_hat))

    # initial state: nd rotating -> SI inertial
    rI, vI = TD.body_to_inertial(X[0:3, 0] * sc.L, X[3:6, 0] * sc.v, 0.0)
    y0 = np.concatenate([rI, vI, [float(np.exp(X[6, 0]))]])

    # TIED MODE: the actuator is refreshed at the integrator step, so the hold grid changes with
    # every rung. The runs are then compared on the COARSEST rung's grid, which every finer grid
    # contains as long as the ladder halves.
    TIE = bool(a.tie_act and a.actuator)
    base_dt = max(ALL)
    if TIE:
        nb = int(np.ceil(tf / base_dt))
        cmp_edges = np.minimum(np.arange(nb + 1) * base_dt, tf)
    else:
        cmp_edges = edges

    samples, evals, walls, nsteps, touch = {}, {}, {}, {}, {}
    for dt in ALL:
        if TIE:
            n_hold = int(np.ceil(tf / dt))
            edges = np.minimum(np.arange(n_hold + 1) * dt, tf)
            stride = int(round(base_dt / dt))
            if abs(base_dt / dt - stride) > 1e-9:
                raise SystemExit(f"--tie-act needs a halving ladder; {base_dt} / {dt} is not whole")
        else:
            stride = 1
        g.reset(); w0 = time.perf_counter()
        act = Actuator(tau=a.tau, slew_max_deg_s=a.slew, t_min=C.TMIN, t_max=C.TMAX)
        # The actuator starts matched to the first command, as the flight loop does.
        mag, d_app = act.clip_command(plan_cmd(0.0)) if a.actuator else (None, None)

        Y = np.zeros((len(edges), 7)); Y[0] = y0
        y = y0.copy(); nst = 0; i_last = len(edges) - 1
        t_td, y_td = float(edges[-1]), None

        def fly(y_in, t0, t1, T_fn, split=None):
            """Integrate [t0, t1] at dt, restarting at `split` when it falls strictly inside."""
            if split is not None and t0 < split < t1:
                ya, na = TD.propagate_fixed(y_in, t0, split, dt, a.scheme,
                                            g, T_fn, veh, frame="inertial")
                yb, nb = TD.propagate_fixed(ya, split, t1, dt, a.scheme,
                                            g, T_fn, veh, frame="inertial")
                return yb, na + nb
            return TD.propagate_fixed(y_in, t0, t1, dt, a.scheme,
                                      g, T_fn, veh, frame="inertial")

        for i in range(len(edges) - 1):
            split = None
            if a.actuator:
                h_i = edges[i + 1] - edges[i]
                mag_cmd, d_cmd = act.clip_command(plan_cmd(edges[i]))
                if a.lag_exact:
                    # The integrator is shown the true profile: the magnitude relaxing toward the
                    # command and the direction turning at the rate limit, both in closed form, with
                    # the frame rotation evaluated at t rather than frozen at the hold epoch.
                    T_rot, split = applied_thrust_exact(edges[i], h_i, mag, d_app, mag_cmd, d_cmd,
                                                        a.tau, np.radians(a.slew))
                    T_k = lambda t, _f=T_rot: rot_to_inertial(_f(t), t)
                else:
                    # APPLIED thrust across this hold is the actuator state at its START, frozen in
                    # the rotating frame and rotated into inertial once -- closed_loop.py:776, 908.
                    T_I = rot_to_inertial(mag * d_app, edges[i])
                    T_k = lambda t, _T=T_I: _T
            else:
                T_k = thrust_plan_segment(i)

            y_new, nn = fly(y, edges[i], edges[i + 1], T_k, split)
            nst += nn

            if height(y_new, edges[i + 1]) <= 0.0:
                # TOUCHDOWN INSIDE THIS INTERVAL. A fixed-step method has no event root-finder, so
                # the crossing is bracketed and bisected by re-flying the interval from its own
                # start to each trial time, at the same dt.
                lo, hi, y_hi = edges[i], edges[i + 1], y_new
                for _ in range(BISECT_MAX):
                    mid = 0.5 * (lo + hi)
                    y_m, _ = fly(y, edges[i], mid, T_k, split)
                    if height(y_m, mid) <= 0.0:
                        hi, y_hi = mid, y_m
                    else:
                        lo = mid
                    if hi - lo < BISECT_TOL_S:
                        break
                i_last, t_td, y_td = i, float(hi), y_hi
                break

            y = y_new
            Y[i + 1] = y
            if a.actuator:
                # The actuator state recursion is UNTOUCHED by --lag-exact: the closed forms above
                # evaluated at t_i + h are exactly what Actuator.step returns, so the hold-to-hold
                # sequence, the clipping and the slew logic are the same model either way. Only what
                # the integrator sees inside a hold changes.
                mag, d_app, _ = act.step(mag, d_app, plan_cmd(edges[i]), h_i)

        samples[dt] = Y[:i_last + 1][::stride]
        touch[dt] = (t_td, y_td if y_td is not None else y)
        evals[dt] = g.n; walls[dt] = time.perf_counter() - w0; nsteps[dt] = nst
        r_td = TD.inertial_to_body(touch[dt][1][0:3], touch[dt][1][3:6], t_td)[0]
        print(f"  dt {dt:8.4f} s   {nst:>8,} steps   {g.n:>10,} gravity calls   "
              f"landing [{r_td[0]:11.4f} {r_td[1]:11.4f} {r_td[2]:11.4f}] m   "
              f"{walls[dt]:7.1f} s wall", flush=True)

    # CONVERGENCE IS JUDGED ON THE LANDING POSITION, against the finest step run. The touchdown
    # TIME is only a proxy: two steps can reach the surface at the same instant by different routes,
    # and it is where the vehicle lands that the mission cares about.
    d_fin = min(ALL)
    r_fin = TD.inertial_to_body(touch[d_fin][1][0:3], touch[d_fin][1][3:6], touch[d_fin][0])[0]
    print(f"\nlanding position against the finest step, dt = {d_fin:g} s\n")
    print(f"  {'dt [s]':>10} {'|dr| [m]':>14} {'order':>7} {'gravity calls':>15}")
    prev = None
    for d in sorted(ALL, reverse=True):
        r_d = TD.inertial_to_body(touch[d][1][0:3], touch[d][1][3:6], touch[d][0])[0]
        e = float(np.linalg.norm(r_d - r_fin))
        o = "" if (prev is None or e <= 0 or prev <= 0) else f"{np.log2(prev / e):>7.2f}"
        print(f"  {d:>10.5f} {e:>14.3e} {o:>7} {evals[d]:>15,}")
        prev = e if e > 0 else prev
    print()

    i_cmp = min(samples[d].shape[0] for d in ALL) - 1
    print(f"\ncomparison runs to t = {cmp_edges[i_cmp]:.3f} s "
          f"({tf - cmp_edges[i_cmp]:.3f} s before tf); touchdown reported separately\n")

    rows = []
    for ca, fb in PAIRS:                                # ca coarse, fb = ca/2 fine
        Ya, Yb = samples[ca][:i_cmp + 1], samples[fb][:i_cmp + 1]
        dr = np.linalg.norm(Ya[:, 0:3] - Yb[:, 0:3], axis=1)
        dv = np.linalg.norm(Ya[:, 3:6] - Yb[:, 3:6], axis=1)
        (ta, ya), (tb, yb) = touch[ca], touch[fb]
        d_ttd = abs(ta - tb)                               # how far the step moves the landing...
        d_rtd = float(np.linalg.norm(ya[0:3] - yb[0:3]))   # ...in time, and in the landing point
        rows.append(dict(dt=ca, t=cmp_edges[:i_cmp + 1].tolist(),
                         err_t=dr.tolist(), errv_t=dv.tolist(),
                         max_r=float(dr.max()), max_v=float(dv.max()),
                         end_r=float(dr[-1]), end_v=float(dv[-1]),
                         t_td=ta, d_t_td=d_ttd, d_r_td=d_rtd,
                         evals=int(evals[ca]), wall=float(walls[ca]), steps=int(nsteps[ca])))
        print(f"  dt {ca:7.3f} vs {fb:7.4f}:  max |dr| {dr.max():.4e} m   "
              f"max |dv| {dv.max():.4e} m/s   touchdown: dt {d_ttd:.3e} s, dr {d_rtd:.4e} m")

    jf = os.path.join(OUT, f"step_convergence{tag}.json")
    json.dump(dict(tf=tf, seg=seg, t_end=float(edges[i_cmp]), fuel=fuel, status=status, K=K,
                   mode="actuator" if a.actuator else "plan", scheme=a.scheme,
                   lag_exact=bool(a.lag_exact),
                   dt_act=a.dt_act if a.actuator else None,
                   tau=a.tau if a.actuator else None, slew=a.slew if a.actuator else None,
                   rows=rows), open(jf, "w"), indent=1)
    print(f"\nwrote {jf}")
    return rows


if __name__ == "__main__":
    main()
